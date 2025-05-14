from fastapi import FastAPI
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app, gpu

# use mid-tier GPU here
gpu_config = gpu.A10G(count=1)

app = App("evodiff")
web_app = FastAPI()

image = (
    Image.debian_slim(python_version="3.9")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install("evodiff")
)
with image.imports():
    import os, glob, pathlib, torch, numpy as np
    from tqdm import tqdm

    # --- EvoDiff / helper imports -------------------------------------------------
    from evodiff.utils import Tokenizer
    from evodiff.plot import aa_reconstruction_parity_plot
    from evodiff.pretrained import (
        CARP_38M, CARP_640M, D3PM_BLOSUM_38M, D3PM_BLOSUM_640M,
        D3PM_UNIFORM_38M, D3PM_UNIFORM_640M, OA_DM_38M, OA_DM_640M,
        LR_AR_38M, LR_AR_640M, ESM1b_650M
    )
    from sequence_models.datasets import UniRefDataset


    def generate_random_seq(seq_len, train_prob_dist, tokenizer=Tokenizer()):
        """
        Generates a set of random sequences drawn from a train distribution
        """
        sample = torch.multinomial(torch.tensor(train_prob_dist), num_samples=seq_len, replacement=True)
        sample = sample.to(torch.long)
        return tokenizer.untokenize(sample)


    def generate_oaardm(model, tokenizer, seq_len, penalty=None, batch_size=3, device='cuda'):
        # Generate a random start string and convert to tokens
        all_aas = tokenizer.all_aas
        mask = tokenizer.mask_id

        # Start from mask
        sample = torch.zeros((batch_size, seq_len)) + mask
        sample = sample.to(torch.long)
        sample = sample.to(device)

        # Unmask 1 loc at a time randomly
        loc = np.arange(seq_len)
        np.random.shuffle(loc)
        with torch.no_grad():
            for i in tqdm(loc):
                timestep = torch.tensor([0] * batch_size)  # placeholder but not called in model
                timestep = timestep.to(device)
                prediction = model(sample,
                                   timestep)  # , input_mask=input_mask.unsqueeze(-1)) #sample prediction given input
                p = prediction[:, i,
                    :len(all_aas) - 6]  # sample at location i (random), dont let it predict non-standard AA
                p = torch.nn.functional.softmax(p, dim=1)  # softmax over categorical probs
                p_sample = torch.multinomial(p, num_samples=1)
                # Repetition penalty
                if penalty is not None:  # ignore if value is None
                    for j in range(batch_size):  # iterate over each obj in batch
                        case1 = (i == 0 and sample[j, i + 1] == p_sample[j])  # beginning of seq
                        case2 = (i == seq_len - 1 and sample[j, i - 1] == p_sample[j])  # end of seq
                        case3 = ((i < seq_len - 1 and i > 0) and ((sample[j, i - 1] == p_sample[j]) or (
                                sample[j, i + 1] == p_sample[j])))  # middle of seq
                        if case1 or case2 or case3:
                            # print("identified repeat", p_sample, sample[i-1], sample[i+1])
                            p[j, int(p_sample[j])] /= penalty  # reduce prob of that token by penalty value
                            p_sample[j] = torch.multinomial(p[j], num_samples=1)  # resample
                sample[:, i] = p_sample.squeeze()
                # print([tokenizer.untokenize(s) for s in sample]) # check that sampling correctly
        # print("final seq", [tokenizer.untokenize(s) for s in sample])
        untokenized = [tokenizer.untokenize(s) for s in sample]
        return sample, untokenized


    def generate_autoreg(model, tokenizer, samples=100, batch_size=1, max_seq_len=1024):
        # Generates 1 seq at a time, no batching, to make it easier to deal w variable seq lengths
        # Generates until max length or until stop token is predicted
        # model.eval().cuda()
        device = model.device()

        start = tokenizer.start_id
        stop = tokenizer.stop_id
        sample_out = []
        untokenized_out = []
        timestep = torch.tensor([0] * batch_size)  # placeholder but not called in model
        timestep = timestep.to(device)
        for s in tqdm(range(samples)):
            # Start from START token
            sample = (torch.zeros((1)) + start).unsqueeze(0)  # add batch dim
            sample = sample.to(torch.long)
            sample = sample.to(device)
            # Iterate over each residue until desired length
            # max_loc = np.arange(max_seq_len)
            reach_stop = False  # initialize
            with torch.no_grad():
                for i in range(max_seq_len):
                    if reach_stop == False:  # Add residues until it predicts STOP token or hits max seq len
                        prediction = model(sample,
                                           timestep)  # , input_mask=input_mask.unsqueeze(-1)) #sample prediction given input
                        p = prediction[:, -1, :]  # predict next token
                        p = torch.nn.functional.softmax(p, dim=1)  # softmax over categorical probs
                        p_sample = torch.multinomial(p, num_samples=1)
                        sample = torch.cat((sample, p_sample), dim=1)
                        # print(tokenizer.untokenize(sample[0]))
                        # print(p_sample, stop)
                        if p_sample == stop:
                            reach_stop = True
                    else:
                        break

            print("final seq", tokenizer.untokenize(sample[0, 1:-1]))  # dont save start/stop tokens
            untokenized = tokenizer.untokenize(sample[0, 1:-1])
            sample_out.append(sample[0, 1:-1])
            untokenized_out.append(untokenized)
        return sample_out, untokenized_out


    def generate_valid_subset(data_valid, samples=20):
        sample = []
        for i in tqdm(range(samples)):
            r_idx = np.random.choice(len(data_valid))
            sequence = data_valid[r_idx][0]
            sample.append(sequence)
        print(sample)


    def generate_d3pm(model, tokenizer, Q, Q_bar, timesteps, seq_len, batch_size=3, device='cuda'):
        """
        Generate a random start string from uniform dist and convert to predictions
        """
        # model.eval()
        # device = model.device()

        sample = torch.randint(0, tokenizer.K, (batch_size, seq_len))
        sample = sample.to(torch.long)
        sample = sample.to(device)
        Q = Q.to(device)
        Q_bar = Q_bar.to(device)

        timesteps = torch.linspace(timesteps - 1, 1, int((timesteps - 1) / 1),
                                   dtype=int)  # iterate over reverse timesteps
        timesteps = timesteps.to(device)
        with torch.no_grad():
            for t in tqdm(timesteps):
                timesteps = torch.tensor([t] * batch_size)
                timesteps = timesteps.to(device)
                prediction = model(sample, timesteps)
                p = prediction[:, :, :tokenizer.K]  # p_theta_tilde (x_0_tilde | x_t) # Don't predict non-standard AAs
                p = torch.nn.functional.softmax(p, dim=-1)  # softmax over categorical probs
                p = p.to(torch.float64)
                x_tminus1 = sample.clone()
                for i, s in enumerate(sample):
                    x_t_b = tokenizer.one_hot(s)
                    A = torch.mm(x_t_b, torch.t(Q[t]))  # [P x K]
                    Q_expand = Q_bar[t - 1].unsqueeze(0).expand(A.shape[0], tokenizer.K, tokenizer.K)  # [ P x K x K]
                    B_pred = torch.mul(p[i].unsqueeze(2), Q_expand)
                    q_t = torch.mul(A.unsqueeze(1), B_pred)  # [ P x K x K ]
                    p_theta_marg = torch.bmm(torch.transpose(q_t, 1, 2),
                                             p[i].unsqueeze(2)).squeeze()  # this marginalizes over dim=2
                    p_theta_marg = p_theta_marg / p_theta_marg.sum(axis=1, keepdim=True)
                    x_tminus1[i] = torch.multinomial(p_theta_marg, num_samples=1).squeeze()
                    # On final timestep pick next best from standard AA
                    if t == 1:
                        x_tminus1[i] = torch.multinomial(p_theta_marg[:, :tokenizer.K - 6], num_samples=1).squeeze()
                    # diff = torch.ne(s, x_tminus1[i])
                    # if t % 100 == 0:
                    #     print("time", t, diff.sum().item(), "mutations", tokenizer.untokenize(x_tminus1[i]), "sample", tokenizer.untokenize(s))
                sample = x_tminus1

        untokenized = [tokenizer.untokenize(s) for s in sample]
        print("final seq", untokenized)
        return sample, untokenized


    def get_pretrained(model_type: str):
        """Return (model, collater, tokenizer, scheme[, timestep, Q̄, Q])."""
        if model_type == "esm1b_650M":            return ESM1b_650M()
        if model_type == "carp_38M":              return CARP_38M()
        if model_type == "carp_640M":             return CARP_640M()
        if model_type == "oa_dm_38M":             return OA_DM_38M()
        if model_type == "oa_dm_640M":            return OA_DM_640M()
        if model_type == "lr_ar_38M":             return LR_AR_38M()
        if model_type == "lr_ar_640M":            return LR_AR_640M()
        if model_type == "d3pm_blosum_38M":       return D3PM_BLOSUM_38M(return_all=True)
        if model_type == "d3pm_blosum_640M":      return D3PM_BLOSUM_640M(return_all=True)
        if model_type == "d3pm_uniform_38M":      return D3PM_UNIFORM_38M(return_all=True)
        if model_type == "d3pm_uniform_640M":     return D3PM_UNIFORM_640M(return_all=True)

        raise ValueError(
            f"Unknown model_type: {model_type}. "
            "Choose from carp_38M carp_640M esm1b_650M "
            "oa_dm_38M oa_dm_640M lr_ar_38M lr_ar_640M "
            "d3pm_blosum_38M d3pm_blosum_640M d3pm_uniform_38M d3pm_uniform_640M."
        )


@app.function(
    image=image,
    gpu=gpu_config,
    concurrency_limit=5,
)
def evodiff_generate(
        *,
        # —–––––––––––––––––––––  things you *might* want to touch  ––––––––––––––—
        model_type: str = "oa_dm_640M",
        num_seqs: int = 20,
        scheme: str | None = None,  # "mask", "causal-mask", "d3pm", …
        penalty: float | None = None,
        device: str | int | None = None,  # default ⇒ first visible GPU
        # —–––––––––––––––––––––  rarely-changed plumbing defaults  ––––––––––––––—
        random_baseline: bool = False,
        delete_prev: bool = False,
        count: int = 0,
        output_root: str | None = None,  # None ⇒ ~/Desktop/DMs/
        # —––––––––––––––––––––––  your task-specific inputs  ––––––––––––––––––––—
        seed_sequences: list[str] | None = None,
        seed_structure: str | None = None,  # pdb / mmCIF path  (not yet used)
        seed_msa: str | None = None,  # .a3m / .sto path (not yet used)
) -> list[str]:
    """
    High-level one-liner for EvoDiff sequence generation.

    Parameters
    ----------
    model_type : str, optional
        Which pretrained checkpoint to use.  See list above.  Default: "oa_dm_640M".
    num_seqs : int, optional
        How many sequences to generate (ignored if `seed_sequences` provided and
        `scheme` is None). Default: 20.
    scheme : str | None, optional
        Generation scheme: "mask", "causal-mask", "d3pm", "test-sample",
        "random", or None.  Default: None (model-specific default).
    penalty : float | None, optional
        Repetition penalty (applies to AR/DM sampling).  Default: None.
    device : str | int | None, optional
        CUDA device ("cuda:0", 0, …) or "cpu".  Default: first GPU.
    random_baseline : bool, delete_prev : bool, count : int
        Same meaning as in the original CLI script.
    output_root : str | None
        Where to create the result directory.  Default: "~/Desktop/DMs/".
    seed_sequences / seed_structure / seed_msa
        User-supplied biological context.  The current generate functions only
        use `seed_sequences` as length hints; the others are placeholders for
        future conditioning pipelines.

    Returns
    -------
    list[str]
        Generated amino-acid sequences (plain strings).
    """

    # ------------------------------------------------------------------ setup
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:0" if device is None else device)
    torch.cuda.set_device(device.index if device.type == "cuda" else 0)

    checkpoint = get_pretrained(model_type)
    d3pm = isinstance(checkpoint, tuple) and len(checkpoint) == 7
    if d3pm:
        model, collater, tokenizer, scheme_ckpt, timestep, Q_bar, Q = checkpoint
    else:
        model, collater, tokenizer, scheme_ckpt = checkpoint

    # fall back to the scheme encoded in the checkpoint if the caller did not
    # specify one explicitly
    scheme = scheme or scheme_ckpt
    model = model.eval().to(device)

    # ---------------------------------------------------------------- outputs
    home_default = str(pathlib.Path.home()) + "/Desktop/DMs/"
    output_root = output_root or home_default
    out_dir = (
        os.path.join(output_root, "random-baseline")
        if random_baseline
        else os.path.join(output_root, model_type)
    )
    os.makedirs(out_dir, exist_ok=True)

    # clean previous runs if asked
    if delete_prev:
        for f in glob.glob(os.path.join(out_dir, "generated*")):
            os.remove(f)

    # ---------------------------------------------------------------- decide what we call “input lengths”
    if seed_sequences:
        lens = [len(s) for s in seed_sequences]
    else:
        # fallback: sample lengths from UniRef (kept from original script)
        train = UniRefDataset("data/uniref50/", "train", structure=False, max_len=2048)
        lens = [len(train[np.random.randint(len(train))][0]) for _ in range(num_seqs)]

    # ---------------------------------------------------------------- generate
    generated_strings: list[str] = []

    if scheme == "causal-mask":
        _, generated_strings = generate_autoreg(
            model, tokenizer, samples=len(lens), batch_size=1, max_seq_len=max(lens)
        )

    elif scheme == "test-sample":
        valid = UniRefDataset("data/uniref50/", "rtest", structure=False, max_len=2048)
        generated_strings = generate_valid_subset(valid, samples=len(lens))

    elif scheme == "random" or random_baseline:
        train_prob = aa_reconstruction_parity_plot(
            output_root, out_dir, "placeholder.csv", gen_file=False
        )
        generated_strings = [
            generate_random_seq(L, train_prob) for L in tqdm(lens)
        ]

    else:  # "mask", "d3pm", …
        for L in tqdm(lens):
            if scheme == "mask":
                _, seqs = generate_oaardm(
                    model, tokenizer, L, penalty=penalty, batch_size=1, device=device
                )
            elif scheme == "d3pm":
                _, seqs = generate_d3pm(
                    model, tokenizer, Q, Q_bar, timestep, L,
                    batch_size=1, device=device
                )
            else:  # fallback: unconditional AR
                _, seqs = generate_autoreg(
                    model, tokenizer, samples=1, batch_size=1, max_seq_len=L
                )
            generated_strings.extend(seqs)

    # ---------------------------------------------------------------- persist
    csv_path = os.path.join(out_dir, "generated_samples_string.csv")
    fasta_path = os.path.join(out_dir, "generated_samples_string.fasta")

    with open(csv_path, "w") as f:
        for s in generated_strings:
            f.write(s + "\n")

    with open(fasta_path, "w") as f:
        for i, s in enumerate(generated_strings, start=count):
            f.write(f">SEQUENCE_{i}\n{s}\n")

    # distribution plot (same helper you already had)
    aa_reconstruction_parity_plot(output_root, out_dir, "generated_samples_string.csv")

    return generated_strings


@web_app.post("/sample/{name}")
async def app_endpoint(
        json_data: dict, name: str
):
    return JSONResponse(
        content=await evodiff_generate.remote.aio(json_data)
    )


@web_app.get("/")
async def root() -> dict:
    return {"message": "EvoDiff Design"}


@app.function()
@asgi_app()
def fastapi_app():
    return web_app
