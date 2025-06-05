import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app, gpu

# EvoDiff / helper imports

# use mid-tier GPU here
gpu_config = gpu.A10G(count=1)

app = App("evodiff")
web_app = FastAPI()

image = (
    Image.debian_slim(python_version="3.9")
    .micromamba()
    .apt_install("wget", "git","gcc", "g++", "libopenblas-dev", "libomp-dev")
    .run_commands("pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118")
    .pip_install( "numpy", "tqdm", "pandas", "scikit-learn")
    .pip_install("evodiff").run_commands(
    # make sure the cache dir exists, then download
    "mkdir -p /root/.cache/torch/hub/checkpoints && "
    "wget https://zenodo.org/record/8045076/files/oaar-38M.tar "
    "-O /root/.cache/torch/hub/checkpoints/oaar-38M.tar"
)
)
with image.imports():
    from evodiff.pretrained import (
        CARP_38M, CARP_640M, D3PM_BLOSUM_38M, D3PM_BLOSUM_640M,
        D3PM_UNIFORM_38M, D3PM_UNIFORM_640M, OA_DM_38M, OA_DM_640M,
        LR_AR_38M, LR_AR_640M, ESM1b_650M
    )
    import evodiff
    import os
    import numpy as np
    from tqdm import tqdm
    from evodiff.data import A3MMSADataset
    from torch.utils.data import Subset
    from torch.utils.data import DataLoader
    import torch
    from sequence_models.collaters import MSAAbsorbingCollater
    from evodiff.collaters import D3PMCollaterMSA
    from sequence_models.constants import MSA_ALPHABET
    from evodiff.utils import Tokenizer


    def get_valid_data(data_top_dir, num_seqs, arg_mask, data_dir='openfold/', selection_type='MaxHamming',
                       n_sequences=64, max_seq_len=512,
                       out_path='../DMs/ref/'):
        valid_msas = []
        query_msas = []
        seq_lens = []

        _ = torch.manual_seed(1)  # same seeds as training
        np.random.seed(1)

        dataset = A3MMSADataset(selection_type, n_sequences, max_seq_len, data_dir=os.path.join(data_top_dir, data_dir),
                                min_depth=64)

        train_size = len(dataset)
        random_ind = np.random.choice(train_size, size=(train_size - 10000), replace=False)
        val_ind = np.delete(np.arange(train_size), random_ind)

        ds_valid = Subset(dataset, val_ind)

        if arg_mask == 'autoreg':
            tokenizer = Tokenizer()
            collater = MSAAbsorbingCollater(alphabet=MSA_ALPHABET)
        elif arg_mask == 'blosum' or arg_mask == 'random':
            diffusion_timesteps = 500
            tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat")
            if arg_mask == 'random':
                Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=diffusion_timesteps)
            if arg_mask == 'blosum':
                Q_prod, Q_t = tokenizer.q_blosum_schedule(timesteps=diffusion_timesteps)
            collater = D3PMCollaterMSA(tokenizer=tokenizer, num_timesteps=diffusion_timesteps, Q=Q_t, Q_bar=Q_prod)

        torch.seed()  # reset seed ater val_ind
        loader = DataLoader(dataset=ds_valid,
                            batch_size=1,
                            shuffle=True,
                            collate_fn=collater,
                            num_workers=8)

        count = 0
        print("NUM SEQS", num_seqs)
        for batch in tqdm(loader):
            if arg_mask == 'blosum' or arg_mask == 'random':
                src, src_one_hot, timestep, tgt, tgt_one_hot, Q, Q_prod, q = batch
            else:
                src, tgt, mask = batch
            if count < num_seqs:
                valid_msas.append(tgt)
                print("QUERY", tokenizer.untokenize(tgt[0][0]), tgt[0][0].shape)
                seq_lens.append(len(tgt[0][0]))
                query_msas.append(tgt[0][0])  # first sequence in batchsize=1
                count += len(tgt)
            else:
                break
        print("LEN VALID MSAS", len(valid_msas))
        untokenized = [[tokenizer.untokenize(msa.flatten())] for msa in valid_msas]
        fasta_string = ""
        with open(out_path + 'valid_msas.a3m', 'a') as f:
            for i, msa in enumerate(untokenized):
                for seq in range(n_sequences):
                    seq_num = seq * seq_lens[i]
                    next_seq_num = (seq + 1) * seq_lens[i]
                    if seq_num == 0:
                        f.write(">SEQUENCE_" + str(i) + "\n" + str(msa[0][seq_num:next_seq_num]) + "\n")
                    else:
                        f.write(">tr \n" + str(msa[0][seq_num:next_seq_num]) + "\n")
            f.write(fasta_string)
            f.close()

        return valid_msas, query_msas, tokenizer

    def generate_msa(model, tokenizer, batch_size, n_sequences, seq_length, penalty_value=2, device='gpu',
                     start_query=False, start_msa=False, data_top_dir='../data', selection_type='MaxHamming',
                     out_path='../ref/'):
        mask_id = tokenizer.mask_id
        src = torch.full((batch_size, n_sequences, seq_length), fill_value=mask_id)
        masked_loc_x = np.arange(n_sequences)
        masked_loc_y = np.arange(seq_length)
        if start_query:
            valid_msas, query_sequences, tokenizer = get_valid_data(data_top_dir, batch_size, 'autoreg',
                                                                    data_dir='openfold/',
                                                                    selection_type=selection_type,
                                                                    n_sequences=n_sequences, max_seq_len=seq_length,
                                                                    out_path=out_path)
            # First row is query sequence
            for i in range(batch_size):
                seq_len = len(query_sequences[i])
                print("PAD ID", tokenizer.pad_id)
                src[i][0][:seq_len] = query_sequences[i]
                padding = torch.full((n_sequences, seq_length - seq_len), fill_value=tokenizer.pad_id)
                src[i, :, seq_len:] = padding
                x_indices = np.arange(1, n_sequences)
                y_indices = np.arange(seq_len)
        elif start_msa:
            valid_msas, query_sequences, tokenizer = get_valid_data(data_top_dir, batch_size, 'autoreg',
                                                                    data_dir='openfold/',
                                                                    selection_type=selection_type,
                                                                    n_sequences=n_sequences,
                                                                    max_seq_len=seq_length,
                                                                    out_path=out_path)
            for i in range(batch_size):
                seq_len = len(query_sequences[i])
                src[i, 1:n_sequences, :seq_len] = valid_msas[i][0, 1:n_sequences, :seq_len].squeeze()
                padding = torch.full((n_sequences, seq_length - seq_len), fill_value=tokenizer.pad_id)
                src[i, :, seq_len:] = padding
                x_indices = np.arange(0, 1)
                y_indices = np.arange(seq_len)
        src = src.to(device)
        sample = src.clone()
        if start_query or start_msa:
            all_ind = np.transpose([np.tile(x_indices, len(y_indices)), np.repeat(y_indices, len(x_indices))])
        else:
            all_ind = np.transpose(
                [np.tile(masked_loc_x, len(masked_loc_y)), np.repeat(masked_loc_y, len(masked_loc_x))])
        np.random.shuffle(all_ind)

        with torch.no_grad():
            for i in tqdm(all_ind):
                random_x, random_y = i
                preds = model(sample)  # Output shape of preds is (BS=1, N=64, L, n_tokens=31)
                p = preds[:, random_x, random_y, :]
                if random_x == 0:  # for first row don't let p_softmax predict gaps
                    p = preds[:, random_x, random_y, :tokenizer.K - 1]
                p_softmax = torch.nn.functional.softmax(p, dim=1)
                # Penalize gaps
                penalty = torch.ones(p.shape).to(p.device)
                penalty[:, -1] += penalty_value
                p_softmax /= penalty
                p_sample = torch.multinomial(input=p_softmax, num_samples=1)
                p_sample = p_sample.squeeze()
                sample[:, random_x, random_y] = p_sample
        untokenized = [[tokenizer.untokenize(msa.flatten())] for msa in sample]
        return sample, untokenized  # return output and untokenized output


def generate_random_seq(seq_len, train_prob_dist, tokenizer):
    sample = torch.multinomial(torch.tensor(train_prob_dist), num_samples=seq_len, replacement=True)
    sample = sample.to(torch.long)
    return tokenizer.untokenize(sample)

def generate_oaardm(model, tokenizer, seq_len, penalty=None, batch_size=3, device='cuda'):
    all_aas = tokenizer.all_aas
    mask = tokenizer.mask_id

    sample = torch.zeros((batch_size, seq_len)) + mask
    sample = sample.to(torch.long).to(device)

    loc = np.arange(seq_len)
    np.random.shuffle(loc)
    with torch.no_grad():
        for i in tqdm(loc):
            timestep = torch.tensor([0] * batch_size, device=device)
            prediction = model(sample, timestep)
            p = prediction[:, i, :len(all_aas) - 6]
            p = torch.nn.functional.softmax(p, dim=1)
            p_sample = torch.multinomial(p, num_samples=1)
            if penalty is not None:
                for j in range(batch_size):
                    case1 = (i == 0 and sample[j, i + 1] == p_sample[j])
                    case2 = (i == seq_len - 1 and sample[j, i - 1] == p_sample[j])
                    case3 = ((0 < i < seq_len - 1) and ((sample[j, i - 1] == p_sample[j]) or (sample[j, i + 1] == p_sample[j])))
                    if case1 or case2 or case3:
                        p[j, int(p_sample[j])] /= penalty
                        p_sample[j] = torch.multinomial(p[j], num_samples=1)
            sample[:, i] = p_sample.squeeze()
    untokenized = [tokenizer.untokenize(s) for s in sample]
    return sample, untokenized

def generate_autoreg(model, tokenizer, samples=100, batch_size=1, max_seq_len=1024):
    device = model.device()
    start, stop = tokenizer.start_id, tokenizer.stop_id
    sample_out, untokenized_out = [], []
    timestep = torch.tensor([0] * batch_size, device=device)

    for _ in tqdm(range(samples)):
        sample = torch.tensor([[start]], device=device)
        reach_stop = False
        with torch.no_grad():
            for _ in range(max_seq_len):
                if not reach_stop:
                    prediction = model(sample, timestep)
                    p = torch.nn.functional.softmax(prediction[:, -1, :], dim=1)
                    p_sample = torch.multinomial(p, num_samples=1)
                    sample = torch.cat((sample, p_sample), dim=1)
                    if p_sample == stop:
                        reach_stop = True
                else:
                    break
        seq = tokenizer.untokenize(sample[0, 1:-1])
        sample_out.append(sample[0, 1:-1])
        untokenized_out.append(seq)
    return sample_out, untokenized_out

def generate_valid_subset(data_valid, samples=20):
    seqs = []
    for _ in tqdm(range(samples)):
        idx = np.random.choice(len(data_valid))
        seqs.append(data_valid[idx][0])
    return seqs

def generate_d3pm(model, tokenizer, Q, Q_bar, timesteps, seq_len, batch_size=3, device='cuda'):
    sample = torch.randint(0, tokenizer.K, (batch_size, seq_len), device=device)
    Q, Q_bar = Q.to(device), Q_bar.to(device)
    timesteps = torch.linspace(timesteps - 1, 1, int((timesteps - 1)), dtype=int, device=device)
    with torch.no_grad():
        for t in tqdm(timesteps):
            t_batch = torch.tensor([int(t)] * batch_size, device=device)
            pred = model(sample, t_batch)
            p = torch.nn.functional.softmax(pred[:, :, :tokenizer.K], dim=-1).double()
            x_tminus1 = sample.clone()
            for i, s in enumerate(sample):
                x_t_b = tokenizer.one_hot(s)
                A = torch.mm(x_t_b, Q[t].t())
                Q_expand = Q_bar[t - 1].unsqueeze(0).expand(batch_size, tokenizer.K, tokenizer.K)
                B_pred = p[i].unsqueeze(2) * Q_expand
                q_t = A.unsqueeze(1) * B_pred
                p_theta = torch.bmm(q_t.transpose(1,2), p[i].unsqueeze(2)).squeeze()
                p_theta /= p_theta.sum(dim=1, keepdim=True)
                x_tminus1[i] = torch.multinomial(p_theta, num_samples=1).squeeze()
                if t == 1:
                    x_tminus1[i] = torch.multinomial(p_theta[:, :tokenizer.K-6], num_samples=1).squeeze()
            sample = x_tminus1
    untokenized = [tokenizer.untokenize(s) for s in sample]
    return sample, untokenized

def get_pretrained(model_type: str):
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
    raise ValueError(f"Unknown model_type: {model_type}")


@app.function(
    image=image,
    gpu=gpu_config,
    concurrency_limit=5,
)
def evodiff_generate(
    sequence
) -> list[str]:
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device.index)

    pretrained_model = evodiff.pretrained.MSA_OA_DM_MAXSUB()
    mask_id = pretrained_model[2].mask_id
    pad_id = pretrained_model[2].pad_id
    model, collater, tokenizer, scheme = pretrained_model
    model = model.eval().to(device)
    mask_id = tokenizer.mask_id
    n_sequences = 64
    batch_size = 1
    seq_length = 1024
    data_top_dir = "/evodiff/data/"
    #make uniqiue id
    unique_id = str(int(time.time() * 1000))
    out_fpath_dir = f"/evodiff/data/{unique_id}"
    os.makedirs(out_fpath_dir, exist_ok=True)
    #make directories
    os.makedirs(data_top_dir, exist_ok=True)
    sample, _string = generate_msa(model, tokenizer, batch_size, n_sequences, seq_length,
                                  penalty_value=0, device="cuda:0", start_query= False,
                                  start_msa=True,
                                  data_top_dir=data_top_dir, selection_type='MaxHamming', out_path=out_fpath_dir)
    generated_strings = []
    for count, msa in enumerate(_string):
        for seq in range(n_sequences):
            seq_num = seq * seq_length
            next_seq_num = (seq + 1) * seq_length
            seq_string = str(msa[0][seq_num:next_seq_num]).replace('!', '')  # remove PADs
            generated_strings.append(seq_string)
    print("Generated strings", generated_strings)
    return generated_strings

@web_app.post("/sample")
async def app_endpoint(json_data: dict):
    seq = json_data.get("sequence")
    if seq is None:
        return JSONResponse({"error": "please provide a 'sequence' key"}, status_code=400)

    msa_path = json_data.get("msa_path")
    msa_str = None
    if msa_path:
        try:
            msa_str = Path(msa_path).read_text()
        except Exception as e:
            return JSONResponse({"error": f"could not read msa at {msa_path}: {e}"}, status_code=400)

    result = await evodiff_generate.remote.aio(
        seed_sequences=[seq],
        seed_msa=msa_str,
        model_type=json_data.get("model_type", "oa_dm_640M"),
        num_seqs=json_data.get("num_seqs", 20),
        scheme=json_data.get("scheme"),
        penalty=json_data.get("penalty"),
        random_baseline=json_data.get("random_baseline", False),
        delete_prev=json_data.get("delete_prev", False),
        count=json_data.get("count", 0),
        output_root=json_data.get("output_root"),
    )
    return JSONResponse(content=result)


@web_app.post("/sample")
async def app_endpoint(json_data: dict):
    seq = json_data.get("sequence")
    if seq is None:
        return JSONResponse(
            {"error": "please provide a 'sequence' key"},
            status_code=400
        )

    # accept an in-memory list of MSA sequences under "msa"
    msa_list = json_data.get("msa")
    print(msa_list)
    if msa_list is not None:
        if not isinstance(msa_list, list) or not all(isinstance(s, str) for s in msa_list):
            return JSONResponse(
                {"error": "'msa' must be a list of sequence strings"},
                status_code=400
            )
        seed_msa = msa_list
    else:
        seed_msa = None

    result = await evodiff_generate.remote.aio(
        seed_sequences=[seq],
        seed_msa=seed_msa,
        model_type=json_data.get("model_type", "oa_dm_640M"),
        num_seqs=json_data.get("num_seqs", 20),
        scheme=json_data.get("scheme"),
        penalty=json_data.get("penalty"),
        random_baseline=json_data.get("random_baseline", False),
        delete_prev=json_data.get("delete_prev", False),
        count=json_data.get("count", 0),
        output_root=json_data.get("output_root"),
    )
    return JSONResponse(content=result)


@web_app.get("/")
async def root() -> dict:
    return {"message": "EvoDiff Design"}


@app.function()
@asgi_app()
def fastapi_app():
    return web_app
