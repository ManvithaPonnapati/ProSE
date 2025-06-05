from fastapi import FastAPI
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app
from modal import enter, method
from modal import gpu
# Downloading: "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt" to /root/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
# Downloading: "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt" to /root/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt

gpu_config = gpu.H100(count=1)

app = App("esm")
web_app = FastAPI()

image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install(
        "torch",
        "fair-esm",
    )
    .run_commands(
        "wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt -P /root/.cache/torch/hub/checkpoints/",
        "wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt -P "
        "/root/.cache/torch/hub/checkpoints/")
)

with image.imports():
    import torch
    import math, random, esm


@app.cls(image=image, gpu="a100")
class ESMModel:
    @enter()
    async def load_model(self):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().cuda()
        return self.model

    @method()
    async def compute_pll_adaptyvbio(self, sequence: str) -> dict:
        data = [("protein", sequence)]
        batch_converter = self.alphabet.get_batch_converter()
        *_, batch_tokens = batch_converter(data)
        log_probs = []
        for i in range(len(sequence)):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i + 1] = self.alphabet.mask_idx
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    self.model(batch_tokens_masked.to("cuda:0"))["logits"], dim=-1
                )
            log_probs.append(
                token_probs[0, i + 1, self.alphabet.get_idx(sequence[i])].item()
            )
        return {"pll": math.fsum(log_probs)}


@app.function(image=image, gpu=gpu_config, timeout=30000)
async def mcmc_with_log_likelihood(
        sequence: str,
        constraints: list[int],
        num_samples: int,
        temperature: float
) -> list[dict]:
    print("Running MCMC with log likelihood sampling")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()
    print("Model and alphabet loaded successfully.")

    all_pos = list(range(len(sequence)))
    frozen_set = set(constraints or [])
    mutable_pos = [p for p in all_pos if p not in frozen_set]
    if not mutable_pos:
        raise ValueError("All positions are constrained; nothing to sample.")

    current_seq = sequence
    current_ll = ESMModel().compute_pll_adaptyvbio.remote(current_seq)["pll"]

    rng = random.Random()
    samples = []
    BATCH_SIZE = 10

    while len(samples) < num_samples:
        props = []
        mutated_pos = []

        for _ in range(BATCH_SIZE):
            pos = rng.choice(mutable_pos)
            mutated_pos.append(pos)

            with torch.inference_mode():
                _, _, toks = batch_converter([("x", current_seq)])
                toks = toks.cuda()
                toks[0, pos + 1] = alphabet.mask_idx  # +1 for BOS
                logits = model(toks)["logits"][0, pos + 1]
                probs = torch.softmax(logits / temperature, dim=0).cpu()

                cur_idx = alphabet.get_idx(current_seq[pos])
                while True:
                    new_idx = torch.multinomial(probs, 1).item()
                    if new_idx != cur_idx:
                        break
                new_aa = alphabet.get_tok(new_idx)

                proposal_list = list(current_seq)
                proposal_list[pos] = new_aa
                props.append("".join(proposal_list))

        ll_prop = [ESMModel().compute_pll_adaptyvbio.remote(pr)["pll"] for pr in props]

        for prop_seq, prop_ll in zip(props, ll_prop):
            if len(samples) >= num_samples:
                break
            delta = prop_ll - current_ll
            if delta >= 0:  # Better or equal likelihood
                accept = True
            else:  # Worse likelihood - accept with probability exp(delta/T)
                accept_p = math.exp(delta / max(temperature, 1e-9))
                accept = rng.random() < accept_p

            if accept:
                current_seq, current_ll = prop_seq, prop_ll
                print(f"Accepted mutation at step {len(samples)}: LL = {current_ll:.4f}")

            # Always add current sequence to samples (whether it changed or not)
            samples.append(
                {"sequence": current_seq, "log_likelihood": current_ll}
            )

    return samples


@app.function(image=image, gpu=gpu_config, timeout=30000)
def metropolis_coupled_mcmc(
        sequence: str,
        constraints: list[int],
        num_samples: int,
        num_chains: int = 4,
        swap_frequency: int = 10,
) -> dict:
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    @torch.inference_mode()
    def pll(seqs: list[str]) -> torch.Tensor:
        data = [(str(i), s) for i, s in enumerate(seqs)]
        _, _, toks = batch_converter(data)
        toks = toks.cuda()

        B, L = toks.shape
        mask_token = alphabet.mask_idx
        logls = torch.zeros(B, device="cuda")

        idx_repeat = torch.repeat_interleave(torch.arange(B, device="cuda"), L)
        toks_big = toks[idx_repeat].clone()
        pos_big = torch.tile(torch.arange(L, device="cuda"), (B,))
        toks_big[torch.arange(toks_big.size(0), device="cuda"), pos_big] = mask_token

        logits = model(toks_big)["logits"]
        logp = torch.log_softmax(logits, dim=-1)

        true_aa = toks[idx_repeat, pos_big]
        ll_per_pos = logp[torch.arange(logp.size(0), device="cuda"), pos_big, true_aa]
        logls = ll_per_pos.view(B, L).sum(dim=1)
        return logls

    all_pos = list(range(len(sequence)))
    frozen_set = set(constraints or [])
    mutable_pos = [p for p in all_pos if p not in frozen_set]
    if not mutable_pos:
        raise ValueError("All positions are constrained; nothing to sample.")

    if num_chains == 1:
        temperatures = [0.1]
    else:
        temperatures = [0.1 + (1.2 - 0.1) * i / (num_chains - 1)
                        for i in range(num_chains)]

    chains = []
    for i in range(num_chains):
        current_ll = pll([sequence])[0].item()
        chains.append({
            'sequence': sequence,
            'log_likelihood': current_ll,
            'temperature': temperatures[i],
            'chain_id': i
        })

    cold_samples = [{"sequence": sequence, "log_likelihood": chains[0]['log_likelihood']}]

    swap_attempts = 0
    swap_accepts = 0
    chain_accepts = [0] * num_chains
    chain_attempts = [0] * num_chains

    rng = random.Random()
    BATCH_SIZE = 20
    iteration = 0
    while len(cold_samples) < num_samples:
        iteration += 1

        for chain_idx in range(num_chains):
            chain = chains[chain_idx]
            temp = chain['temperature']


            props = []
            mutated_pos = []
            for _ in range(BATCH_SIZE):
                pos = rng.choice(mutable_pos)
                mutated_pos.append(pos)

                with torch.inference_mode():
                    _, _, toks = batch_converter([("x", chain['sequence'])])
                    toks = toks.cuda()
                    toks[0, pos + 1] = alphabet.mask_idx
                    logits = model(toks)["logits"][0, pos + 1]
                    probs = torch.softmax(logits / temp, dim=0).cpu()

                cur_idx = alphabet.get_idx(chain['sequence'][pos])
                while True:
                    new_idx = torch.multinomial(probs, 1).item()
                    if new_idx != cur_idx:
                        break

                new_aa = alphabet.get_tok(new_idx)
                proposal_list = list(chain['sequence'])
                proposal_list[pos] = new_aa
                props.append("".join(proposal_list))

            ll_props = pll(props).cpu().tolist()
            for prop_seq, prop_ll in zip(props, ll_props):
                chain_attempts[chain_idx] += 1

                delta = prop_ll - chain['log_likelihood']
                accept_p = math.exp(delta / temp)

                if rng.random() < min(1.0, accept_p):
                    chain['sequence'] = prop_seq
                    chain['log_likelihood'] = prop_ll
                    chain_accepts[chain_idx] += 1

                    if chain_idx == 0:
                        cold_samples.append({
                            "sequence": prop_seq,
                            "log_likelihood": prop_ll
                        })
                        if len(cold_samples) >= num_samples:
                            break

            if len(cold_samples) >= num_samples:
                break

        if iteration % swap_frequency == 0 and num_chains > 1:
            i = rng.randint(0, num_chains - 2)
            j = i + 1

            chain_i = chains[i]
            chain_j = chains[j]

            swap_attempts += 1
            α_swap = min(1, [π(x_j)^(1/T_i) * π(x_i)^(1/T_j)] / [π(x_i)^(1/T_i) * π(x_j)^(1/T_j)])
            ll_i = chain_i['log_likelihood']
            ll_j = chain_j['log_likelihood']
            temp_i = chain_i['temperature']
            temp_j = chain_j['temperature']

            delta_temp = (1.0 / temp_i) - (1.0 / temp_j)
            delta_ll = ll_j - ll_i

            swap_log_prob = delta_temp * delta_ll
            swap_prob = math.exp(min(0, swap_log_prob))  # min with 0 for numerical stability

            if rng.random() < swap_prob:
                # Swap the sequences and log-likelihoods
                chain_i['sequence'], chain_j['sequence'] = chain_j['sequence'], chain_i['sequence']
                chain_i['log_likelihood'], chain_j['log_likelihood'] = chain_j['log_likelihood'], chain_i[
                    'log_likelihood']
                swap_accepts += 1

    swap_rate = swap_accepts / max(swap_attempts, 1)
    chain_rates = [accepts / max(attempts, 1) for accepts, attempts in zip(chain_accepts, chain_attempts)]

    return {
        'samples': cold_samples,
        'diagnostics': {
            'temperatures': temperatures,
            'swap_acceptance_rate': swap_rate,
            'chain_acceptance_rates': chain_rates,
            'total_iterations': iteration,
            'swap_attempts': swap_attempts,
            'swap_accepts': swap_accepts
        }
    }

@app.function(image=image,timeout=30000)
def parallel_mc_mcmc(sequence,constraints, num_samples):
    arguments_list = []
    for _ in range(num_samples):
        arguments_list.append((sequence, constraints,1))
    return list(metropolis_coupled_mcmc.starmap(arguments_list))

@app.function(image=image,timeout=30000)
def parallel_mcmc(sequence,constraints, num_samples,temperature):
    arguments_list = []
    for _ in range(num_samples):
        arguments_list.append((sequence, constraints,1,temperature))
    return list(mcmc_with_log_likelihood.starmap(arguments_list))




@web_app.post("/esm/{name}")
async def esm_endpoint(json_data: dict, name: str):
    blob = {}
    if name == "pll":
        blob = ESMModel().compute_pll_adaptyvbio.remote(json_data["sequence"])
    elif name == "mcmc":
        blob = await parallel_mcmc.remote.aio(
            json_data["sequence"],
            json_data["constraints"],
            json_data["num_samples"],
            json_data["temperature"]
        )
    elif name == "mcmc_coupled":
        blob = await parallel_mc_mcmc.remote.aio(
            json_data["sequence"],
            json_data["constraints"],
            json_data["num_samples"]
        )
    return JSONResponse(content=blob)


@web_app.get("/")
async def root() -> dict:
    return {"message": "ESM2"}


@app.function()
@asgi_app()
def fastapi_app():
    return web_app
