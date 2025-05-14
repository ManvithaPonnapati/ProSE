import secrets

from fastapi import Depends, HTTPException, status
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from modal import App, Image, asgi_app
from modal import enter, method
from modal import gpu

auth_scheme = HTTPBearer()
gpu_config = gpu.A100(count=1)

def validate_token(
        credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),  # noqa: B008
) -> str:
    if not secrets.compare_digest(credentials.credentials, "hgtaaproteindesign2025"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


app = App("esm")
web_app = FastAPI(dependencies=[Depends(validate_token)])

image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install(
        "torch",
        "fair-esm",
    )
    .run_commands(
        "wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
    )
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
def mcmc_with_log_likelihood(
    sequence: str,
    constraints: list[int],          # 0-based residue indices you do NOT want to touch
    num_samples: int,                # number of accepted samples you want back (inc. the starting seq)
    temperature: float               # Metropolis-Hastings temperature
) -> list[dict]:

    # ---------------------------------------------------------------------
    # 1.  Load model on the worker GPU just once
    # ---------------------------------------------------------------------
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    # ---------------------------------------------------------------------
    # 2.  Helper: vectorised PLL for *any* batch of sequences
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def pll(seqs: list[str]) -> torch.Tensor:            # (B,) tensor (CUDA)
        data = [(str(i), s) for i, s in enumerate(seqs)] # dummy names
        _, _, toks = batch_converter(data)
        toks = toks.cuda()

        B, L = toks.shape
        mask_token = alphabet.mask_idx
        logls = torch.zeros(B, device="cuda")

        # *****  Mask every position ONCE, feed the whole (B·L) mega-batch  *****
        # build (B·L, L) tensor where row i⋅L + p is seq-i with pos-p masked
        idx_repeat = torch.repeat_interleave(torch.arange(B, device="cuda"), L)
        toks_big   = toks[idx_repeat].clone()
        pos_big    = torch.tile(torch.arange(L, device="cuda"), (B,))
        toks_big[torch.arange(toks_big.size(0), device="cuda"), pos_big] = mask_token

        logits = model(toks_big)["logits"]              # (B·L, L, |Σ|)
        logp   = torch.log_softmax(logits, dim=-1)

        # gather correct tokens
        true_aa = toks[idx_repeat, pos_big]
        ll_per_pos = logp[torch.arange(logp.size(0), device="cuda"), pos_big, true_aa]

        # sum over positions, then reshape back to (B,)
        logls = ll_per_pos.view(B, L).sum(dim=1)
        return logls                                     # CUDA tensor

    # ---------------------------------------------------------------------
    # 3.  Pre-compute immutable / mutable positions
    # ---------------------------------------------------------------------
    all_pos       = list(range(len(sequence)))
    frozen_set    = set(constraints or [])
    mutable_pos   = [p for p in all_pos if p not in frozen_set]
    if not mutable_pos:
        raise ValueError("All positions are constrained; nothing to sample.")

    # ---------------------------------------------------------------------
    # 4.  Initial state
    # ---------------------------------------------------------------------
    current_seq = sequence
    current_ll  = pll([current_seq])[0].item()

    rng      = random.Random()
    samples  = [{"sequence": current_seq, "log_likelihood": current_ll}]

    # ---------------------------------------------------------------------
    # 5.  Main loop – we work in *batches* of proposals for GPU efficiency
    # ---------------------------------------------------------------------
    BATCH_SIZE = 100                         # tune for your GPU memory
    while len(samples) < num_samples:
        # ---------- 5a. build proposal batch ----------
        props          = []
        mutated_pos    = []                  # remember which site each proposal changed
        for _ in range(BATCH_SIZE):
            pos              = rng.choice(mutable_pos)
            mutated_pos.append(pos)

            # mask-and-score in one pass to get proposal distribution
            with torch.inference_mode():
                _, _, toks = batch_converter([("x", current_seq)])
                toks       = toks.cuda()
                toks[0, pos + 1] = alphabet.mask_idx   # +1 for BOS
                logits     = model(toks)["logits"][0, pos + 1]
                probs      = torch.softmax(logits / temperature, dim=0).cpu()

            # sample an aa that is different from the current one
            cur_idx   = alphabet.get_idx(current_seq[pos])
            while True:
                new_idx = torch.multinomial(probs, 1).item()
                if new_idx != cur_idx and alphabet.is_valid_idx(new_idx):
                    break

            new_aa        = alphabet.get_tok(new_idx)
            proposal_list = list(current_seq)
            proposal_list[pos] = new_aa
            props.append("".join(proposal_list))

        # ---------- 5b. score whole batch in parallel ----------
        ll_prop = pll(props).cpu().tolist()            # list[float]

        # ---------- 5c. MH accept/reject for each proposal in order ----------
        for prop_seq, prop_ll in zip(props, ll_prop):
            if len(samples) >= num_samples:
                break

            delta    = prop_ll - current_ll
            accept_p = math.exp(delta / max(temperature, 1e-9))
            if rng.random() < min(1.0, accept_p):
                current_seq, current_ll = prop_seq, prop_ll
                samples.append(
                    {"sequence": current_seq, "log_likelihood": current_ll}
                )

    return samples


@web_app.post("/esm/{name}")
async def esm_endpoint(json_data: dict, name: str):
    blob = {}
    if name == "pll":
        blob = ESMModel().compute_pll_adaptyvbio.remote(json_data["sequence"])
    elif name=="mcmc":
        blob = await mcmc_with_log_likelihood.remote.aio(
            json_data["sequence"],
            json_data["constraints"],
            json_data["num_samples"],
            json_data["temperature"]
        )
    return JSONResponse(content=blob)


@web_app.get("/")
async def root() -> dict:
    return {"message": "ESM2"}


@app.function()
@asgi_app()
def fastapi_app():
    return web_app
