import os
from dataclasses import dataclass
from pathlib import Path
import base64
import modal
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse

# A constant for timeout calculations
MINUTES = 60

# Create a Modal App and a FastAPI web app
app = modal.App(name="boltz1")
web_app = FastAPI()

# Define a Modal image that installs boltz and PyYAML
image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "pip install boltz",
        "pip install pyyaml",
        "pip install huggingface_hub",
        "pip install hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands("pip install biopython")
)

# Create a Modal Volume for model weights and assign a directory path
boltz_model_volume = modal.Volume.from_name("boltz1-models", create_if_missing=True)
models_dir = Path("/models/boltz1")

with image.imports():
    from Bio.PDB import MMCIFParser, PDBIO


@app.function(
    image=image,
    volumes={models_dir: boltz_model_volume},
    timeout=10 * MINUTES,
    gpu="H100",
)
def boltz1_inference(
        protein_sequence: str, ligand_sequence: str, idx: int
) -> bytes:
    print("🧬 Starting Boltz1 inference")
    import subprocess
    from uuid import uuid4
    unique_id = str(uuid4())
    print(os.listdir("."))
    with open(f"sequence_input_{unique_id}.fasta", "w+") as f:
        f.write(f">A|protein|\n{protein_sequence}")
        if ligand_sequence:
            f.write(f">B|smiles|\n{ligand_sequence}" if ligand_sequence else "")
    print("Sequence file written to directory")
    # Build the command as a list of arguments.
    print(os.listdir("."))
    boltz_command = f"boltz predict sequence_input_{unique_id}.fasta --use_msa_server --override"
    print(f"Running command: {boltz_command}")
    result = subprocess.run(  # noqa: S602
        boltz_command, capture_output=True, shell=True, check=False
    )
    output_bytes = package_outputs(
        f"boltz_results_sequence_input_{unique_id}"  # The directory where Boltz outputs its results.
    )
    print("🧬 outputs packaged")
    return output_bytes


@app.function(
    volumes={models_dir: boltz_model_volume},
    timeout=20 * MINUTES,
    image=image,
)
def download_model(
        force_download: bool = False,
        revision: str = "7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1",
):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="boltz-community/boltz-1",
        revision=revision,
        local_dir=models_dir,
        force_download=force_download,
    )
    boltz_model_volume.commit()
    print(f"🧬 model downloaded to {models_dir}")


@dataclass
class MSA:
    data: str
    path: Path


def find_msas(boltz_yaml_path: Path) -> list[MSA]:
    """Finds the MSA data in a YAML file in the Boltz input format.

    See https://github.com/jwohlwend/boltz/blob/2355c62c957e95305527290112e9742d0565c458/docs/prediction.md for details."""
    import yaml
    data = yaml.safe_load(boltz_yaml_path.read_text())
    data_dir = boltz_yaml_path.parent
    sequences = data["sequences"]
    msas = []
    for sequence in sequences:
        if protein := sequence.get("protein"):
            if msa_path := protein.get("msa"):
                if msa_path == "empty":  # special value
                    continue
                if not msa_path.startswith("."):
                    raise ValueError(
                        f"Must specify MSA paths relative to the input yaml path, but got {msa_path}"
                    )
                msa_data = (data_dir / Path(msa_path).name).read_text()
                msas.append(MSA(msa_data, Path(msa_path)))
    return msas


def package_outputs(output_dir: str) -> bytes:
    import io
    import tarfile
    tar_buffer = io.BytesIO()
    print(os.listdir(output_dir) if os.path.exists(output_dir) else "Output directory does not exist")  # noqa: S608
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tar.add(output_dir, arcname=output_dir)
    return tar_buffer.getvalue()


@app.function(
    volumes={models_dir: boltz_model_volume},
    image=image,
)
def boltz1_train_parallel(protein_sequences, ligand_sequence=None):
    argument_list = []
    for idx, protein_sequence in enumerate(protein_sequences):
        argument_list.append((protein_sequence, ligand_sequence, idx))
    return list(boltz1_inference.starmap(argument_list))


# FastAPI endpoint for inference.
@web_app.post("/infer")
async def infer_endpoint(json_data: dict):
    protein_sequences = json_data["protein_sequences"]
    ligand_sequence = json_data.get("ligand_sequence", None)
    output_data = await boltz1_train_parallel.remote.aio(protein_sequences, ligand_sequence)
    encoded_results = [
        base64.b64encode(data).decode("utf-8") for data in output_data
    ]
    return JSONResponse(content={"results": encoded_results})


# Expose the FastAPI app via Modal's ASGI integration.
@app.function()
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    return web_app
