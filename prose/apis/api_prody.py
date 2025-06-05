import os
import subprocess
import tempfile

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app

app = App("prody")
web_app = FastAPI()

image = (
    Image.debian_slim(python_version="3.9")
    .micromamba()
    .apt_install("wget", "git", "curl", "gcc", "g++")
    .run_commands(
        "git clone https://github.com/prody/ProDy.git; cd ProDy; python setup.py install"
    )
    .micromamba_install(
        "openbabel", channels=["conda-forge", "bioconda"]
    )  # this is the version used by the model
    .run_commands("cd /ProDy/prody/proteins/hpbmodule; ls -l")
    .run_commands("cp /ProDy/prody/proteins/hpbmodule/hpb_Python3.9/hpb.so /usr/lib/")
    .run_commands(
        "cp /ProDy/prody/proteins/hpbmodule/hpb_Python3.9/hpb.so /ProDy/prody/proteins/"
    )
    .run_commands("cp /ProDy/prody/proteins/hpbmodule/hpb_Python3.9/hpb.so .")
    .pip_install("pdb-tools")
)

with image.imports():
    from prody import addMissingAtoms, parsePDB
    from prody.proteins.interactions import Interactions


@app.function(image=image)
def get_bonds(json_data: dict) -> dict:
    interaction_json = {}
    pdb_str = json_data["pdb_str"]
    interactions = Interactions()
    with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".pdb", dir="."
    ) as temp_file:
        temp_file.write(pdb_str)
        temp_file.flush()
        addMissingAtoms(temp_file.name, method="openbabel")
        base_file_name = os.path.basename(temp_file.name)
        processed_file_name = "addH_" + str(base_file_name)
        renumbered_file_name = "renumbered_" + base_file_name
        with open(renumbered_file_name, "w") as renumbered_file:
            subprocess.run(  # noqa: S603
                ["pdb_reres", "-1", processed_file_name],  # noqa: S607
                stdout=renumbered_file,
                check=True,
            )
        # Parse the renumbered PDB file
        atoms = parsePDB(renumbered_file_name).select("protein")
        interactions.calcProteinInteractions(atoms)
        try:
            interaction_json["hydrogen_bonds"] = interactions.getHydrogenBonds()
        except Exception:
            interaction_json["hydrogen_bonds"] = []
        interaction_json["salt_bridges"] = interactions.getSaltBridges()
        interaction_json["repulsive_ionic_bonding"] = (
            interactions.getRepulsiveIonicBonding()
        )
        interaction_json["pi_stacking"] = interactions.getPiStacking()
        interaction_json["pi_cation"] = interactions.getPiCation()
        interaction_json["disulfide_bonds"] = interactions.getDisulfideBonds()
        return interaction_json


@web_app.post("/prody/{name}")
async def endpoint(json_data: dict, name: str):
    blob = None
    if name == "get_bonds_monomer":
        blob = await get_bonds.remote.aio(json_data)
    return JSONResponse(content=blob)


@web_app.get("/")
async def root() -> dict:
    return {"message": "Multimer or Monomer AlphaFold2"}


@app.function()
@asgi_app()
def fastapi_app():
    return web_app
