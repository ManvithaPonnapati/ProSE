import hashlib
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from modal import App, Image
from modal import Dict

rosetta_volume_cache = Dict.from_name("rosetta_cache", create_if_missing=True)

app = App("rosetta")
web_app = FastAPI()

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "mc",
        "git",
        "bash",
        "zlib1g-dev",
        "build-essential",
        "cmake",
        "ninja-build",
        "clang",
        "clang-tools",
        "clangd",
        "curl",
        "wget",
    )
    .run_commands("git clone https://github.com/RosettaCommons/rosetta.git")
    .run_commands("cd rosetta/source; ./scons.py -j8 mode=release bin")
    .run_commands("pip install --upgrade pip")
    .pip_install("pyrosetta-installer")
    .pip_install(
        "https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Debug.python310.linux.wheel/pyrosetta-2024.39+release.59628fb-cp310-cp310-linux_x86_64.whl"
    )
    .run_commands(
        "python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'"
    )
    .run_commands(
        "rm -rf /dl_binder_design; git clone https://github.com/nrbennet/dl_binder_design.git"
    )
    .pip_install("google-cloud-storage", "google-auth")
    .run_commands(
        "wget https://files.ipd.uw.edu/pub/robust_de_novo_design_minibinders_2021/supplemental_files/scripts_and_main_pdbs.tar.gz; "
        "tar -xvf /scripts_and_main_pdbs.tar.gz"
    )
    .run_commands("git clone https://github.com/martinpacesa/BindCraft.git; ls -l")
    .run_commands("chmod +x BindCraft/functions/DAlphaBall.gcc")
)


def delete_file(filepath: Path):
    """Function to delete the file after sending the response."""
    if filepath.exists():
        os.remove(filepath)
        print(f"Deleted file: {filepath}")


with image.imports():
    import pyrosetta

    # TODO: One of the protocols uses these weights - per_sap_score. In the future we might want to not init
    # this way and instead do it in the call but for now since this is the only function we have, it's fine
    # it takes too long to init otherwise
    # these weights are used by the paper so we are using the same rosetta weights.
    pyrosetta.init(
        options="-corrections:beta_nov16 -renumber_pdb -holes:dalphaball /BindCraft/functions/DAlphaBall.gcc"
    )
    one_letter_amino_acid_alphabet = list("ARNDCQEGHILKMFPSTWYV-")
    three_letter_amino_acid_alphabet = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "GAP",
    ]
    # python 3.8 doesn't have the strict flag, so we have to use noqa
    aa_1_3 = dict(zip(one_letter_amino_acid_alphabet, three_letter_amino_acid_alphabet))  # noqa: B905


    def hash_pdb(input_string):
        sha256_hash = hashlib.sha256()
        sha256_hash.update(input_string.encode("utf-8"))
        return sha256_hash.hexdigest()


@app.function(image=image)
def fast_relax(pdb_str: str) -> dict:
    temp_pdb_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")  # noqa: SIM115
    Path(temp_pdb_path.name).write_text(pdb_str)
    pose = pyrosetta.pose_from_file(temp_pdb_path.name)
    xml_parser = pyrosetta.rosetta.protocols.rosetta_scripts.RosettaScriptsParser()
    # This xml path comes from the git clone in the image setup above.
    protocol = xml_parser.generate_mover(
        "/dl_binder_design/mpnn_fr/RosettaFastRelaxUtil.xml"
    )
    protocol.apply(pose)
    output_file_path = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w+", delete=False, suffix=".pdb"
    )
    pose.dump_pdb(output_file_path.name)
    pdb_str = Path(output_file_path.name).read_text()
    os.remove(output_file_path.name)
    os.remove(temp_pdb_path.name)
    return {"pdb_str": pdb_str}


@app.function(image=image, secrets=[modal.Secret.from_name("gcp-proteincrow")])
def get_dssp_of_a_structure_with_pdb_path(pdb_string: str) -> dict[str, Any]:
    dssp_alphabet = {
        "H": "Helix",
        "E": "Beta Strand",
        "L": "Loop/Other",
    }
    input_pdb_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pdb")  # noqa: SIM115
    input_pdb_file.write(pdb_string)
    input_pdb_file.flush()
    pose = pyrosetta.pose_from_pdb(input_pdb_file.name)
    pyrosetta.rosetta.protocols.moves.DsspMover().apply(pose)
    secondary_structure = pose.secstruct()

    residue_annotations = dict(enumerate(secondary_structure))

    # Process the secondary structure to identify ranges
    description_list = []
    current_ss = secondary_structure[0]
    start_res = 0

    def make_region_tag(dssp_code: str, start: int, end: int) -> str:
        # Represents a secondary structure region in XML format
        dssp_desc_code = dssp_alphabet.get(dssp_code, "unknown")
        return f"- Region from residues {start} to {end} is {dssp_desc_code} \n"

    for i, ss in enumerate(secondary_structure[1:], start=1):
        if ss != current_ss:
            description_list.append(make_region_tag(current_ss, start_res, i - 1))
            start_res = i
            current_ss = ss

    # Add the last range
    description_list.append(
        make_region_tag(current_ss, start_res, len(secondary_structure) - 1)
    )

    # TODO: maybe use XMLTree instead of making this string by hand?
    description = textwrap.indent("\n".join(description_list), "  ") + "\n"
    return {"residue_annotations": residue_annotations, "description": description}


@web_app.post("/rosetta/{name}")
async def rosetta_endpoint(name: str, json_data: dict) -> JSONResponse:
    blob = {}
    if name == "fast_relax":
        blob = await fast_relax.remote.aio(json_data["pdb_str"])
    elif name == "dssp":
        blob = await get_dssp_of_a_structure_with_pdb_path.remote.aio(json_data["pdb_str"])
    return JSONResponse(content=blob)
