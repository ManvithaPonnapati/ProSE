import os
import os.path
import subprocess
import uuid
from collections import defaultdict

from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app

app = App("cartddg")
web_app = FastAPI()


@web_app.exception_handler(RequestValidationError)
def validation_exception_handler(exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )

image = (
    Image.debian_slim(python_version="3.9")
    .micromamba()
    .apt_install("wget", "git", "curl")
    .run_commands(
        "wget https://downloads.rosettacommons.org/downloads/academic/3.12/rosetta_bin_linux_3.12_bundle.tgz; tar "
        "-xvzf rosetta_bin_linux_3.12_bundle.tgz"
    )
    .run_commands("git clone https://github.com/ELELAB/RosettaDDGPrediction.git")
    .pip_install(
        "google-cloud-storage",
        "google-auth",
    )
)

rosetta_path = os.path.join(
    os.getcwd(), "rosetta_bin_linux_3.12_bundle", "main/source/bin"
)
path_to_executable = "/rosetta_bin_linux_2020.08.61146_bundle/main/source/bin/cartesian_ddg.static.linuxgccrelease"
path_to_rosetta_scripts = (
    "/rosetta_bin_linux_2020.08.61146_bundle/main/source/bin/rosetta_scripts.static"
    ".linuxgccrelease"
)
# cartDDG monomer protocol params - These should not be changed unless you know what you are doing.
cartddg_ref2015_protocol = (
    "-in:file:s {pdb_file_name} -ddg:mut_file {mut_list_filename} -ddg:iterations 3 "
    "-ddg::score_cutoff 1.0 -ddg:bbnbrs 1 -fa_max_dis 9.0 -score:weights ref2015_cart"
)

@app.function(image=image, timeout=60 * 60, concurrency_limit=100)
def compute_ddg_monomer(json_data: dict) -> dict:
    """
    Uses Rosetta to predict the change in free energy upon mutation of a protein structure,
    calculating the stability difference if it is a monomeric protein.
    """

    def parse_ddg_list(lines):
        """Parses the lines to extract scores based on 'WT' and 'MUT' identifiers."""
        scores = defaultdict(list)
        for line in lines:
            if "COMPLEX:" in line:
                parts = line.split()
                try:
                    identifier = parts[2]
                    total_score = float(parts[3])
                    scores[identifier].append(total_score)
                except (IndexError, ValueError) as e:
                    print(
                        f"Warning: Skipping line due to parsing error: {line}. Error: {e}"
                    )
        return scores

    def calculate_average(scores):
        """Calculates the average of each list of scores for WT and MUT, returning the ddG."""
        wt_scores = [
            sum(scores[key]) / len(scores[key])
            for key in scores
            if key.startswith("WT")
        ]
        mut_scores = [
            sum(scores[key]) / len(scores[key])
            for key in scores
            if key.startswith("MUT")
        ]
        return (
            (sum(mut_scores) / len(mut_scores) - sum(wt_scores) / len(wt_scores))
            if wt_scores and mut_scores
            else 0
        )

    unique_id = str(uuid.uuid4())
    pdb_file_path = f"{unique_id}.pdb"
    mut_list_path = f"{unique_id}.mut_list"

    # Write files to disk. Not using tempfiles due to Subprocess visibility issues
    with open(pdb_file_path, "w") as pdb_file, open(mut_list_path, "w") as mut_file:  # noqa: FURB103
        pdb_file.write(json_data["pdb_string"])
        mut_file.write(json_data["mutation_string"])

    # Run the Rosetta DDG protocol
    print("Running Rosetta DDG protocol")
    cart_ddg_command = f"{path_to_executable} {cartddg_ref2015_protocol.format(pdb_file_name=pdb_file_path, mut_list_filename=mut_list_path)}"
    print(f"Running command: {cart_ddg_command}")
    result = subprocess.run(  # noqa: S602
        cart_ddg_command, capture_output=True, shell=True, check=False
    )
    print("Protocol run complete")
    print("-----")
    print(os.listdir("."))
    try:
        mut_list_path_ddg = f"{mut_list_path.replace('.mut_list', '.ddg')}"
        print("mut_list_path_ddg", mut_list_path_ddg)
        with open(mut_list_path_ddg) as ddg_file:
            ddg = calculate_average(parse_ddg_list(ddg_file.readlines()))

        return {"ddg": ddg, "message": "Protocol completed successfully."}  # noqa: TRY300
    except Exception as e:
        crash_message = "Unknown error occurred."
        if os.path.exists("ROSETTA_CRASH.log"):
            with open("ROSETTA_CRASH.log") as log_file:
                crash_message = "".join(log_file.readlines()[-50:])
                print(crash_message)
        print(f"Error: {e}")
        return {"ddg": 0.0, "message": f"Protocol failed: {crash_message}"}



@web_app.post("/compute/{protocol}")
async def compute(json_data: dict, protocol: str = ""):
    if protocol == "ddg_monomer":
        blob = await compute_ddg_monomer.remote.aio(json_data)
    else:
        blob = {
            "message": "Invalid protocol. Please choose either 'ddg_monomer' or 'ddg_complex'."
        }
    return JSONResponse(content=blob)


@app.function(secrets=[], timeout=60 * 60)
@asgi_app()
def endpoint() -> FastAPI:
    return web_app
