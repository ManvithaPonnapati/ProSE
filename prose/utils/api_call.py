import aiohttp

ALPHAFOLD_URL = "https://molecularmachines--colabfold-fastapi-app.modal.run"  # Replace with your actual URL
ESM_URL = "https://molecularmachines--esm-fastapi-app.modal.run"  # Replace with your actual URL
MPNN_URL = "https://molecularmachines--colabdesign-fastapi-app.modal.run"  # Replace with your actual URL

async def post_sample_request(url: str,payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer ",
        "Content-Type": "application/json"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            response_json = await response.json()
            return response_json

