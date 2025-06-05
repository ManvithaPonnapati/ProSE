from math import sqrt

import networkx as nx
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB import PPBuilder
from Bio.SeqUtils import seq1
from aiohttp import ClientSession
from tmtools import tm_align

from prose.modalmol.sequence import Sequence
from prose.utils.api_call import PRODY_URL


def load_structure(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    return structure


def tm_align_two_pdbs(pdb_file_path_1: str, pdb_file_path_2: str):
    pdb_1_atom_array = load_structure(pdb_file_path_1)
    pdb_2_atom_array = load_structure(pdb_file_path_2)
    pdb_1_coords = pdb_1_atom_array.coord
    pdb_2_coords = pdb_2_atom_array.coord
    sequence_1 = extract_sequence(pdb_1_atom_array)
    sequence_2 = extract_sequence(pdb_2_atom_array)
    res = tm_align(pdb_1_coords, pdb_2_coords, sequence_1, sequence_2)
    return res.rmsd, res.tm_norm_chain1, res.tm_norm_chain2


def radius_of_gyration(structure):
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
    coords = np.array(coords)
    centroid = np.mean(coords, axis=0)
    rg = sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))
    return rg


def compute_betweenness_centrality(structure, cutoff=8.0):
    G = nx.Graph()
    coords = []
    res_indices = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
                    res_indices.append((chain.id, residue.id[1]))

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            distance = np.linalg.norm(coords[i] - coords[j])
            if distance <= cutoff:
                G.add_edge(res_indices[i], res_indices[j])

    bc = nx.betweenness_centrality(G)
    return bc


def extract_sequence(structure):
    ppb = PPBuilder()
    sequence = ""
    for pp in ppb.build_peptides(structure):
        sequence += seq1(pp.get_sequence())
    return sequence


async def analyze_sequence(sequence, namespace="test", msa: bool = False):
    seq_obj = await Sequence.from_sequence(sequence)
    msa_output_string = ""
    if msa:
        msa_output_string = await seq_obj.get_msa()
    folded_structure, plddt = await seq_obj.fold_sequence()
    with open(f"{namespace}_folded_structure.pdb", "w") as f:
        f.write(folded_structure)
    return {
        "MSA": msa_output_string,
        "Folded Structure": folded_structure,
        "pLDDT": plddt
    }


async def analyze_pdb(pdb_file, namespace="test"):
    structure = load_structure(pdb_file)
    rg = radius_of_gyration(structure)
    bc = compute_betweenness_centrality(structure)
    sequence = extract_sequence(structure)

    return {
        "Radius of Gyration": rg,
        "Betweenness Centrality": bc,
        "Sequence": sequence
    }


async def get_bonds_in_monomer_prody(
        pdb_file_path: str, sequence: str
) -> str:
    async with (
        ClientSession() as session,
        session.post(
            f"{PRODY_URL}/prody/get_bonds_monomer",
            json={"pdb_str": open(pdb_file_path).read(), "sequence": sequence},
            timeout=60 * 60,
        ) as resp,
    ):
        resp.raise_for_status()
        response = await resp.json()
        return response


async def get_esm_pll(sequence: str):
    async with (
        ClientSession() as session,
        session.post(
            f"{PRODY_URL}/esm/pll",
            json={"sequence": sequence},
            timeout=60 * 60,
        ) as resp,
    ):
        resp.raise_for_status()
        response = await resp.json()
        return response['pll']


async def compute_rosetta_ddg(pdb_file: str, mut_sequence: str):
    structure = load_structure(pdb_file)
    wild_type_sequence = extract_sequence(structure)
    async with (
        ClientSession() as session,
        session.post(
            f"{PRODY_URL}/rosetta/fast_relax",
            json={"pdb_str": open(pdb_file).read(), "mut_sequence": mut_sequence, "wt_sequence": wild_type_sequence},
            timeout=60 * 60,
        ) as resp,
    ):
        resp.raise_for_status()
        response = await resp.json()
        return response
