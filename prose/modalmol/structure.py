import math
import os
import tempfile
import urllib.request

import biotite.structure as struc
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray
from biotite.structure.io import pdb

from prose.utils.api_call import post_sample_request, MPNN_URL


class Structure:
    def __init__(
            self,
            pdb_id: str,
            chain_id: str,
            structure: AtomArray,
            **kwargs,
    ):
        self.structure = structure
        self.pdb_id = pdb_id
        self.chain_id = chain_id
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return len(self.structure)

    @classmethod
    def from_pdb_id(cls, pdb_id, chain_id):
        temp_file_path = tempfile.NamedTemporaryFile(delete=False, mode="w+", suffix=".pdb")
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urllib.request.urlretrieve(pdb_url, temp_file_path.name)
        temp_file_path.flush()
        # pdb_file_path = rcsb.fetch(pdb_id, "pdb", gettempdir())
        pdb_file = pdb.PDBFile.read(temp_file_path.name)
        atom_array = pdb.get_structure(pdb_file)[0]
        atom_array = atom_array[struc.filter_amino_acids(atom_array)]
        if chain_id:
            atom_array = atom_array[atom_array.chain_id == chain_id]
        os.remove(temp_file_path.name)
        return cls(
            structure=atom_array,
            pdb_id=pdb_id,
            chain_id=chain_id)

    @classmethod
    def from_pdb_file(cls, pdb_file_path, chain_id):
        temp_file_path = tempfile.NamedTemporaryFile(delete=False, mode="w+", suffix=".pdb")
        atom_array = pdb.PDBFile.read(pdb_file_path).get_structure(model=1,extra_fields=["b_factor"])
        atom_array = atom_array[struc.filter_amino_acids(atom_array)]
        if chain_id:
            atom_array = atom_array[atom_array.chain_id == chain_id]
        os.remove(temp_file_path.name)
        return cls(
            structure=atom_array,
            pdb_id=os.path.basename(pdb_file_path),
            chain_id=chain_id)

    def get_sequence(self):
        res_protein_names = self.structure[self.structure.atom_name == "CA"].get_annotation(
            "res_name"
        )
        return "".join([ProteinSequence.convert_letter_3to1(x) for x in res_protein_names])

    def get_string_from_atom_array(self):
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, self.structure)
        temp_file_path = tempfile.NamedTemporaryFile(delete=False, mode="w+", suffix=".pdb")
        pdb_file.write(temp_file_path.name)
        temp_file_path.flush()
        return temp_file_path.read()

    async def get_mpnn_samples(self,constraints:list[int]=[],num_samples:int=10,temperature:float=0.1):
        mpnn_config = {
            "fix_pos": ",".join([f"{self.chain_id}{x + 1}" for x in constraints if x >= 0 and x < len(self.structure) - 1] if constraints else []),
            "inverse": False,
            "temp": float(temperature),
            "batch": int(num_samples),
            "chains": self.chain_id
        }
        mpnn_prediction_json = await post_sample_request(
            f"{MPNN_URL}/sample",
            {
                "pdb_string": self.get_string_from_atom_array(),
                "params": mpnn_config
            },
        )
        return mpnn_prediction_json

    def get_coordinates(self):
        """
        Get the coordinates of the structure.
        """
        return self.structure.coord

    def get_pairwise_distance(self,residue_idxs):
        """
        Get the pairwise distance between the residues in the structure.
        """
        if len(residue_idxs) < 2:
            raise ValueError("At least two residue indices are required to calculate pairwise distances.")
        ca_structure = self.structure[self.structure.atom_name == "CA"]
        coords = ca_structure[residue_idxs]
        pairwise_distances = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = math.sqrt(
                    sum((coords[i].coord[k] - coords[j].coord[k]) ** 2 for k in range(3))
                )
                pairwise_distances.append((i, j, dist))
        return pairwise_distances


    def average_b_factor(self):
        return float(self.structure.b_factor.mean())