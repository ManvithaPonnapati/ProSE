import os
import asyncio

from prose.modalmol.structure import Structure
from prose.modalmol.sequence import Sequence

pdb_file = "2vvb/2vvb_A.pdb"

structure_in_pdb = Structure.from_pdb_file(pdb_file, chain_id="A")
sequence_of_pdb = structure_in_pdb.get_sequence()
#mpnn samples
# mpnn_samples = asyncio.run(structure_in_pdb.get_mpnn_samples(constraints=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                                                              num_samples=10, temperature=0.1))
#fold sequence
sequence_structure_prediction = Sequence.from_sequence(sequence_of_pdb)
print(sequence_structure_prediction.fold_sequence())


