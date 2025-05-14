from prose.utils.api_call import post_sample_request, ALPHAFOLD_URL, ESM_URL

class Sequence:
    """
    copied from moleculib and edited to work with prose. Might merge it in later
    and import from there directly
    """

    def __init__(
            self,
            sequence: str,
            fasta_file: str,
            **kwargs,
    ):
        self.sequence = sequence
        self.fasta_file = fasta_file
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return len(self.sequence)

    @classmethod
    def from_sequence(cls, sequence):
        """
        Create a sequence object from a sequence string
        """
        return cls(sequence=sequence, fasta_file="")

    @classmethod
    def from_fasta_file(cls, fasta_file):
        """
        Create a sequence object from a fasta file
        """
        with open(fasta_file, "r") as f:
            lines = f.readlines()
        sequence = "".join([line.strip() for line in lines[1:]])
        return cls(sequence=sequence, fasta_file=fasta_file)

    async def fold_sequence(self):
        af2_prediction_json = await post_sample_request(
            f"{ALPHAFOLD_URL}/alphafold/monomer",
            {
                "sequence": self.sequence
            },
        )
        return af2_prediction_json['results']['predicted_output'], af2_prediction_json['results']['confidence']

    async def compute_esm_pll(self):
        esm_prediction_json = await post_sample_request(
            f"{ESM_URL}/esm/pll",
            {
                "sequence": self.sequence
            },
        )
        return esm_prediction_json['pll']

    async def get_msa(self):
        esm_prediction_json = await post_sample_request(
            f"{ALPHAFOLD_URL}/esm/msa",
            {
                "sequence": self.sequence
            },
        )
        return esm_prediction_json['msa']

    async def get_mcmc_samples(self,constraints:list[int]=[],num_samples:int=10,temperature:float=0.1):
        """
        Get MCMC samples from the sequence by masking the sequence at all locations but the constrained residues

        Args:
            constraints (list[int]): list of indices to constrain 0-indexed
            num_samples (int): number of samples to generate
            temperature (float): temperature for sampling
        Returns:
            list[str]: list of sampled sequences
        """
        esm_prediction_json = await post_sample_request(
            f"{ESM_URL}/esm/mcmc",
            {
                "sequence": self.sequence,
                "constraints": constraints,
                "num_samples": num_samples,
                "temperature": temperature
            },
        )
        return esm_prediction_json['samples']

    async def get_evodiff_samples(self,msa):
        esm_prediction_json = await post_sample_request(
            f"{ALPHAFOLD_URL}/evodiff/sample",
            {
                "sequence": self.sequence,
                "msa": msa
            },
        )
        return esm_prediction_json['samples']

    def compare_to_sequences(self, sequences):
        """
        Compare the sequence to a list of sequences and return the best match
        """
        best_match = None
        best_score = 0
        for seq in sequences:
            #compute similarity score
            # This is a placeholder for the actual comparison logic
            score = sum(1 for a, b in zip(self.sequence, seq) if a == b) / len(self.sequence)
            if score > best_score:
                best_match = seq
                best_score = score
        return best_match, best_score


