# ProSE
Protein Sequence Evolver

> One-command library for generating sequence libraries for improving thermostability of an enzyme using state of the art deep learning methods

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)  

## ✨ What it does
Given

* **Backbone** – a reference **PDB** file  
* **Sequence** – the corresponding amino-acid string (FASTA or plain text)  
* **Functional sites** – one or more residue indices (0-indexed)

the sampler builds a **diverse, high-quality library** of candidate sequences in a
single call

## 🏗️ Installation
Because all build-time metadata and dependencies live in **`pyproject.toml`** ,
you can install the package straight from the source tree:

```bash
# clone
git clone https://github.com/<your-username>/prose.git
cd prose

# install 
pip install -e .         

# ----- OR, if you use Poetry -----
poetry install
```

## 🧪 Sample


# Acknowledgement 

The development of this tool was made possible by a grant from **Homeward Bio**
