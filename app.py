###############################################################################
#  Imports
###############################################################################
import streamlit as st
import pandas as pd
import py3Dmol
from stmol import showmol
from pathlib import Path
import glob
from Bio import SeqIO
import os

###############################################################################
#  Read the two CSV files ------------------------------------------------------
###############################################################################
DATA_PATH = Path("/workspaces/ProSE/app_data/")   # directory that holds app.py

anno_csv = "/workspaces/ProSE/app_data/climate_enzymes_residue_annotations.csv"
desc_csv = "/workspaces/ProSE/app_data/climate_enzymes.csv"

# # fail loudly if either file is missing ---------------------------------------
# if not anno_csv.exists() or not desc_csv.exists():
#     missing = " and ".join(
#         [f"**`{p.name}`**" for p in (anno_csv, desc_csv) if not p.exists()]
#     )
#     st.error(f"💥  Required data file(s) {missing} not found next to `app.py`.")
#     st.stop()

anno_df = pd.read_csv(anno_csv).fillna("")
desc_df = pd.read_csv(desc_csv).fillna("")

###############################################################################
#  Function to read FASTA sequences -------------------------------------------
###############################################################################
def read_fasta_sequences(pdb_id):
    """Read MPNN designed sequences for a given PDB ID"""
    sequences = []
    designed_seq_path = f"/workspaces/ProSE/app_data/designed_sequences/"
    
    # Look for FASTA files matching the pattern
    pattern = f"{designed_seq_path}{pdb_id}_*_mpnn_temp_*.fasta"
    fasta_files = glob.glob(pattern)
    
    for fasta_file in fasta_files:
        filename = os.path.basename(fasta_file)
        # Extract temperature from filename
        temp = filename.split('temp_')[1].split('.fasta')[0] if 'temp_' in filename else 'unknown'
        
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                sequences.append({
                    'Sequence_ID': record.id,
                    'Temperature': temp,
                    'Sequence': str(record.seq),
                    'Length': len(record.seq),
                    'Filename': filename
                })
        except Exception as e:
            st.warning(f"Could not read {filename}: {str(e)}")
    
    return sequences

###############################################################################
#  Pretty-print helpers --------------------------------------------------------
###############################################################################
BOOL_COLS = [
    "important_for_function_residue",
    "mutation_improved_function",
    "mutation_improved_thermostability",
    "mutation_affected_conformation",
    "active_site",
    "conserved_site",
    "ligand_binding_site",
    "allosteric_site",
]

def highlight_true(v):
    return "background-color:#b8e1b4" if str(v).upper() == "TRUE" else ""

###############################################################################
#  3-D viewer helper -----------------------------------------------------------
###############################################################################
def show_st_3dmol(
    pdb_code,
    style_lst=None,
    label_lst=None,
    reslabel_lst=None,
    zoom_dict=None,
    surface_lst=None,
    cartoon_style="trace",
    cartoon_radius=0.2,
    cartoon_color="lightgray",
    zoom=1,
    spin_on=False,
    width=900,
    height=600,
):

    """
    Show 3D view of protein structure from the
    Protein Data Bank (PDB)

    Parameters
    ----------
    pdb_code: str
        Four-letter code of protein structure in the PDB
        (e.g., 5P21)
    style_lst: list of lists of dicts
        A nested list with each sublist containing a
        selection dictionary at index 0 and coloring
        dictionary at index 1
    label_lst: list of lists of dicts
        A nested list with each sublist containing a
        label string at index 0, coloring dictionary
        at index 1, and selection dictionary at
        index 2
    reslabel_lst: list of lists of dicts
        A nested list with each sublist containing a
        selection dictionary at index 0 and coloring
        dictionary at index 1
    zoom_dict: dict
    surface_lst: list of lists of dicts
        A nested list with each sublist containing a
        coloring dictionary at index 0 and selection
        dictionary at index 1
    cartoon_style: str
        Style of protein structure backbone cartoon
        rendering, which can be "trace", "oval", "rectangle",
        "parabola", or "edged"
    cartoon_radius: float
        Radius of backbone cartoon rendering
    cartoon_color: str
        Color of backbone cartoon rendering
    zoom: float
        Level of zoom into protein structure
        in unit of Angstroms
    spin_on: bool
        Boolean specifying whether the visualized
        protein structure should be continually
        spinning (True) or not (False)
    width: int
        Width of molecular viewer
    height: int
        Height of molecular viewer
    """
    view = py3Dmol.view(query=f"pdb:{pdb_code.lower()}", width=width, height=height)

    view.setStyle(
        {
            "cartoon": {
                "style": cartoon_style,
                "color": cartoon_color,
                "thickness": cartoon_radius,
            }
        }
    )

    if surface_lst is not None:
        for surface in surface_lst:
            view.addSurface(py3Dmol.VDW, surface[0], surface[1])

    if style_lst is not None:
        for style in style_lst:
            view.addStyle(
                style[0],
                style[1],
            )

    if label_lst is not None:
        for label in label_lst:
            view.addLabel(label[0], label[1], label[2])

    if reslabel_lst is not None:
        for reslabel in reslabel_lst:
            view.addResLabels(reslabel[0], reslabel[1])

    if zoom_dict is None:
        view.zoomTo()
    else:
        view.zoomTo(zoom_dict)

    view.spin(spin_on)

    view.zoom(zoom)
    showmol(view, height=height, width=width)

st.set_page_config(page_title="PaperQA Protein Annotation", layout="wide")
st.title("🧬 Protein Residue Annotation with PaperQA")

# ─────────────────────────────────────────────────────────────────────────────
# Top-of-page selector (replaces sidebar)
# ─────────────────────────────────────────────────────────────────────────────
pdb_options = anno_df["pdb_id"].unique().tolist()
current_pdb = st.selectbox("Select a PDB entry", pdb_options)

# Fetch descriptor & annotations for the chosen PDB
desc_row = desc_df[desc_df["pdb_id"] == current_pdb].iloc[0]
subset   = anno_df[anno_df["pdb_id"] == current_pdb].reset_index(drop=True)

###############################################################################
#  Two-column layout  (80 % / 20 %)
###############################################################################
left, right = st.columns((4, 1))

# ───────── LEFT : info card + annotation grid ───────────────────────────────
with left:
    st.markdown(
        f"""
### **{desc_row['protein_name']}**
**PDB:** `{desc_row['pdb_id']}` | **Chain:** `{desc_row['chain']}`  

{desc_row['description']}
"""
    )

    # Residue "cards" in a responsive grid
    n_cols   = 3
    card_row = st.columns(n_cols, gap="small")

    for i, r in enumerate(subset.itertuples(index=False)):
        col = card_row[i % n_cols]

        with col:
            with st.container(border=True):
                st.markdown(f"### **{r.residue}**", help="Residue number")
                st.markdown(f"**References:** {r.reference}")
                st.markdown(f"**Explanation**  \n{r.explanation}")

                if r.mutation:
                    st.markdown(f"**Mutation:** `{r.mutation}`")

                flags = {
                    "Important": r.important_for_function_residue,
                    "Improved function": r.mutation_improved_function,
                    "Improved thermostability": r.mutation_improved_thermostability,
                    "Affected conformation": r.mutation_affected_conformation,
                    "Active site": r.active_site,
                    "Conserved site": r.conserved_site,
                    "Ligand binding": r.ligand_binding_site,
                    "Allosteric site": r.allosteric_site,
                }
                chips = "  ".join(
                    f"✅ *{name}*" for name, val in flags.items()
                    if str(val).upper() == "TRUE"
                )
                if chips:
                    st.markdown(chips)

                with st.expander("Full LLM answer", expanded=False):
                    st.markdown(r.original_answer)

        # start a new row every n_cols cards
        if (i + 1) % n_cols == 0 and i + 1 < len(subset):
            card_row = st.columns(n_cols, gap="small")

# ───────── RIGHT : smaller 3-D viewer ────────────────────────────────────────
with right:
    st.subheader("3-D structure")
    show_st_3dmol(
        current_pdb.lower(),
        spin_on=True,
        zoom=1.2,
        width=300,
        height=300,
    )

###############################################################################
#  MPNN Designed Sequences Section --------------------------------------------
###############################################################################
st.markdown("---")
st.header("🧪 MPNN Designed Sequences")

# Read designed sequences for the current PDB
designed_sequences = read_fasta_sequences(current_pdb)

if designed_sequences:
    # Create DataFrame with designed sequences
    seq_df = pd.DataFrame(designed_sequences)
    
    # Add empty columns for the requested analyses
    seq_df['Salt_Bridges'] = ''
    seq_df['H_Bonds'] = ''
    seq_df['Secondary_Structure'] = ''
    seq_df['Radius_of_Gyration'] = ''
    seq_df['ESM2_Log_Likelihood'] = ''
    seq_df['AlphaFold2_Confidence'] = ''
    seq_df['Boltz1_Structure_Prediction'] = ''
    seq_df['Evolutionary_Consensus'] = ''
    seq_df['Rosetta_Stability_ddG'] = ''
    
    # Reorder columns for better display
    display_columns = [
        'Sequence_ID', 'Temperature', 'Length', 'Sequence',
        'Salt_Bridges', 'H_Bonds', 'Secondary_Structure', 
        'Radius_of_Gyration', 'ESM2_Log_Likelihood', 
        'AlphaFold2_Confidence', 'Boltz1_Structure_Prediction',
        'Evolutionary_Consensus', 'Rosetta_Stability_ddG'
    ]
    
    # Display summary
    st.markdown(f"**Found {len(designed_sequences)} designed sequences for PDB {current_pdb}**")
    
    # Group by temperature if multiple temperatures exist
    if len(seq_df['Temperature'].unique()) > 1:
        st.markdown("**Sequences grouped by temperature:**")
        for temp in sorted(seq_df['Temperature'].unique()):
            temp_subset = seq_df[seq_df['Temperature'] == temp]
            st.markdown(f"*Temperature: {temp}*")
            st.dataframe(
                temp_subset[display_columns],
                use_container_width=True,
                hide_index=True
            )
            st.markdown("")
    else:
        # Display all sequences in one table
        st.dataframe(
            seq_df[display_columns],
            use_container_width=True,
            hide_index=True
        )
    
    # Add analysis placeholder section
    st.markdown("### 📊 Analysis Pipeline")
    st.info("""
    **Analysis:**
    - **ProDy Analysis**: Salt bridges and hydrogen bonds calculation
    - **Secondary Structure**: DSSP-based structure prediction
    - **Radius of Gyration**: Structural compactness measurement
    - **ESM2 Log Likelihood**: Sequence probability scoring
    - **AlphaFold2 Confidence**: Structure prediction confidence
    - **Boltz-1 Structure Prediction**: Protein-ligand complex prediction
    - **Evolutionary Consensus**: Conservation analysis
    - **Rosetta Stability ddG**: Thermodynamic stability prediction
    """)
    st.markdown("### 📊 Experimental Library Selection")
    
else:
    st.warning(f"No designed sequences found for PDB {current_pdb}. Expected files matching pattern: `/workspaces/ProSE/app_data/designed_sequences/{current_pdb}_*_mpnn_temp_*.fasta`")
    
    # Show what files are available
    designed_seq_path = "/workspaces/ProSE/app_data/designed_sequences/"
    if os.path.exists(designed_seq_path):
        available_files = os.listdir(designed_seq_path)
        if available_files:
            st.markdown("**Available files in designed_sequences directory:**")
            for file in sorted(available_files):
                st.markdown(f"- {file}")
        else:
            st.markdown("*No files found in designed_sequences directory*")
    else:
        st.error(f"Directory {designed_seq_path} does not exist")