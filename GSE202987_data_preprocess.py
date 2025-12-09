"""
Script for preprocessing GSE202987 data for Waddington-OT
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csr_matrix
import os
import gzip


# File path configuration

STAGE_FILES = {
    12: {
        'matrix': 'data_1/stage12_matrix.mtx.gz',
        'barcodes': 'data_1/stage12_barcodes.tsv.gz',
        'features': 'data_1/stage12_features.tsv.gz'
    },
    14: {
        'matrix': 'data_1/stage14_matrix.mtx.gz',
        'barcodes': 'data_1/stage14_barcodes.tsv.gz',
        'features': 'data_1/stage14_features.tsv.gz'
    },
    16: {
        'matrix': 'data_1/stage16_matrix.mtx.gz',
        'barcodes': 'data_1/stage16_barcodes.tsv.gz',
        'features': 'data_1/stage16_features.tsv.gz'
    }
}

OUTPUT_DIR = 'wot_input_new_D_melanogaster/'


# Helper Functions

def read_tsv(filepath):
    """Read tsv files, handling gzipped files automatically."""
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt') as f:
            return pd.read_csv(f, sep='\t', header=None)
    else:
        return pd.read_csv(filepath, sep='\t', header=None)


def load_mtx(filepath):
    """Load and automatically handle gzipped mtx files."""
    return mmread(filepath)


def load_stage(stage, files_dict):
    """
    Load a single stage from explicit file paths.
    
    Parameters
    ----------
    stage : int
        Stage/timepoint number
    files_dict : dict
        Dictionary with 'matrix', 'barcodes', 'features' keys
        
    Returns
    -------
    AnnData object
    """
    print(f"\nLoading stage {stage}...")
    
    # Load matrix (in 10x format: genes × cells)
    print(f"  Matrix: {files_dict['matrix']}")
    matrix = load_mtx(files_dict['matrix'])
    
    # 10x matrices are genes × cells, we need cells × genes
    matrix = matrix.T.tocsr()
    
    # Load barcodes (cell IDs)
    print(f"  Barcodes: {files_dict['barcodes']}")
    barcodes_df = read_tsv(files_dict['barcodes'])
    barcodes = barcodes_df[0].values
    
    # Load features (gene info)
    print(f"  Features: {files_dict['features']}")
    features_df = read_tsv(files_dict['features'])
    
    # Features file typically has: gene_id, gene_name, feature_type
    # Or just: gene_id, gene_name
    # Or sometimes just: gene_name
    if features_df.shape[1] >= 2:
        gene_ids = features_df[0].values
        gene_names = features_df[1].values
    else:
        gene_ids = features_df[0].values
        gene_names = features_df[0].values
    
    # Create AnnData
    adata = sc.AnnData(X=matrix)
    adata.obs_names = pd.Index(barcodes)
    adata.var_names = pd.Index(gene_names)
    adata.var['gene_ids'] = gene_ids
    
    # Make var names unique (in case of duplicate gene names)
    adata.var_names_make_unique()
    
    # Add stage metadata
    adata.obs['day'] = stage
    adata.obs['stage'] = f'stage{stage}'
    adata.obs['original_barcode'] = barcodes
    
    # Make barcodes unique by prepending stage
    adata.obs_names = pd.Index([f"s{stage}_{bc}" for bc in adata.obs_names])
    
    print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    
    return adata


# ============================================================================
# NEUROGENESIS FILTERING
# ============================================================================

# CNS/neurogenesis marker genes from Seroka et al. 2022
# These genes identify cells involved in neurogenesis
CNS_MARKERS = {
    'neuroblast': ['mira', 'dpn', 'wor', 'ase', 'insc'],  # Neural progenitors
    'gmc': ['tap', 'pros'],  # Ganglion mother cells
    'neuron_newborn': ['Hey', 'nSyb'],  # Newborn neurons
    'neuron_mature': ['brp', 'nSyb', 'elav'],  # Mature neurons
    'glia': ['repo', 'gcm'],  # Glial cells (part of neurogenesis)
}

# All unique neurogenesis markers
ALL_NEURO_MARKERS = list(set(
    gene for markers in CNS_MARKERS.values() for gene in markers
))


def filter_neurogenesis_cells(adata, min_marker_expression=0.5, min_markers_detected=1):
    """
    Filter cells to keep only those involved in neurogenesis.
    
    Uses marker genes from Seroka et al. 2022 to identify CNS cells:
    - Neuroblasts (mira+, dpn+, wor+, ase+, insc+)
    - GMCs (tap+, pros+)
    - Newborn neurons (Hey+, nSyb+)
    - Mature neurons (brp+, nSyb+, elav+)
    - Glia (repo+, gcm+)
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (should be raw counts or normalized)
    min_marker_expression : float
        Minimum expression threshold to consider a marker "detected"
    min_markers_detected : int
        Minimum number of markers that must be detected to keep a cell
        
    Returns
    -------
    AnnData with only neurogenesis cells
    """
    print(f"  Using {len(ALL_NEURO_MARKERS)} neurogenesis marker genes")
    
    # Find which markers are present in the dataset
    available_markers = [g for g in ALL_NEURO_MARKERS if g in adata.var_names]
    missing_markers = [g for g in ALL_NEURO_MARKERS if g not in adata.var_names]
    
    print(f"  Found {len(available_markers)}/{len(ALL_NEURO_MARKERS)} markers in dataset")
    if missing_markers:
        print(f"  Missing markers: {', '.join(missing_markers)}")
    
    if len(available_markers) == 0:
        print("  WARNING: No neurogenesis markers found! Returning all cells.")
        return adata
    
    # Calculate expression of each marker per cell
    # Use raw counts if available, otherwise use .X
    if 'counts' in adata.layers:
        expr_matrix = adata[:, available_markers].layers['counts']
    else:
        expr_matrix = adata[:, available_markers].X
    
    # Convert to dense if sparse
    if hasattr(expr_matrix, 'toarray'):
        expr_matrix = expr_matrix.toarray()
    
    # Count how many markers are expressed above threshold per cell
    markers_detected = (expr_matrix > min_marker_expression).sum(axis=1)
    
    # Create mask for neurogenesis cells
    neuro_mask = markers_detected >= min_markers_detected
    
    # Add cell type annotation based on marker expression
    adata.obs['n_neuro_markers'] = markers_detected
    adata.obs['is_neurogenesis'] = neuro_mask
    
    # Annotate likely cell type based on strongest markers
    cell_types = []
    for i in range(adata.n_obs):
        if not neuro_mask[i]:
            cell_types.append('non_neural')
        else:
            # Check which marker category has highest expression
            best_type = 'neural_unspecified'
            best_score = 0
            for cell_type, markers in CNS_MARKERS.items():
                type_markers = [m for m in markers if m in available_markers]
                if type_markers:
                    marker_idx = [available_markers.index(m) for m in type_markers]
                    score = expr_matrix[i, marker_idx].mean()
                    if score > best_score:
                        best_score = score
                        best_type = cell_type
            cell_types.append(best_type)
    
    adata.obs['neural_cell_type'] = cell_types
    
    # Filter to keep only neurogenesis cells
    adata_neuro = adata[neuro_mask, :].copy()
    
    # Report cell type composition
    print(f"\n  Cell type composition after filtering:")
    for ct in ['neuroblast', 'gmc', 'neuron_newborn', 'neuron_mature', 'glia', 'neural_unspecified']:
        n = (adata_neuro.obs['neural_cell_type'] == ct).sum()
        if n > 0:
            print(f"    {ct}: {n} cells")
    
    return adata_neuro


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 60)
    print("WADDINGTON-OT PREPROCESSING")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all stages
    adata_list = []
    for stage, files in STAGE_FILES.items():
        try:
            adata = load_stage(stage, files)
            adata_list.append(adata)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue
    
    if len(adata_list) < 2:
        raise ValueError("Need at least 2 timepoints!")
    
    # Combine stages
    print("\n" + "-" * 60)
    print("Combining stages...")
    adata = sc.concat(adata_list, join='inner')
    print(f"Combined dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # QC and filtering
    print("\n" + "-" * 60)
    print("Quality control...")
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    
    # Filter
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs['pct_counts_mt'] < 20, :].copy()
    print(f"Filtered: {n_before} → {adata.n_obs} cells")
    
    # Filter for neurogenesis cells only
    # Based on Seroka et al. 2022 (GSE202987) marker genes
    print("\n" + "-" * 60)
    print("Filtering for neurogenesis cells...")
    adata = filter_neurogenesis_cells(adata)
    print(f"After neurogenesis filter: {adata.n_obs} cells")
    
    # Normalize
    print("\n" + "-" * 60)
    print("Normalizing...")
    adata.layers['counts'] = adata.X.copy()  # Store raw counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Variable genes
    # Convert 'day' to categorical (required for batch_key in highly_variable_genes)
    adata.obs['day'] = adata.obs['day'].astype('category')
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='day')
    n_hvg = adata.var['highly_variable'].sum()
    print(f"Highly variable genes: {n_hvg}")
    
    # Dimensionality reduction for visualization
    print("\n" + "-" * 60)
    print("Computing embeddings...")
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.umap(adata)
    
    # Save outputs
    print("\n" + "-" * 60)
    print("Saving files...")
    
    # Full matrix
    adata.write(f'{OUTPUT_DIR}expression_matrix.h5ad')
    print(f"  {OUTPUT_DIR}expression_matrix.h5ad")
    
    # HVG matrix for wot
    adata_hvg = adata[:, adata.var['highly_variable']].copy()
    adata_hvg.write(f'{OUTPUT_DIR}var_genes_matrix.h5ad')
    print(f"  {OUTPUT_DIR}var_genes_matrix.h5ad")
    
    # Cell days file (REQUIRED for wot)
    cell_days = pd.DataFrame({
        'id': adata.obs_names,
        'day': adata.obs['day']
    })
    cell_days.to_csv(f'{OUTPUT_DIR}cell_days.txt', sep='\t', index=False)
    print(f"  {OUTPUT_DIR}cell_days.txt")
    
    # UMAP coordinates for visualization
    umap_df = pd.DataFrame(
        adata.obsm['X_umap'],
        index=adata.obs_names,
        columns=['UMAP1', 'UMAP2']
    )
    umap_df.to_csv(f'{OUTPUT_DIR}umap_coords.txt', sep='\t')
    print(f"  {OUTPUT_DIR}umap_coords.txt")
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    
    print("\nCells per timepoint:")
    for day in sorted(adata.obs['day'].unique()):
        n = (adata.obs['day'] == day).sum()
        print(f"  Stage {day}: {n} cells")
    
    print(f"""
\nRun Waddington-OT:
------------------
wot optimal_transport \\
    --matrix {OUTPUT_DIR}var_genes_matrix.h5ad \\
    --cell_days {OUTPUT_DIR}cell_days.txt \\
    --out {OUTPUT_DIR}tmaps/wot

Then compute trajectories:
wot trajectory \\
    --tmap {OUTPUT_DIR}tmaps/wot \\
    --cell_set cell_sets.gmt \\
    --day 16 \\
    --embedding {OUTPUT_DIR}umap_coords.txt
""")
    
    return adata


if __name__ == '__main__':
    adata = main()

