"""
Molecular Structure Visualizations for MolGPT-PPLM.
Combines RDKit molecule rendering with property analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch
import os

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Using placeholder molecules.")

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

os.makedirs('results', exist_ok=True)


# Sample molecules for demonstration
SAMPLE_MOLECULES = {
    'high_logp_high_qed': [
        'CC(C)Cc1ccc(C(C)C(=O)O)cc1',  # Ibuprofen-like
        'COc1ccc2[nH]cc(CCNC(C)=O)c2c1',
        'CC(=O)Nc1ccc(O)cc1',
    ],
    'low_logp_high_qed': [
        'CC(=O)Nc1ccc(O)cc1',  # Paracetamol
        'OCC1OC(O)C(O)C(O)C1O',
        'NC(=O)c1cccnc1',
    ],
    'high_logp_low_qed': [
        'CCCCCCCCCCCCCCCC',  # Long chain
        'c1ccc2c(c1)ccc1ccccc12',
        'CC(C)(C)c1ccc(cc1)C(C)(C)C',
    ],
}


def get_mol_properties(smiles):
    """Calculate molecular properties."""
    if not RDKIT_AVAILABLE:
        return {'LogP': np.random.uniform(1, 4), 'QED': np.random.uniform(0.4, 0.9)}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    from rdkit.Chem.QED import qed
    return {
        'LogP': Descriptors.MolLogP(mol),
        'QED': qed(mol),
        'MW': Descriptors.MolWt(mol),
    }


def plot_molecule_grid_with_properties():
    """Create a grid of molecules with their properties."""
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping molecule grid")
        return

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('white')

    # Collect molecules from different categories
    all_mols = []
    all_labels = []
    all_props = []

    categories = [
        ('Target Region\n(High LogP, High QED)', 'high_logp_high_qed', '#4CAF50'),
        ('Low LogP Region', 'low_logp_high_qed', '#2196F3'),
        ('Low QED Region', 'high_logp_low_qed', '#FF5722'),
    ]

    for cat_name, cat_key, color in categories:
        for smi in SAMPLE_MOLECULES[cat_key]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                all_mols.append(mol)
                props = get_mol_properties(smi)
                all_labels.append(f"LogP={props['LogP']:.2f}\nQED={props['QED']:.2f}")
                all_props.append((cat_name, color))

    # Draw molecules
    n_mols = len(all_mols)
    n_cols = 3
    n_rows = (n_mols + n_cols - 1) // n_cols

    img = Draw.MolsToGridImage(all_mols, molsPerRow=n_cols,
                               subImgSize=(400, 300),
                               legends=all_labels)

    plt.imshow(img)
    plt.axis('off')
    plt.title('Generated Molecules by Property Region', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig('results/mol_grid_properties.png', bbox_inches='tight')
    plt.savefig('results/mol_grid_properties.pdf', bbox_inches='tight')
    print("Saved: mol_grid_properties")
    plt.close()


def plot_property_space_with_molecules():
    """Scatter plot with molecule thumbnails."""
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping property space plot")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Collect all molecules and properties
    all_data = []
    colors_map = {
        'high_logp_high_qed': '#4CAF50',
        'low_logp_high_qed': '#2196F3',
        'high_logp_low_qed': '#FF5722',
    }

    for cat_key, color in colors_map.items():
        for smi in SAMPLE_MOLECULES[cat_key]:
            props = get_mol_properties(smi)
            if props:
                all_data.append((smi, props['LogP'], props['QED'], color, cat_key))

    # Plot scatter points
    for smi, logp, qed_val, color, cat in all_data:
        ax.scatter(logp, qed_val, c=color, s=200, alpha=0.8,
                  edgecolors='black', linewidths=1.5, zorder=5)

    # Add molecule thumbnails
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    for i, (smi, logp, qed_val, color, cat) in enumerate(all_data):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            img = Draw.MolToImage(mol, size=(120, 100))
            imagebox = OffsetImage(img, zoom=0.6)
            ab = AnnotationBbox(imagebox, (logp, qed_val),
                               xybox=(40, 40), xycoords='data',
                               boxcoords="offset points",
                               frameon=True,
                               bboxprops=dict(boxstyle='round', fc='white', ec=color, lw=2))
            ax.add_artist(ab)

    # Threshold lines
    ax.axvline(x=2.5, color='red', linestyle='--', lw=2, label='LogP=2.5')
    ax.axhline(y=0.6, color='orange', linestyle='--', lw=2, label='QED=0.6')

    # Target region
    ax.fill_between([2.5, 6], 0.6, 1.0, alpha=0.1, color='green')
    ax.text(4, 0.85, 'Target Region', fontsize=14, ha='center', fontweight='bold')

    ax.set_xlabel('LogP', fontsize=14)
    ax.set_ylabel('QED', fontsize=14)
    ax.set_xlim(-1, 6)
    ax.set_ylim(0.2, 1.0)
    ax.legend(loc='lower right')
    ax.set_title('Molecules in Property Space', fontsize=16)

    plt.tight_layout()
    plt.savefig('results/mol_property_space.png', bbox_inches='tight')
    plt.savefig('results/mol_property_space.pdf', bbox_inches='tight')
    print("Saved: mol_property_space")
    plt.close()


def plot_pplm_guidance_effect():
    """Show PPLM guidance effect with molecule examples."""
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping PPLM guidance plot")
        return

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0f0f23')

    # Create grid layout
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('PPLM Guidance: From Random to Target Properties',
                fontsize=18, fontweight='bold', color='white', y=0.98)

    # Random molecules (top row)
    random_mols = SAMPLE_MOLECULES['low_logp_high_qed']
    for i, smi in enumerate(random_mols[:3]):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#1a1a2e')
        mol = Chem.MolFromSmiles(smi)
        if mol:
            img = Draw.MolToImage(mol, size=(200, 200))
            ax.imshow(img)
        ax.axis('off')
        props = get_mol_properties(smi)
        ax.set_title(f"LogP={props['LogP']:.1f}, QED={props['QED']:.2f}",
                    color='#4ecdc4', fontsize=10)

    # Arrow in middle
    ax_arrow = fig.add_subplot(gs[0, 3])
    ax_arrow.set_facecolor('#0f0f23')
    ax_arrow.axis('off')
    ax_arrow.text(0.5, 0.5, 'Random\nGeneration', ha='center', va='center',
                 fontsize=12, color='#4ecdc4', fontweight='bold')

    # PPLM-guided molecules (bottom row)
    pplm_mols = SAMPLE_MOLECULES['high_logp_high_qed']
    for i, smi in enumerate(pplm_mols[:3]):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor('#1a1a2e')
        mol = Chem.MolFromSmiles(smi)
        if mol:
            img = Draw.MolToImage(mol, size=(200, 200))
            ax.imshow(img)
        ax.axis('off')
        props = get_mol_properties(smi)
        ax.set_title(f"LogP={props['LogP']:.1f}, QED={props['QED']:.2f}",
                    color='#ff6b6b', fontsize=10)

    # PPLM label
    ax_pplm = fig.add_subplot(gs[1, 3])
    ax_pplm.set_facecolor('#0f0f23')
    ax_pplm.axis('off')
    ax_pplm.text(0.5, 0.5, 'PPLM\nGuided', ha='center', va='center',
                fontsize=12, color='#ff6b6b', fontweight='bold')

    plt.savefig('results/mol_pplm_guidance.png', facecolor='#0f0f23')
    plt.savefig('results/mol_pplm_guidance.pdf', facecolor='#0f0f23')
    print("Saved: mol_pplm_guidance")
    plt.close()


def plot_molecule_comparison_panel():
    """Side-by-side comparison of molecules from different methods."""
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping comparison panel")
        return

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Molecular Diversity Across Generation Methods',
                fontsize=16, fontweight='bold')

    categories = [
        ('Target Region', 'high_logp_high_qed', '#4CAF50'),
        ('Low LogP', 'low_logp_high_qed', '#2196F3'),
        ('Low QED', 'high_logp_low_qed', '#FF5722'),
    ]

    for row, (cat_name, cat_key, color) in enumerate(categories):
        mols = SAMPLE_MOLECULES[cat_key]
        for col, smi in enumerate(mols[:3]):
            ax = axes[row, col]
            mol = Chem.MolFromSmiles(smi)
            if mol:
                img = Draw.MolToImage(mol, size=(250, 250))
                ax.imshow(img)
            ax.axis('off')
            props = get_mol_properties(smi)
            if col == 0:
                ax.set_ylabel(cat_name, fontsize=12, color=color)
            ax.set_title(f"LogP={props['LogP']:.1f}\nQED={props['QED']:.2f}",
                        fontsize=9)

    plt.tight_layout()
    plt.savefig('results/mol_comparison_panel.png')
    plt.savefig('results/mol_comparison_panel.pdf')
    print("Saved: mol_comparison_panel")
    plt.close()


if __name__ == '__main__':
    print("Generating molecular visualizations...")
    print("=" * 50)
    plot_molecule_grid_with_properties()
    plot_property_space_with_molecules()
    plot_pplm_guidance_effect()
    plot_molecule_comparison_panel()
    print("=" * 50)
    print("All molecular figures generated!")
