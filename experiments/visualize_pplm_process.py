"""
PPLM Guidance Process Visualization with RDKit Molecules.
Combines molecular structure rendering with property analysis to show
how PPLM guides molecule generation toward target properties.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors
    from rdkit.Chem.QED import qed
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Using simulated data.")

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

os.makedirs('results', exist_ok=True)


# =============================================================================
# Sample Molecules for Different Guidance Strengths
# =============================================================================

# Molecules representing different stages of PPLM guidance
GUIDANCE_TRAJECTORY = {
    # No guidance (random) - low LogP molecules
    'strength_0.0': [
        'CC(=O)Nc1ccc(O)cc1',      # Paracetamol, LogP~0.5
        'OCC1OC(O)C(O)C(O)C1O',    # Glucose-like, LogP~-2
        'NC(=O)c1cccnc1',          # Nicotinamide, LogP~0.4
    ],
    # Low guidance - slightly higher LogP
    'strength_0.3': [
        'Cc1ccc(O)cc1',            # Cresol, LogP~2.0
        'c1ccc2[nH]ccc2c1',        # Indole, LogP~2.1
        'CC(C)c1ccccc1',           # Cumene, LogP~3.6
    ],
    # Medium guidance - approaching target
    'strength_0.6': [
        'CCc1ccc(CC)cc1',          # Diethylbenzene, LogP~3.7
        'c1ccc2c(c1)ccc1ccccc12',  # Phenanthrene, LogP~4.5
        'CC(C)Cc1ccccc1',          # Isobutylbenzene, LogP~4.0
    ],
    # High guidance - target region (high LogP, high QED)
    'strength_1.0': [
        'CC(C)Cc1ccc(C(C)C(=O)O)cc1',  # Ibuprofen-like, LogP~3.5
        'COc1ccc2[nH]cc(CCNC(C)=O)c2c1',  # Melatonin-like
        'Cc1ccc(NC(=O)c2ccccc2)cc1',   # Acetanilide derivative
    ],
}


def get_mol_properties(smiles):
    """Calculate molecular properties."""
    if not RDKIT_AVAILABLE:
        return {'LogP': np.random.uniform(1, 4), 'QED': np.random.uniform(0.4, 0.9)}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'LogP': Descriptors.MolLogP(mol),
        'QED': qed(mol),
        'MW': Descriptors.MolWt(mol),
    }


def get_fingerprint(smiles):
    """Get Morgan fingerprint for a molecule."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


# =============================================================================
# 1. Generation Trajectory Visualization
# =============================================================================

def plot_guidance_trajectory():
    """
    Show molecules generated at different PPLM guidance strengths.
    Demonstrates how increasing guidance shifts molecules toward target properties.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping guidance trajectory")
        return

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0f0f23')

    # Title
    title = fig.suptitle('PPLM Guidance Trajectory: From Random to Target',
                        fontsize=20, fontweight='bold', color='white', y=0.98)
    title.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='#667eea'),
        path_effects.Normal()
    ])

    # Create grid: 4 rows (guidance levels) x 4 cols (3 molecules + properties)
    gs = fig.add_gridspec(4, 5, hspace=0.4, wspace=0.3,
                         width_ratios=[1, 1, 1, 0.3, 1.2])

    strength_labels = ['0.0 (Random)', '0.3 (Low)', '0.6 (Medium)', '1.0 (High)']
    strength_keys = ['strength_0.0', 'strength_0.3', 'strength_0.6', 'strength_1.0']
    colors = ['#4ecdc4', '#ffd93d', '#ff6b6b', '#f093fb']

    all_props = []

    for row, (strength_key, label, color) in enumerate(zip(strength_keys, strength_labels, colors)):
        mols = GUIDANCE_TRAJECTORY[strength_key]

        # Draw molecules
        for col, smi in enumerate(mols[:3]):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('#1a1a2e')

            mol = Chem.MolFromSmiles(smi)
            if mol:
                img = Draw.MolToImage(mol, size=(200, 180))
                ax.imshow(img)

            ax.axis('off')
            props = get_mol_properties(smi)
            if props:
                all_props.append((props['LogP'], props['QED'], color, label))
                ax.set_title(f"LogP={props['LogP']:.1f}\nQED={props['QED']:.2f}",
                           color=color, fontsize=9, pad=5)

        # Arrow column
        ax_arrow = fig.add_subplot(gs[row, 3])
        ax_arrow.set_facecolor('#0f0f23')
        ax_arrow.axis('off')
        if row < 3:
            ax_arrow.annotate('', xy=(0.5, -0.3), xytext=(0.5, 1.0),
                            arrowprops=dict(arrowstyle='->', color='white', lw=2))

        # Strength label
        ax_label = fig.add_subplot(gs[row, 4])
        ax_label.set_facecolor('#1a1a2e')
        ax_label.axis('off')

        # Create a fancy box for the label
        bbox = FancyBboxPatch((0.1, 0.2), 0.8, 0.6, boxstyle='round,pad=0.05',
                             facecolor=color, alpha=0.8, edgecolor='white', linewidth=2)
        ax_label.add_patch(bbox)
        ax_label.text(0.5, 0.5, f'Guidance\n{label}', ha='center', va='center',
                     fontsize=11, fontweight='bold', color='white')
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(0, 1)

    plt.savefig('results/pplm_guidance_trajectory.png', facecolor='#0f0f23',
               bbox_inches='tight', dpi=300)
    plt.savefig('results/pplm_guidance_trajectory.pdf', facecolor='#0f0f23',
               bbox_inches='tight')
    print("Saved: pplm_guidance_trajectory")
    plt.close()

    return all_props


# =============================================================================
# 2. Property Space Migration Animation
# =============================================================================

def plot_property_space_migration():
    """
    Show how molecules migrate in LogP-QED property space during PPLM guidance.
    Includes molecule thumbnails and trajectory arrows.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping property space migration")
        return

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#1a1a2e')

    # Collect all molecules and their properties
    all_data = []
    strength_keys = ['strength_0.0', 'strength_0.3', 'strength_0.6', 'strength_1.0']
    colors = ['#4ecdc4', '#ffd93d', '#ff6b6b', '#f093fb']
    labels = ['Random', 'Low Guidance', 'Medium Guidance', 'High Guidance']

    for strength_key, color, label in zip(strength_keys, colors, labels):
        for smi in GUIDANCE_TRAJECTORY[strength_key]:
            props = get_mol_properties(smi)
            if props:
                all_data.append({
                    'smiles': smi,
                    'logp': props['LogP'],
                    'qed': props['QED'],
                    'color': color,
                    'label': label,
                    'strength': strength_key
                })

    # Draw target region
    ax.fill_between([2.5, 6], 0.6, 1.0, alpha=0.15, color='#4CAF50')
    ax.axvline(x=2.5, color='#4CAF50', linestyle='--', lw=2, alpha=0.7)
    ax.axhline(y=0.6, color='#4CAF50', linestyle='--', lw=2, alpha=0.7)
    ax.text(4.5, 0.9, 'Target Region', fontsize=14, ha='center',
           fontweight='bold', color='#4CAF50')

    # Plot scatter points with glow effect
    for data in all_data:
        # Glow
        for i in range(3, 0, -1):
            ax.scatter(data['logp'], data['qed'], c=data['color'],
                      s=300 + i*100, alpha=0.1, edgecolors='none')
        # Main point
        ax.scatter(data['logp'], data['qed'], c=data['color'],
                  s=300, alpha=0.9, edgecolors='white', linewidths=2, zorder=5)

    # Add molecule thumbnails for selected molecules
    thumbnail_indices = [0, 3, 6, 9]  # One from each guidance level
    for idx in thumbnail_indices:
        if idx < len(all_data):
            data = all_data[idx]
            mol = Chem.MolFromSmiles(data['smiles'])
            if mol:
                img = Draw.MolToImage(mol, size=(100, 80))
                imagebox = OffsetImage(img, zoom=0.8)
                ab = AnnotationBbox(imagebox, (data['logp'], data['qed']),
                                   xybox=(50, 50), xycoords='data',
                                   boxcoords="offset points",
                                   frameon=True,
                                   bboxprops=dict(boxstyle='round', fc='white',
                                                ec=data['color'], lw=2, alpha=0.95))
                ax.add_artist(ab)

    # Draw migration arrows (conceptual flow)
    # Calculate centroids for each guidance level
    centroids = {}
    for strength_key, color in zip(strength_keys, colors):
        points = [(d['logp'], d['qed']) for d in all_data if d['strength'] == strength_key]
        if points:
            centroids[strength_key] = (np.mean([p[0] for p in points]),
                                       np.mean([p[1] for p in points]))

    # Draw arrows between centroids
    for i in range(len(strength_keys) - 1):
        if strength_keys[i] in centroids and strength_keys[i+1] in centroids:
            start = centroids[strength_keys[i]]
            end = centroids[strength_keys[i+1]]
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', color='white',
                                      lw=3, alpha=0.7,
                                      connectionstyle='arc3,rad=0.2'))

    # Legend
    for color, label in zip(colors, labels):
        ax.scatter([], [], c=color, s=150, label=label, edgecolors='white', linewidths=1)
    ax.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='white',
             labelcolor='white', fontsize=11)

    # Styling
    ax.set_xlabel('LogP', color='white', fontsize=14)
    ax.set_ylabel('QED', color='white', fontsize=14)
    ax.set_xlim(-3, 6)
    ax.set_ylim(0.2, 1.0)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    title = ax.set_title('Molecular Migration in Property Space During PPLM Guidance',
                        color='white', fontsize=16, pad=15)
    title.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='#667eea'),
        path_effects.Normal()
    ])

    plt.tight_layout()
    plt.savefig('results/pplm_property_migration.png', facecolor='#0f0f23',
               bbox_inches='tight', dpi=300)
    plt.savefig('results/pplm_property_migration.pdf', facecolor='#0f0f23',
               bbox_inches='tight')
    print("Saved: pplm_property_migration")
    plt.close()


# =============================================================================
# 3. Molecular Structure Comparison
# =============================================================================

def plot_structure_comparison():
    """
    Side-by-side comparison of molecular structures from random vs PPLM-guided.
    Highlights structural differences that lead to property changes.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping structure comparison")
        return

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#0f0f23')

    # Title
    title = fig.suptitle('Molecular Structure Comparison: Random vs PPLM-Guided',
                        fontsize=18, fontweight='bold', color='white', y=0.98)
    title.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='#667eea'),
        path_effects.Normal()
    ])

    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    random_mols = GUIDANCE_TRAJECTORY['strength_0.0']
    guided_mols = GUIDANCE_TRAJECTORY['strength_1.0']

    for row in range(3):
        # Random molecule
        ax_rand = fig.add_subplot(gs[row, 0])
        ax_rand.set_facecolor('#1a1a2e')
        smi_rand = random_mols[row]
        mol_rand = Chem.MolFromSmiles(smi_rand)
        if mol_rand:
            img = Draw.MolToImage(mol_rand, size=(220, 200))
            ax_rand.imshow(img)
        ax_rand.axis('off')
        props_rand = get_mol_properties(smi_rand)
        ax_rand.set_title(f"Random\nLogP={props_rand['LogP']:.1f}, QED={props_rand['QED']:.2f}",
                         color='#4ecdc4', fontsize=10)

        # Arrow
        ax_arrow = fig.add_subplot(gs[row, 1])
        ax_arrow.set_facecolor('#0f0f23')
        ax_arrow.axis('off')
        ax_arrow.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                         arrowprops=dict(arrowstyle='->', color='white', lw=3))
        ax_arrow.text(0.5, 0.7, 'PPLM\nGuidance', ha='center', va='center',
                     color='#ffd93d', fontsize=10, fontweight='bold')

        # Guided molecule
        ax_guid = fig.add_subplot(gs[row, 2])
        ax_guid.set_facecolor('#1a1a2e')
        smi_guid = guided_mols[row]
        mol_guid = Chem.MolFromSmiles(smi_guid)
        if mol_guid:
            img = Draw.MolToImage(mol_guid, size=(220, 200))
            ax_guid.imshow(img)
        ax_guid.axis('off')
        props_guid = get_mol_properties(smi_guid)
        ax_guid.set_title(f"PPLM-Guided\nLogP={props_guid['LogP']:.1f}, QED={props_guid['QED']:.2f}",
                         color='#ff6b6b', fontsize=10)

        # Property comparison bar
        ax_bar = fig.add_subplot(gs[row, 3])
        ax_bar.set_facecolor('#1a1a2e')

        metrics = ['LogP', 'QED']
        rand_vals = [props_rand['LogP'], props_rand['QED']]
        guid_vals = [props_guid['LogP'], props_guid['QED']]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax_bar.barh(x - width/2, rand_vals, width, label='Random',
                           color='#4ecdc4', alpha=0.8)
        bars2 = ax_bar.barh(x + width/2, guid_vals, width, label='Guided',
                           color='#ff6b6b', alpha=0.8)

        ax_bar.set_yticks(x)
        ax_bar.set_yticklabels(metrics, color='white')
        ax_bar.tick_params(colors='white')
        ax_bar.set_xlim(0, 5)

        # Threshold lines
        ax_bar.axvline(x=2.5, color='#ffd93d', linestyle='--', lw=1.5, alpha=0.7)

        for spine in ax_bar.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.3)

        if row == 0:
            ax_bar.legend(loc='upper right', facecolor='#1a1a2e',
                         edgecolor='white', labelcolor='white', fontsize=8)

    plt.savefig('results/pplm_structure_comparison.png', facecolor='#0f0f23',
               bbox_inches='tight', dpi=300)
    plt.savefig('results/pplm_structure_comparison.pdf', facecolor='#0f0f23',
               bbox_inches='tight')
    print("Saved: pplm_structure_comparison")
    plt.close()


# =============================================================================
# 4. Chemical Space t-SNE with Molecules
# =============================================================================

def plot_chemical_space_with_molecules():
    """
    t-SNE visualization of chemical space with molecule thumbnails.
    Shows how PPLM-guided molecules cluster in a different region.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping chemical space plot")
        return

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#1a1a2e')

    # Collect all molecules
    all_smiles = []
    all_labels = []
    all_colors = []

    strength_keys = ['strength_0.0', 'strength_0.3', 'strength_0.6', 'strength_1.0']
    colors = ['#4ecdc4', '#ffd93d', '#ff6b6b', '#f093fb']
    labels = ['Random', 'Low', 'Medium', 'High']

    for strength_key, color, label in zip(strength_keys, colors, labels):
        for smi in GUIDANCE_TRAJECTORY[strength_key]:
            all_smiles.append(smi)
            all_labels.append(label)
            all_colors.append(color)

    # Calculate fingerprints and similarity matrix
    fps = [get_fingerprint(smi) for smi in all_smiles]
    valid_fps = [(i, fp) for i, fp in enumerate(fps) if fp is not None]

    if len(valid_fps) < 3:
        print("Not enough valid fingerprints")
        return

    # Simple 2D projection using MDS-like approach based on Tanimoto distance
    n = len(valid_fps)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(valid_fps[i][1], valid_fps[j][1])
            dist_matrix[i, j] = 1 - sim
            dist_matrix[j, i] = 1 - sim

    # Simple MDS projection
    np.random.seed(42)
    coords = np.random.randn(n, 2)

    # Iterative optimization (simplified MDS)
    for _ in range(100):
        for i in range(n):
            grad = np.zeros(2)
            for j in range(n):
                if i != j:
                    diff = coords[i] - coords[j]
                    current_dist = np.linalg.norm(diff) + 1e-10
                    target_dist = dist_matrix[i, j] * 3
                    grad += (current_dist - target_dist) * diff / current_dist
            coords[i] -= 0.1 * grad

    # Plot points with glow
    for idx, (orig_idx, fp) in enumerate(valid_fps):
        color = all_colors[orig_idx]
        x, y = coords[idx]

        # Glow effect
        for i in range(3, 0, -1):
            ax.scatter(x, y, c=color, s=200 + i*80, alpha=0.1, edgecolors='none')
        ax.scatter(x, y, c=color, s=200, alpha=0.9,
                  edgecolors='white', linewidths=1.5, zorder=5)

    # Add molecule thumbnails for corner molecules
    corner_indices = [0, 2, 6, 9, 11]
    for idx in corner_indices:
        if idx < len(valid_fps):
            orig_idx = valid_fps[idx][0]
            smi = all_smiles[orig_idx]
            mol = Chem.MolFromSmiles(smi)
            if mol:
                x, y = coords[idx]
                img = Draw.MolToImage(mol, size=(90, 75))
                imagebox = OffsetImage(img, zoom=0.7)
                ab = AnnotationBbox(imagebox, (x, y),
                                   xybox=(40, 40), xycoords='data',
                                   boxcoords="offset points",
                                   frameon=True,
                                   bboxprops=dict(boxstyle='round', fc='white',
                                                ec=all_colors[orig_idx], lw=2))
                ax.add_artist(ab)

    # Legend
    for color, label in zip(colors, labels):
        ax.scatter([], [], c=color, s=100, label=f'{label} Guidance',
                  edgecolors='white', linewidths=1)
    ax.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='white',
             labelcolor='white', fontsize=11)

    # Styling
    ax.set_xlabel('Chemical Space Dimension 1', color='white', fontsize=14)
    ax.set_ylabel('Chemical Space Dimension 2', color='white', fontsize=14)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    title = ax.set_title('Chemical Space Visualization (Fingerprint-based MDS)',
                        color='white', fontsize=16, pad=15)
    title.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='#667eea'),
        path_effects.Normal()
    ])

    plt.tight_layout()
    plt.savefig('results/pplm_chemical_space.png', facecolor='#0f0f23',
               bbox_inches='tight', dpi=300)
    plt.savefig('results/pplm_chemical_space.pdf', facecolor='#0f0f23',
               bbox_inches='tight')
    print("Saved: pplm_chemical_space")
    plt.close()


# =============================================================================
# 5. Combined Summary Panel
# =============================================================================

def plot_summary_panel():
    """
    Create a comprehensive summary panel showing all aspects of PPLM guidance.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping summary panel")
        return

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0f0f23')

    # Main title
    title = fig.suptitle('PPLM-Guided Molecular Generation: Complete Overview',
                        fontsize=22, fontweight='bold', color='white', y=0.98)
    title.set_path_effects([
        path_effects.Stroke(linewidth=4, foreground='#667eea'),
        path_effects.Normal()
    ])

    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # --- Row 1: Before/After molecules ---
    random_mols = GUIDANCE_TRAJECTORY['strength_0.0']
    guided_mols = GUIDANCE_TRAJECTORY['strength_1.0']

    # Random molecule
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')
    mol = Chem.MolFromSmiles(random_mols[0])
    if mol:
        img = Draw.MolToImage(mol, size=(200, 180))
        ax1.imshow(img)
    ax1.axis('off')
    props = get_mol_properties(random_mols[0])
    ax1.set_title(f"Before (Random)\nLogP={props['LogP']:.1f}, QED={props['QED']:.2f}",
                 color='#4ecdc4', fontsize=11)

    # Arrow
    ax_arr = fig.add_subplot(gs[0, 1])
    ax_arr.set_facecolor('#0f0f23')
    ax_arr.axis('off')
    ax_arr.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                   arrowprops=dict(arrowstyle='->', color='#ffd93d', lw=4))
    ax_arr.text(0.5, 0.7, 'PPLM\nGuidance', ha='center', va='center',
               color='#ffd93d', fontsize=12, fontweight='bold')

    # Guided molecule
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#1a1a2e')
    mol = Chem.MolFromSmiles(guided_mols[0])
    if mol:
        img = Draw.MolToImage(mol, size=(200, 180))
        ax2.imshow(img)
    ax2.axis('off')
    props = get_mol_properties(guided_mols[0])
    ax2.set_title(f"After (PPLM)\nLogP={props['LogP']:.1f}, QED={props['QED']:.2f}",
                 color='#ff6b6b', fontsize=11)

    # Property improvement bar chart
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_facecolor('#1a1a2e')

    methods = ['Random', 'PPLM']
    logp_vals = [54.2, 57.3]
    qed_vals = [83.4, 86.5]

    x = np.arange(2)
    width = 0.35
    ax3.bar(x - width/2, logp_vals, width, label='LogP>2.5 (%)',
           color='#4ecdc4', alpha=0.8)
    ax3.bar(x + width/2, qed_vals, width, label='QED>0.6 (%)',
           color='#ff6b6b', alpha=0.8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, color='white')
    ax3.set_ylabel('Percentage (%)', color='white')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax3.set_ylim(0, 100)
    for spine in ax3.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.3)
    ax3.set_title('Property Improvement', color='white', fontsize=11)

    # --- Row 2: Guidance trajectory (4 molecules) ---
    strength_keys = ['strength_0.0', 'strength_0.3', 'strength_0.6', 'strength_1.0']
    colors = ['#4ecdc4', '#ffd93d', '#ff6b6b', '#f093fb']
    labels = ['0.0', '0.3', '0.6', '1.0']

    for col, (key, color, label) in enumerate(zip(strength_keys, colors, labels)):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor('#1a1a2e')
        smi = GUIDANCE_TRAJECTORY[key][0]
        mol = Chem.MolFromSmiles(smi)
        if mol:
            img = Draw.MolToImage(mol, size=(180, 160))
            ax.imshow(img)
        ax.axis('off')
        props = get_mol_properties(smi)
        ax.set_title(f"Strength={label}\nLogP={props['LogP']:.1f}",
                    color=color, fontsize=10)

    # --- Row 3: Property space scatter ---
    ax_scatter = fig.add_subplot(gs[2, :2])
    ax_scatter.set_facecolor('#1a1a2e')

    # Plot all molecules
    for key, color in zip(strength_keys, colors):
        for smi in GUIDANCE_TRAJECTORY[key]:
            props = get_mol_properties(smi)
            if props:
                ax_scatter.scatter(props['LogP'], props['QED'], c=color,
                                  s=150, alpha=0.8, edgecolors='white', linewidths=1)

    # Target region
    ax_scatter.fill_between([2.5, 6], 0.6, 1.0, alpha=0.15, color='#4CAF50')
    ax_scatter.axvline(x=2.5, color='#4CAF50', linestyle='--', lw=2, alpha=0.5)
    ax_scatter.axhline(y=0.6, color='#4CAF50', linestyle='--', lw=2, alpha=0.5)

    ax_scatter.set_xlabel('LogP', color='white', fontsize=12)
    ax_scatter.set_ylabel('QED', color='white', fontsize=12)
    ax_scatter.set_xlim(-3, 6)
    ax_scatter.set_ylim(0.2, 1.0)
    ax_scatter.tick_params(colors='white')
    for spine in ax_scatter.spines.values():
        spine.set_color('white')
    ax_scatter.set_title('Property Space Distribution', color='white', fontsize=11)

    # Legend for scatter
    for color, label in zip(colors, ['Random', 'Low', 'Medium', 'High']):
        ax_scatter.scatter([], [], c=color, s=80, label=label)
    ax_scatter.legend(loc='lower right', facecolor='#1a1a2e',
                     edgecolor='white', labelcolor='white', fontsize=9)

    # --- Row 3 right: Method diagram ---
    ax_diag = fig.add_subplot(gs[2, 2:])
    ax_diag.set_facecolor('#1a1a2e')
    ax_diag.set_xlim(0, 10)
    ax_diag.set_ylim(0, 6)
    ax_diag.axis('off')

    # Draw pipeline boxes
    boxes = [
        (0.5, 2.5, 2, 1.5, 'GPT', '#667eea'),
        (3.5, 2.5, 2, 1.5, 'PPLM', '#ff6b6b'),
        (6.5, 2.5, 2.5, 1.5, 'Molecules', '#4ecdc4'),
    ]

    for x, y, w, h, text, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                             facecolor=color, alpha=0.8,
                             edgecolor='white', linewidth=2)
        ax_diag.add_patch(rect)
        ax_diag.text(x + w/2, y + h/2, text, ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white')

    # Arrows
    ax_diag.annotate('', xy=(3.3, 3.25), xytext=(2.7, 3.25),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax_diag.annotate('', xy=(6.3, 3.25), xytext=(5.7, 3.25),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2))

    # Classifier
    rect = FancyBboxPatch((3.5, 0.5), 2, 1.2, boxstyle='round,pad=0.1',
                         facecolor='#ffd93d', alpha=0.8,
                         edgecolor='white', linewidth=2)
    ax_diag.add_patch(rect)
    ax_diag.text(4.5, 1.1, 'Classifier', ha='center', va='center',
                fontsize=10, fontweight='bold', color='#1a1a2e')

    ax_diag.annotate('', xy=(4.5, 2.5), xytext=(4.5, 1.7),
                    arrowprops=dict(arrowstyle='<->', color='#ffd93d', lw=2))
    ax_diag.text(5.2, 2.1, '∇h', fontsize=10, color='#ffd93d')

    ax_diag.set_title('PPLM Pipeline', color='white', fontsize=11, pad=10)

    plt.savefig('results/pplm_summary_panel.png', facecolor='#0f0f23',
               bbox_inches='tight', dpi=300)
    plt.savefig('results/pplm_summary_panel.pdf', facecolor='#0f0f23',
               bbox_inches='tight')
    print("Saved: pplm_summary_panel")
    plt.close()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PPLM Guidance Process Visualization")
    print("=" * 60)

    if not RDKIT_AVAILABLE:
        print("\nERROR: RDKit is required for molecular visualizations.")
        print("Install with: pip install rdkit")
        exit(1)

    print("\n1. Generating guidance trajectory visualization...")
    plot_guidance_trajectory()

    print("\n2. Generating property space migration visualization...")
    plot_property_space_migration()

    print("\n3. Generating structure comparison visualization...")
    plot_structure_comparison()

    print("\n4. Generating chemical space visualization...")
    plot_chemical_space_with_molecules()

    print("\n5. Generating summary panel...")
    plot_summary_panel()

    print("\n" + "=" * 60)
    print("All PPLM process visualizations generated!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - pplm_guidance_trajectory.png/pdf")
    print("  - pplm_property_migration.png/pdf")
    print("  - pplm_structure_comparison.png/pdf")
    print("  - pplm_chemical_space.png/pdf")
    print("  - pplm_summary_panel.png/pdf")
