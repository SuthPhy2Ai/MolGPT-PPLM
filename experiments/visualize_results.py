"""
Visualization script for MolGPT-PPLM experiment results.
Generates publication-ready figures.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set style for publication
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def plot_baseline_comparison():
    """Plot baseline comparison bar chart."""

    # Experiment 2 results
    methods = ['Random', 'Single-PPLM', 'Multi-PPLM']

    metrics = {
        'Validity': [100.0, 99.2, 99.6],
        'Uniqueness': [100.0, 100.0, 100.0],
        'Novelty': [99.8, 100.0, 100.0],
        'Diversity': [86.7, 87.3, 86.2],
        'LogP>2.5': [54.2, 57.3, 55.0],
        'QED>0.6': [83.4, 86.5, 86.5]
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']

    for idx, (metric, values) in enumerate(metrics.items()):
        ax = axes[idx]
        bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1)
        ax.set_ylabel('Percentage (%)')
        ax.set_title(metric)
        ax.set_ylim(0, 105)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/fig1_baseline_comparison.png')
    plt.savefig('results/fig1_baseline_comparison.pdf')
    print("Saved: fig1_baseline_comparison.png/pdf")
    plt.close()


def plot_property_improvement():
    """Plot property improvement comparison."""

    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['Random\n(Baseline)', 'Single-PPLM\n(LogP only)', 'Multi-PPLM\n(LogP+QED)']
    logp_values = [54.2, 57.3, 55.0]
    qed_values = [83.4, 86.5, 86.5]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, logp_values, width, label='LogP > 2.5',
                   color='#FF6B6B', edgecolor='black')
    bars2 = ax.bar(x + width/2, qed_values, width, label='QED > 0.6',
                   color='#4ECDC4', edgecolor='black')

    ax.set_ylabel('Percentage of molecules meeting threshold (%)')
    ax.set_title('Property Optimization: PPLM vs Random Generation')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    # Add improvement annotations
    ax.annotate('', xy=(1, 57.3), xytext=(0, 54.2),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(0.5, 58, '+3.1%', ha='center', color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/fig2_property_improvement.png')
    plt.savefig('results/fig2_property_improvement.pdf')
    print("Saved: fig2_property_improvement.png/pdf")
    plt.close()


def plot_method_overview():
    """Plot method overview diagram."""

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Boxes
    boxes = [
        (1, 3, 2, 1.5, 'GPT\nGenerator', '#E8F4FD'),
        (4, 3, 2, 1.5, 'PPLM\nGuidance', '#FFE4E1'),
        (7, 3, 2, 1.5, 'Property\nClassifier', '#E8F8E8'),
        (10, 3, 2, 1.5, 'Generated\nMolecules', '#FFF8DC'),
    ]

    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color,
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=11, fontweight='bold')

    # Arrows
    arrows = [(3, 3.75), (6, 3.75), (9, 3.75)]
    for x, y in arrows:
        ax.annotate('', xy=(x+0.8, y), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', lw=2))

    # Feedback loop
    ax.annotate('', xy=(7.5, 2.8), xytext=(7.5, 1.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.annotate('', xy=(5, 1.5), xytext=(7.5, 1.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.annotate('', xy=(5, 2.8), xytext=(5, 1.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax.text(6.25, 1.2, 'Gradient Feedback\n(Hidden State Perturbation)',
           ha='center', fontsize=10, color='red')

    ax.set_title('PPLM-based Controlled Molecular Generation', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig('results/fig3_method_overview.png')
    plt.savefig('results/fig3_method_overview.pdf')
    print("Saved: fig3_method_overview.png/pdf")
    plt.close()


def plot_active_learning_flow():
    """Plot active learning workflow."""

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Active Learning Loop with Entropy-based Acquisition',
           ha='center', fontsize=14, fontweight='bold')

    # Steps
    steps = [
        (5, 8, '1. Generate molecules\nwith PPLM'),
        (5, 6.5, '2. Evaluate properties\n(Oracle)'),
        (5, 5, '3. Compute uncertainty\nH = -p log p'),
        (5, 3.5, '4. Select high-uncertainty\nsamples'),
        (5, 2, '5. Update classifier'),
    ]

    colors = ['#E8F4FD', '#FFE4E1', '#E8F8E8', '#FFF8DC', '#F0E6FF']

    for i, (x, y, text) in enumerate(steps):
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((x-1.8, y-0.5), 3.6, 1, facecolor=colors[i],
                             edgecolor='black', linewidth=1.5,
                             boxstyle='round,pad=0.05')
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11)

    # Arrows between steps
    for i in range(len(steps)-1):
        ax.annotate('', xy=(5, steps[i+1][1]+0.5), xytext=(5, steps[i][1]-0.5),
                   arrowprops=dict(arrowstyle='->', lw=2))

    # Loop back arrow
    ax.annotate('', xy=(7.5, 8), xytext=(7.5, 2),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue',
                              connectionstyle='arc3,rad=0.3'))
    ax.text(8.5, 5, 'Iterate', fontsize=11, color='blue', rotation=90, va='center')

    plt.tight_layout()
    plt.savefig('results/fig4_active_learning_flow.png')
    plt.savefig('results/fig4_active_learning_flow.pdf')
    print("Saved: fig4_active_learning_flow.png/pdf")
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)

    print("Generating figures...")
    plot_baseline_comparison()
    plot_property_improvement()
    plot_method_overview()
    plot_active_learning_flow()
    print("\nAll figures generated successfully!")
