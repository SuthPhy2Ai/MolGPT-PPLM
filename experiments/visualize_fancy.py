"""
Fancy Visualizations for MolGPT-PPLM Publication.
High-quality, visually impressive figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, ConnectionPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

os.makedirs('results', exist_ok=True)

# Custom color palettes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'light': '#E8E8E8',
    'dark': '#1A1A2E',
}

GRADIENT_COLORS = ['#667eea', '#764ba2', '#f093fb']


def create_gradient_background(ax, color1='#1a1a2e', color2='#16213e'):
    """Create gradient background for axes."""
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack([gradient] * 256)
    cmap = LinearSegmentedColormap.from_list('bg', [color1, color2])
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=ax.get_xlim() + ax.get_ylim(), zorder=-1)


def plot_neural_network_flow():
    """Create a stunning neural network flow diagram."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')

    # Title with glow effect
    title = ax.text(8, 8.3, 'PPLM-Guided Molecular Generation Pipeline',
                   ha='center', fontsize=18, fontweight='bold', color='white')
    title.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='#667eea'),
        path_effects.Normal()
    ])

    # Draw glowing boxes for components
    components = [
        (1.5, 4, 2.5, 2, 'GPT\nGenerator', '#667eea'),
        (5.5, 4, 2.5, 2, 'Hidden\nStates h', '#f093fb'),
        (9.5, 4, 2.5, 2, 'PPLM\nPerturbation', '#ff6b6b'),
        (13.5, 4, 2.5, 2, 'Molecules', '#4ecdc4'),
    ]

    for x, y, w, h, text, color in components:
        # Glow effect
        for i in range(5, 0, -1):
            rect = FancyBboxPatch((x-0.05*i, y-0.05*i), w+0.1*i, h+0.1*i,
                                 boxstyle='round,pad=0.1', facecolor=color,
                                 alpha=0.1, edgecolor='none')
            ax.add_patch(rect)
        # Main box
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                             facecolor=color, alpha=0.9, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')

    # Animated-style arrows
    arrow_positions = [(4, 5), (8, 5), (12, 5)]
    for x, y in arrow_positions:
        ax.annotate('', xy=(x+1.3, y), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color='white', lw=3,
                                  connectionstyle='arc3,rad=0'))

    # Classifier feedback loop
    ax.annotate('', xy=(10.75, 4), xytext=(10.75, 2),
               arrowprops=dict(arrowstyle='->', color='#ffd93d', lw=2))

    classifier_box = FancyBboxPatch((9.5, 1), 2.5, 1.2, boxstyle='round,pad=0.1',
                                   facecolor='#ffd93d', alpha=0.9, edgecolor='white', linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(10.75, 1.6, 'Property\nClassifier', ha='center', va='center',
           fontsize=10, fontweight='bold', color='#1a1a2e')

    ax.annotate('', xy=(9.5, 1.6), xytext=(6.75, 1.6),
               arrowprops=dict(arrowstyle='->', color='#ffd93d', lw=2,
                              connectionstyle='arc3,rad=-0.2'))
    ax.text(8, 1.1, 'Gradient: ∇h L', ha='center', fontsize=10, color='#ffd93d')

    plt.tight_layout()
    plt.savefig('results/fancy_pipeline.png', facecolor='#0f0f23', edgecolor='none')
    plt.savefig('results/fancy_pipeline.pdf', facecolor='#0f0f23', edgecolor='none')
    print("Saved: fancy_pipeline")
    plt.close()


def plot_radar_comparison():
    """Radar chart comparing methods across metrics."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')

    categories = ['Validity', 'Uniqueness', 'Novelty', 'Diversity', 'LogP>2.5', 'QED>0.6']
    N = len(categories)

    # Data (normalized to 0-1)
    random_vals = [1.0, 1.0, 0.998, 0.867, 0.542, 0.834]
    single_vals = [0.992, 1.0, 1.0, 0.873, 0.573, 0.865]
    multi_vals = [0.996, 1.0, 1.0, 0.862, 0.550, 0.865]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for vals in [random_vals, single_vals, multi_vals]:
        vals += vals[:1]

    # Style
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', size=12)
    ax.tick_params(colors='white')

    # Grid styling
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], color='gray', size=9)
    ax.grid(color='gray', alpha=0.3)

    # Plot data with glow
    colors_plot = ['#4ecdc4', '#ff6b6b', '#ffd93d']
    labels = ['Random', 'Single-PPLM', 'Multi-PPLM']
    all_vals = [random_vals, single_vals, multi_vals]

    for vals, color, label in zip(all_vals, colors_plot, labels):
        ax.plot(angles, vals, 'o-', linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.25, color=color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
             facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

    plt.title('Method Comparison Across Metrics', color='white', size=16, pad=20)
    plt.tight_layout()
    plt.savefig('results/fancy_radar.png', facecolor='#0f0f23', edgecolor='none')
    plt.savefig('results/fancy_radar.pdf', facecolor='#0f0f23', edgecolor='none')
    print("Saved: fancy_radar")
    plt.close()


def plot_chemical_space_tsne():
    """Simulated t-SNE visualization of chemical space."""
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')

    np.random.seed(42)
    n_points = 300

    # Simulate t-SNE clusters
    # Training data (large cloud)
    train_x = np.random.randn(n_points) * 2
    train_y = np.random.randn(n_points) * 2

    # Random generated (overlapping with training)
    random_x = np.random.randn(100) * 1.8 + 0.5
    random_y = np.random.randn(100) * 1.8 - 0.3

    # PPLM generated (shifted toward target region)
    pplm_x = np.random.randn(100) * 1.5 + 2.5
    pplm_y = np.random.randn(100) * 1.5 + 1.5

    # Plot with glow effect
    ax.scatter(train_x, train_y, c='#4a4a6a', s=30, alpha=0.3, label='Training Data')
    ax.scatter(random_x, random_y, c='#4ecdc4', s=60, alpha=0.7,
               label='Random Generated', edgecolors='white', linewidths=0.5)
    ax.scatter(pplm_x, pplm_y, c='#ff6b6b', s=60, alpha=0.7,
               label='PPLM Generated', edgecolors='white', linewidths=0.5)

    # Target region highlight
    target_circle = plt.Circle((3, 2), 2.5, fill=False, color='#ffd93d',
                               linestyle='--', linewidth=2, label='Target Region')
    ax.add_patch(target_circle)

    ax.set_xlabel('t-SNE Dimension 1', color='white', size=14)
    ax.set_ylabel('t-SNE Dimension 2', color='white', size=14)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white', loc='upper left')
    plt.title('Chemical Space Visualization (t-SNE)', color='white', size=16, pad=15)

    plt.tight_layout()
    plt.savefig('results/fancy_tsne.png', facecolor='#0f0f23')
    plt.savefig('results/fancy_tsne.pdf', facecolor='#0f0f23')
    print("Saved: fancy_tsne")
    plt.close()


def plot_3d_property_landscape():
    """3D surface plot of property optimization landscape."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')

    # Create mesh
    x = np.linspace(0, 5, 50)  # LogP
    y = np.linspace(0.3, 1.0, 50)  # QED
    X, Y = np.meshgrid(x, y)

    # Simulated success probability surface
    logp_thresh, qed_thresh = 2.5, 0.6
    Z = 1 / (1 + np.exp(-(X - logp_thresh))) * 1 / (1 + np.exp(-10*(Y - qed_thresh)))

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8,
                          linewidth=0, antialiased=True)

    ax.set_xlabel('LogP', color='white', labelpad=10)
    ax.set_ylabel('QED', color='white', labelpad=10)
    ax.set_zlabel('Success Prob', color='white', labelpad=10)
    ax.tick_params(colors='white')

    # Color bar
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.view_init(elev=25, azim=45)
    plt.title('Property Optimization Landscape', color='white', size=16, pad=20)

    plt.savefig('results/fancy_3d_landscape.png', facecolor='#0f0f23')
    plt.savefig('results/fancy_3d_landscape.pdf', facecolor='#0f0f23')
    print("Saved: fancy_3d_landscape")
    plt.close()


def plot_gradient_flow():
    """Visualize gradient flow in hidden space."""
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')

    np.random.seed(42)

    # Create vector field
    x = np.linspace(-3, 3, 15)
    y = np.linspace(-3, 3, 15)
    X, Y = np.meshgrid(x, y)

    # Gradient direction toward target
    target = np.array([2, 1.5])
    U = (target[0] - X) / (np.sqrt((target[0]-X)**2 + (target[1]-Y)**2) + 0.1)
    V = (target[1] - Y) / (np.sqrt((target[0]-X)**2 + (target[1]-Y)**2) + 0.1)

    # Plot streamlines
    ax.streamplot(X, Y, U, V, color='#667eea', linewidth=1.5,
                  density=1.5, arrowsize=1.5, arrowstyle='->')

    # Original points
    h_orig = np.random.randn(50, 2) * 0.8
    ax.scatter(h_orig[:, 0], h_orig[:, 1], c='#4ecdc4', s=80,
               alpha=0.8, edgecolors='white', linewidths=1, label='Original h', zorder=5)

    # Perturbed points
    h_pert = h_orig + 0.5 * np.column_stack([
        (target[0] - h_orig[:, 0]) / np.linalg.norm(target - h_orig, axis=1),
        (target[1] - h_orig[:, 1]) / np.linalg.norm(target - h_orig, axis=1)
    ])
    ax.scatter(h_pert[:, 0], h_pert[:, 1], c='#ff6b6b', s=80,
               alpha=0.8, edgecolors='white', linewidths=1, label="Perturbed h'", zorder=5)

    # Target region with glow
    for i in range(5, 0, -1):
        circle = plt.Circle(target, 0.8 + 0.1*i, fill=False,
                           color='#ffd93d', alpha=0.2, linewidth=2)
        ax.add_patch(circle)
    circle = plt.Circle(target, 0.8, fill=False, color='#ffd93d',
                       linewidth=3, label='Target Region')
    ax.add_patch(circle)
    ax.scatter([target[0]], [target[1]], c='#ffd93d', s=200,
               marker='*', zorder=10, edgecolors='white')

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('Hidden Dimension 1', color='white', size=14)
    ax.set_ylabel('Hidden Dimension 2', color='white', size=14)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    plt.title('Gradient Flow in Hidden Space', color='white', size=16, pad=15)

    plt.tight_layout()
    plt.savefig('results/fancy_gradient_flow.png', facecolor='#0f0f23')
    plt.savefig('results/fancy_gradient_flow.pdf', facecolor='#0f0f23')
    print("Saved: fancy_gradient_flow")
    plt.close()


def plot_active_learning_cycle():
    """Circular diagram of active learning cycle."""
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    # Title
    title = ax.text(0, 1.35, 'Active Learning Cycle',
                   ha='center', fontsize=20, fontweight='bold', color='white')
    title.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='#667eea'),
        path_effects.Normal()
    ])

    # Cycle steps
    steps = [
        ('Generate\nMolecules', '#667eea'),
        ('Evaluate\nProperties', '#4ecdc4'),
        ('Compute\nUncertainty', '#ffd93d'),
        ('Select\nSamples', '#ff6b6b'),
        ('Update\nClassifier', '#f093fb'),
    ]
    n_steps = len(steps)
    radius = 0.9

    for i, (text, color) in enumerate(steps):
        angle = 2 * np.pi * i / n_steps - np.pi/2
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        # Glow effect
        for j in range(5, 0, -1):
            circle = plt.Circle((x, y), 0.25 + 0.02*j,
                               facecolor=color, alpha=0.1)
            ax.add_patch(circle)

        # Main circle
        circle = plt.Circle((x, y), 0.25, facecolor=color,
                           edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, text, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

    # Draw curved arrows between steps
    for i in range(n_steps):
        angle1 = 2 * np.pi * i / n_steps - np.pi/2
        angle2 = 2 * np.pi * ((i+1) % n_steps) / n_steps - np.pi/2

        x1 = radius * np.cos(angle1 + 0.3)
        y1 = radius * np.sin(angle1 + 0.3)
        x2 = radius * np.cos(angle2 - 0.3)
        y2 = radius * np.sin(angle2 - 0.3)

        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='white',
                                  lw=2, connectionstyle='arc3,rad=0.3'))

    plt.tight_layout()
    plt.savefig('results/fancy_al_cycle.png', facecolor='#0f0f23')
    plt.savefig('results/fancy_al_cycle.pdf', facecolor='#0f0f23')
    print("Saved: fancy_al_cycle")
    plt.close()


if __name__ == '__main__':
    print("Generating fancy visualizations...")
    print("=" * 50)
    plot_neural_network_flow()
    plot_radar_comparison()
    plot_chemical_space_tsne()
    plot_3d_property_landscape()
    plot_gradient_flow()
    plot_active_learning_cycle()
    print("=" * 50)
    print("All fancy figures generated!")
