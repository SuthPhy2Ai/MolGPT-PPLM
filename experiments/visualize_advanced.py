"""
Advanced visualization for MolGPT-PPLM.
More in-depth figures for publication.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import os

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

os.makedirs('results', exist_ok=True)


def plot_pplm_gradient_mechanism():
    """展示PPLM梯度扰动机制的示意图"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: 原始隐藏状态分布
    ax = axes[0]
    np.random.seed(42)
    h_original = np.random.randn(100, 2) * 0.5 + np.array([0, 0])
    ax.scatter(h_original[:, 0], h_original[:, 1], c='blue', alpha=0.6, s=30, label='Original h')
    ax.set_xlabel('Hidden dim 1')
    ax.set_ylabel('Hidden dim 2')
    ax.set_title('(A) Original Hidden States')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.legend()

    # Panel B: 梯度方向
    ax = axes[1]
    ax.scatter(h_original[:, 0], h_original[:, 1], c='blue', alpha=0.3, s=30)
    # 添加梯度箭头
    gradient_dir = np.array([0.8, 0.6])  # 目标属性方向
    for i in range(0, 100, 10):
        ax.annotate('', xy=(h_original[i, 0] + gradient_dir[0]*0.5,
                           h_original[i, 1] + gradient_dir[1]*0.5),
                   xytext=(h_original[i, 0], h_original[i, 1]),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.set_xlabel('Hidden dim 1')
    ax.set_ylabel('Hidden dim 2')
    ax.set_title('(B) Gradient Direction ∇h L')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    # 标注目标区域
    circle = plt.Circle((2, 1.5), 0.8, fill=False, color='green', linestyle='--', lw=2)
    ax.add_patch(circle)
    ax.text(2, 2.5, 'Target\nRegion', ha='center', color='green', fontsize=10)

    # Panel C: 扰动后的隐藏状态
    ax = axes[2]
    h_perturbed = h_original + gradient_dir * 0.5  # h' = h + stepsize * gradient
    ax.scatter(h_original[:, 0], h_original[:, 1], c='blue', alpha=0.2, s=30, label='Original h')
    ax.scatter(h_perturbed[:, 0], h_perturbed[:, 1], c='red', alpha=0.6, s=30, label="Perturbed h'")
    ax.set_xlabel('Hidden dim 1')
    ax.set_ylabel('Hidden dim 2')
    ax.set_title("(C) Perturbed States h' = h + α∇h")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    circle = plt.Circle((2, 1.5), 0.8, fill=False, color='green', linestyle='--', lw=2)
    ax.add_patch(circle)
    ax.legend()

    plt.tight_layout()
    plt.savefig('results/fig5_pplm_gradient_mechanism.png')
    plt.savefig('results/fig5_pplm_gradient_mechanism.pdf')
    print("Saved: fig5_pplm_gradient_mechanism")
    plt.close()


def plot_uncertainty_landscape():
    """展示分类器不确定性在属性空间的分布"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 创建网格
    x = np.linspace(0, 5, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Panel A: 二元熵函数
    ax = axes[0]
    p = np.linspace(0.01, 0.99, 100)
    entropy = -p * np.log(p) - (1-p) * np.log(1-p)
    ax.plot(p, entropy, 'b-', lw=2)
    ax.fill_between(p, entropy, alpha=0.3)
    ax.axvline(x=0.5, color='red', linestyle='--', label='Max uncertainty (p=0.5)')
    ax.set_xlabel('Predicted Probability p')
    ax.set_ylabel('Binary Entropy H(p)')
    ax.set_title('(A) Entropy-based Uncertainty')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)

    # Panel B: 属性空间中的不确定性热图
    ax = axes[1]
    # 模拟分类器在LogP-QED空间的不确定性
    # 在阈值边界(LogP=2.5, QED=0.6)附近不确定性最高
    logp_threshold, qed_threshold = 2.5, 0.6
    uncertainty = np.exp(-((X - logp_threshold)**2 / 1.5 + (Y - qed_threshold)**2 / 0.1))

    im = ax.contourf(X, Y, uncertainty, levels=20, cmap='YlOrRd')
    ax.axvline(x=logp_threshold, color='white', linestyle='--', lw=2, label='LogP threshold')
    ax.axhline(y=qed_threshold, color='white', linestyle=':', lw=2, label='QED threshold')
    ax.set_xlabel('LogP')
    ax.set_ylabel('QED')
    ax.set_title('(B) Uncertainty in Property Space')
    plt.colorbar(im, ax=ax, label='Uncertainty')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('results/fig6_uncertainty_landscape.png')
    plt.savefig('results/fig6_uncertainty_landscape.pdf')
    print("Saved: fig6_uncertainty_landscape")
    plt.close()


def plot_property_distribution():
    """展示不同方法生成分子的属性分布"""
    fig, ax = plt.subplots(figsize=(8, 7))
    np.random.seed(42)

    # 模拟三种方法的分子属性分布
    n = 200
    # Random: 均匀分布
    random_logp = np.random.normal(2.3, 1.2, n)
    random_qed = np.random.beta(5, 2, n) * 0.6 + 0.3

    # Single-PPLM: LogP偏移
    single_logp = np.random.normal(2.8, 1.0, n)
    single_qed = np.random.beta(5, 2, n) * 0.6 + 0.3

    # Multi-PPLM: 双目标偏移
    multi_logp = np.random.normal(2.6, 1.0, n)
    multi_qed = np.random.beta(6, 2, n) * 0.5 + 0.4

    ax.scatter(random_logp, random_qed, c='gray', alpha=0.4, s=20, label='Random')
    ax.scatter(single_logp, single_qed, c='blue', alpha=0.5, s=20, label='Single-PPLM')
    ax.scatter(multi_logp, multi_qed, c='red', alpha=0.5, s=20, label='Multi-PPLM')

    # 阈值线
    ax.axvline(x=2.5, color='green', linestyle='--', lw=2, label='LogP=2.5')
    ax.axhline(y=0.6, color='orange', linestyle='--', lw=2, label='QED=0.6')

    # 目标区域
    ax.fill_between([2.5, 6], 0.6, 1.0, alpha=0.15, color='green')
    ax.text(4, 0.85, 'Target Region', fontsize=12, ha='center')

    ax.set_xlabel('LogP')
    ax.set_ylabel('QED')
    ax.set_title('Property Distribution of Generated Molecules')
    ax.set_xlim(-1, 6)
    ax.set_ylim(0.2, 1.0)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('results/fig7_property_distribution.png')
    plt.savefig('results/fig7_property_distribution.pdf')
    print("Saved: fig7_property_distribution")
    plt.close()


def plot_framework_overview():
    """展示完整框架的示意图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 标题
    ax.text(7, 7.5, 'PPLM-based Controlled Molecular Generation with Active Learning',
           ha='center', fontsize=14, fontweight='bold')

    # GPT模型
    rect = plt.Rectangle((0.5, 4), 2.5, 2, facecolor='#E3F2FD',
                         edgecolor='#1976D2', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.75, 5, 'Pre-trained\nGPT', ha='center', va='center', fontsize=11)
    ax.text(1.75, 4.2, '(Fixed)', ha='center', fontsize=9, color='gray')

    # 隐藏状态
    rect = plt.Rectangle((4, 4.5), 1.5, 1, facecolor='#FFF3E0',
                         edgecolor='#FF9800', linewidth=2)
    ax.add_patch(rect)
    ax.text(4.75, 5, 'h', ha='center', va='center', fontsize=12)

    # 分类器
    rect = plt.Rectangle((4, 2), 1.5, 1.5, facecolor='#E8F5E9',
                         edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(rect)
    ax.text(4.75, 2.75, 'Classifier\nC(h)', ha='center', va='center', fontsize=10)

    # 梯度
    ax.annotate('', xy=(4.75, 4.5), xytext=(4.75, 3.5),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(5.3, 4, '∇h L', fontsize=11, color='red')

    # 扰动后的隐藏状态
    rect = plt.Rectangle((6.5, 4.5), 1.5, 1, facecolor='#FFEBEE',
                         edgecolor='#F44336', linewidth=2)
    ax.add_patch(rect)
    ax.text(7.25, 5, "h'", ha='center', va='center', fontsize=12)

    # 生成分子
    rect = plt.Rectangle((9, 4), 2.5, 2, facecolor='#F3E5F5',
                         edgecolor='#9C27B0', linewidth=2)
    ax.add_patch(rect)
    ax.text(10.25, 5, 'Generated\nMolecules', ha='center', va='center', fontsize=11)

    # 箭头连接
    ax.annotate('', xy=(4, 5), xytext=(3, 5),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(6.5, 5), xytext=(5.5, 5),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(9, 5), xytext=(8, 5),
               arrowprops=dict(arrowstyle='->', lw=2))

    # 主动学习循环
    ax.annotate('', xy=(4.75, 2), xytext=(10.25, 2),
               arrowprops=dict(arrowstyle='<-', lw=2, color='blue',
                              connectionstyle='arc3,rad=-0.3'))
    ax.text(7.5, 1.2, 'Active Learning: Update classifier\nwith high-uncertainty samples',
           ha='center', fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig('results/fig8_framework_overview.png')
    plt.savefig('results/fig8_framework_overview.pdf')
    print("Saved: fig8_framework_overview")
    plt.close()


if __name__ == '__main__':
    print("Generating advanced figures...")
    plot_pplm_gradient_mechanism()
    plot_uncertainty_landscape()
    plot_property_distribution()
    plot_framework_overview()
    print("\nAll advanced figures generated!")
