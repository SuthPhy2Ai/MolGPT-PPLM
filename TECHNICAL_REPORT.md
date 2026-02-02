# MolGPT + PPLM 分子生成技术报告

## 项目概述

本项目实现了基于GPT的分子生成模型，结合PPLM（Plug and Play Language Model）技术实现可控分子生成。通过多目标优化和主动学习，能够生成具有特定性质的有效分子。

---

## 1. 模型架构

### 1.1 GPT基础模型

```
GPTConfig:
  - vocab_size: 63 (完整SMILES词汇表)
  - block_size: 71 (最大序列长度)
  - n_layer: 8 (Transformer层数)
  - n_head: 8 (注意力头数)
  - n_embd: 256 (嵌入维度)
  - 总参数量: ~5.2M
```

**模型结构:**
```
GPT
├── tok_emb: Embedding(63, 256)      # Token嵌入
├── pos_emb: Parameter(1, 72, 256)   # 位置嵌入
├── type_emb: Embedding(4, 256)      # 类型嵌入
├── drop: Dropout(0.1)
├── blocks: 8 x TransformerBlock
│   ├── ln1: LayerNorm(256)
│   ├── attn: CausalSelfAttention
│   │   ├── key: Linear(256, 256)
│   │   ├── query: Linear(256, 256)
│   │   └── value: Linear(256, 256)
│   ├── ln2: LayerNorm(256)
│   └── mlp: MLP
│       ├── fc: Linear(256, 1024)
│       ├── gelu: GELU
│       └── proj: Linear(1024, 256)
├── ln_f: LayerNorm(256)
└── head: Linear(256, 63)
```

### 1.2 SMILES词汇表

**完整词汇表 (63 tokens):**

| 类别 | Tokens |
|------|--------|
| 原子 | C, N, O, S, P, F, I, B, c, n, o, s, p, b |
| 卤素 | Cl, Br |
| 立体化学 | [C@@H], [C@H], [C@@], [C@], [nH], [N+], [O-], [n+], [NH+], [S@], [S@@] |
| 键 | =, #, -, /, \ |
| 环 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| 结构 | (, ), [, ] |
| 特殊 | <(padding), ., @, +, : |

**词汇表文件:** `moses2_stoi_full.json`

---

## 2. 训练过程

### 2.1 数据集

- **数据集:** MOSES2 (Molecular Sets)
- **训练集:** 1,584,664 分子
- **测试集:** 176,075 分子
- **数据文件:** `datasets/moses2.csv`

### 2.2 训练配置

```python
# 训练超参数
epochs = 50
batch_size = 128
learning_rate = 3e-4
weight_decay = 0.01
scheduler = CosineAnnealingLR(T_max=50, eta_min=1e-5)
gradient_clip = 1.0
```

### 2.3 训练脚本

**文件:** `train/train_fullvocab.py`

```python
# 核心训练循环
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        _, loss, _ = model(x, y, None, None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
```

### 2.4 训练结果

| Epoch | Train Loss | Test Loss | Validity |
|-------|------------|-----------|----------|
| 5     | 0.3514     | 0.3400    | 92.0%    |
| 10    | 0.3225     | 0.3188    | 98.0%    |
| 15    | 0.3078     | 0.3102    | 99.0%    |
| 20    | 0.2959     | 0.3063    | 98.5%    |
| 25    | 0.2862     | 0.3044    | 99.0%    |
| 30    | 0.2780     | 0.3037    | 99.0%    |
| **35**| 0.2708     | 0.3034    | **100.0%** |
| 40    | 0.2665     | 0.3044    | 98.5%    |
| 45    | 0.2640     | 0.3053    | 99.0%    |
| 50    | 0.2626     | 0.3061    | 99.5%    |

**最佳模型:** Epoch 35, Validity = 100%

**模型权重:** `cond_gpt/weights/fullvocab_gpt.pt`

---

## 3. PPLM 可控生成

### 3.1 PPLM 原理

PPLM通过在生成过程中对隐藏状态进行梯度引导，实现属性可控的分子生成：

```
h' = h + stepsize * gm_scale * ∇_h L(classifier(h), target)
```

其中:
- `h`: 原始隐藏状态
- `h'`: 扰动后的隐藏状态
- `L`: 分类器损失函数
- `target`: 目标属性类别

### 3.2 多目标分类器

**文件:** `train/pplm.py`

```python
class MultiObjectiveClassifier(nn.Module):
    def __init__(self, hidden_size=256, property_names=['LogP', 'QED']):
        # 共享特征提取器
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # 每个属性独立的分类头
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            ) for name in property_names
        })
```

### 3.3 Pareto梯度聚合

多目标优化使用加权梯度聚合:

```python
def compute_multi_objective_gradient(hidden, target_classes, weights):
    gradients = {}
    for name in property_names:
        loss = -log_softmax(classifier(hidden))[target]
        gradients[name] = normalize(grad(loss, hidden))

    # 加权聚合
    aggregated = sum(weights[name] * gradients[name] for name in names)
    return aggregated
```

### 3.4 PPLM参数

| 参数 | 值 | 说明 |
|------|-----|------|
| stepsize | 0.01 | 梯度步长 |
| num_iterations | 3 | 每token迭代次数 |
| gm_scale | 0.5 | 扰动缩放因子 |
| temperature | 0.85 | 采样温度 |
| top_k | 30 | Top-k过滤 |

---

## 4. 基于熵的不确定性采集策略

### 4.1 理论基础

在主动学习框架中，我们采用**基于熵的不确定性采集策略（Entropy-based Uncertainty Acquisition）**来选择最具信息量的样本。该策略的核心思想是：分类器预测不确定性越高的样本，对模型改进的潜在价值越大。

#### 4.1.1 信息熵定义

对于离散概率分布，香农熵（Shannon Entropy）定义为：

$$H(X) = -\sum_{i=1}^{K} p_i \log p_i$$

其中 $p_i$ 是第 $i$ 个类别的预测概率，$K$ 是类别总数。

**熵的性质：**
- **最小值 (H=0):** 当分布完全确定时（某个 $p_i=1$），熵为0
- **最大值 (H=log K):** 当分布均匀时（所有 $p_i=1/K$），熵达到最大
- **单调性:** 预测越不确定，熵越高

#### 4.1.2 二元分类的熵

对于本项目中的属性分类（如LogP>2.5或QED>0.6），采用二元交叉熵：

$$H_{binary}(p) = -p \log p - (1-p) \log(1-p)$$

其中 $p$ 是正类（满足阈值条件）的预测概率。

**二元熵特性：**
- 当 $p=0$ 或 $p=1$ 时，$H=0$（完全确定）
- 当 $p=0.5$ 时，$H=\log 2 \approx 0.693$（最大不确定性）

#### 4.1.3 多目标不确定性聚合

对于多属性优化（LogP和QED），我们计算各属性熵的平均值：

$$H_{total} = \frac{1}{M} \sum_{j=1}^{M} H_j(p_j)$$

其中 $M$ 是属性数量，$H_j$ 是第 $j$ 个属性的二元熵。

### 4.2 算法实现

#### 4.2.1 不确定性计算

**核心代码 (`pplm.py:1013-1031`):**

```python
def compute_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    计算预测不确定性（熵）用于样本采集。

    Args:
        hidden_states: [N, H] 分子的隐藏状态表示

    Returns:
        uncertainties: [N] 每个分子的不确定性分数
    """
    self.classifier.eval()
    with torch.no_grad():
        if isinstance(self.classifier, MultiObjectiveClassifier):
            # 多目标情况：计算每个属性的二元熵，取平均
            probs = self.classifier.get_probabilities(hidden_states)
            uncertainties = []
            for name, p in probs.items():
                # 二元交叉熵: H = -p*log(p) - (1-p)*log(1-p)
                ent = -p * torch.log(p + 1e-10) - (1-p) * torch.log(1-p + 1e-10)
                uncertainties.append(ent)
            return torch.stack(uncertainties, dim=-1).mean(dim=-1)
        else:
            # 单目标情况：计算softmax熵
            logits = self.classifier(hidden_states)
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            return entropy
```

#### 4.2.2 样本选择策略

**采集函数 (`pplm.py:1033-1055`):**

```python
def select_samples(self, smiles_list, hidden_states, n_select, strategy='uncertainty'):
    """
    使用采集函数选择信息量最大的样本。

    策略选项:
    - 'uncertainty': 选择熵最高的样本（分类器最不确定）
    - 'diversity': 选择分子指纹距离最大的样本
    - 'pareto': 基于Pareto前沿选择多目标最优样本
    """
    if strategy == 'uncertainty':
        # 计算所有样本的不确定性分数
        scores = self.compute_uncertainty(hidden_states)
        # 选择不确定性最高的n_select个样本
        _, indices = torch.topk(scores, n_select)
        return indices.cpu().tolist()
```

### 4.3 主动学习循环

```
算法: 基于熵不确定性的主动学习PPLM

输入: 预训练生成模型G, 种子数据D_0, 迭代次数T, 每轮生成数N, 选择数K
输出: 优化后的分类器C, 高质量分子集合M

1. 初始化: 用种子数据D_0训练分类器C
2. FOR t = 1 TO T:
   a. 使用PPLM(G, C)生成N个分子候选集 S_t
   b. 验证分子有效性，过滤无效分子
   c. 使用Oracle评估分子属性（LogP, QED）
   d. 计算每个分子的不确定性: u_i = H(C(h_i))
   e. 选择不确定性最高的K个样本: D_t = TopK(S_t, u, K)
   f. 更新分类器: C ← Train(C, D_t)
   g. 更新分子集合: M ← M ∪ ValidMolecules(S_t)
3. RETURN C, M
```

### 4.4 为什么选择熵不确定性？

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **熵不确定性** | 计算高效、理论基础扎实、无需额外采样 | 仅反映预测不确定性 | 分类边界探索 |
| MC Dropout | 可分解认知/偶然不确定性 | 需多次前向传播、计算开销大 | 模型不确定性分析 |
| 多样性采样 | 保证样本多样性 | 忽略模型预测信息 | 覆盖化学空间 |

**选择熵不确定性的理由：**
1. **计算效率:** 单次前向传播即可获得不确定性估计
2. **理论保证:** 信息论基础，最大化信息增益
3. **实践有效:** 在分子属性优化任务中表现稳定

### 4.5 主动学习的作用机制分析

#### 4.5.1 主动学习驱动的核心目标

在本框架中，主动学习**主要强化的是属性分类器，而非直接强化生成模型的定向生成能力**。

**组件更新情况：**

| 组件 | 是否被主动学习更新 | 说明 |
|------|-------------------|------|
| GPT生成模型 | ❌ 否 | 权重在整个过程中保持固定 |
| PPLM属性分类器 | ✅ 是 | 每轮迭代用新采集的样本更新 |

#### 4.5.2 信息增益的传递路径

```
主动学习循环的信息流:

┌─────────────────────────────────────────────────────────┐
│  1. PPLM生成 → 使用当前分类器引导GPT生成分子            │
│  2. Oracle评估 → 获取分子的真实属性标签                 │
│  3. 不确定性采集 → 选择分类器最不确定的样本             │
│  4. 分类器更新 → 用新样本训练分类器 (GPT不更新)         │
│  5. 循环 → 更准确的分类器提供更精确的PPLM引导           │
└─────────────────────────────────────────────────────────┘
```

#### 4.5.3 主动学习的实际增益

**直接增益：**
- 分类器在「决策边界」附近获得更多训练样本
- 分类器对「模糊区域」（如LogP≈2.5）的预测更准确

**间接增益：**
- 更准确的分类器 → 更精确的PPLM梯度引导 → 生成分子更符合目标

**效果限制：**
- 生成模型GPT本身没有学习目标属性的知识
- PPLM引导强度受限于validity约束（参数保守：stepsize=0.01, gm_scale=0.5）
- 实验结果：LogP>2.5从54%提升到57%（+3%），提升幅度有限

#### 4.5.4 框架定位

本方法更准确的描述是：

> **「基于熵不确定性采集的分类器主动学习 + PPLM可控生成」**

而非「主动学习驱动的定向分子生成」。主动学习的核心贡献在于**高效地改进属性分类器**，使其能够更准确地引导PPLM生成过程。

---

## 5. 实验结果

### 5.1 实验1: 扩大规模主动学习实验

**实验配置:**
```python
n_iterations = 10      # 主动学习迭代轮数
n_generate = 200       # 每轮生成分子数
n_select = 20          # 每轮选择样本数
n_runs = 3             # 独立运行次数
acquisition = 'uncertainty'
thresholds = {'LogP': 2.5, 'QED': 0.6}
```

**实验结果 (3次独立运行的均值±标准差):**

| 指标 | 结果 |
|------|------|
| LogP > 2.5 | **56.0% ± 3.2%** |
| QED > 0.6 | **85.1% ± 2.3%** |
| 有效性 (Validity) | 99-100% |

**结果文件:** `experiments/results/exp1_scaled_al_20260202_195123.pkl`

### 5.2 实验2: 基线对比实验

**对比方法:**
1. **Random (随机生成):** 无PPLM引导的基础模型采样
2. **Single-PPLM (单目标):** 仅优化LogP的PPLM生成
3. **Multi-PPLM (多目标):** 同时优化LogP和QED的PPLM生成

**实验结果:**

| 方法 | 有效性 | 唯一性 | 新颖性 | 多样性 | LogP>2.5 | QED>0.6 |
|------|--------|--------|--------|--------|----------|---------|
| Random | 100.0% | 100.0% | 99.8% | 86.7% | 54.2% | 83.4% |
| Single-PPLM | 99.2% | 100.0% | 100.0% | 87.3% | **57.3%** | 86.5% |
| Multi-PPLM | 99.6% | 100.0% | 100.0% | 86.2% | 55.0% | **86.5%** |

**结果文件:** `experiments/results/exp2_baseline_20260202_195430.pkl`

### 5.3 分子质量指标定义

| 指标 | 定义 | 计算方法 |
|------|------|----------|
| **有效性 (Validity)** | 生成的SMILES能被RDKit解析的比例 | `Chem.MolFromSmiles(s) is not None` |
| **唯一性 (Uniqueness)** | 有效分子中不重复的比例 | `len(set(valid)) / len(valid)` |
| **新颖性 (Novelty)** | 不在训练集中的分子比例 | `len(generated - training) / len(generated)` |
| **多样性 (Diversity)** | 分子间平均Tanimoto距离 | `1 - mean(Tanimoto(fp_i, fp_j))` |

**Morgan指纹计算:**
```python
from rdkit import Chem
from rdkit.Chem import AllChem

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
```

### 5.4 关键发现

1. **有效性保持:** 所有方法均达到99-100%有效性，证明底座模型质量优秀
2. **PPLM有效性:** 单目标PPLM将LogP>2.5从54.2%提升到57.3% (+3.1%)
3. **多样性保持:** PPLM引导不损害分子多样性（86-87%）
4. **新颖性:** 生成分子几乎100%为训练集外的新分子

---

## 6. 文件结构

```
molgpt/
├── train/
│   ├── model.py              # GPT模型定义
│   ├── pplm.py               # PPLM实现 (含不确定性采集)
│   ├── uncertainty_methods.py # 不确定性估计方法
│   ├── train_fullvocab.py    # 完整词汇表训练
│   ├── pplm_al_demo.py       # 主动学习演示
│   └── utils.py              # 工具函数
│
├── experiments/
│   ├── run_full_experiments.py  # 扩大规模实验脚本
│   ├── run_exp2.py              # 基线对比实验脚本
│   ├── generate_report.py       # 报告生成脚本
│   └── results/
│       ├── exp1_scaled_al_*.pkl # 实验1结果
│       └── exp2_baseline_*.pkl  # 实验2结果
│
├── datasets/
│   └── moses2.csv            # MOSES2数据集
│
├── cond_gpt/weights/
│   ├── fullvocab_gpt.pt      # 最佳模型权重
│   └── fullvocab_stoi.json   # 词汇表副本
│
├── moses2_stoi_full.json     # 完整词汇表 (63 tokens)
└── TECHNICAL_REPORT.md       # 本报告
```

---

## 7. 关键问题与解决方案

### 7.1 词汇表不完整问题

**问题:** 原词汇表仅26个token，导致validity仅~32%

**原因分析:**
- 缺失立体化学标记: `[C@@H]`, `[C@H]`
- 缺失带电原子: `[N+]`, `[O-]`
- 缺失部分环编号: `7`, `8`, `9`
- 缺失元素: `P`, `I`

**解决方案:**
1. 扫描全部训练数据提取完整token集合
2. 创建63 token的完整词汇表
3. 重新训练模型

**结果:** Validity从32%提升到100%

### 7.2 RDKit多进程问题

**问题:** DataLoader使用num_workers>0时RDKit导入失败

**解决方案:** 设置`num_workers=0`

### 7.3 NumPy版本兼容性

**问题:** NumPy 2.0与RDKit不兼容

**解决方案:** `pip install 'numpy<2'`

---

## 8. 使用指南

### 8.1 训练基础模型

```bash
cd /scratch/sutianhao/data/edl_transformer/molgpt/train
python train_fullvocab.py
```

### 8.2 运行PPLM演示

```bash
python pplm_al_demo.py
```

### 8.3 加载预训练模型

```python
import torch
import json
from model import GPT, GPTConfig

# 加载词汇表
with open('../moses2_stoi_full.json') as f:
    stoi = json.load(f)

# 配置模型
config = GPTConfig(
    vocab_size=len(stoi),  # 63
    block_size=71,
    n_layer=8,
    n_head=8,
    n_embd=256
)

# 加载权重
model = GPT(config)
model.load_state_dict(torch.load('weights/fullvocab_gpt.pt'))
model.eval()
```

### 8.4 生成分子

```python
# 从'C'开始生成
start_token = stoi['C']
x = torch.tensor([[start_token]])

with torch.no_grad():
    for _ in range(70):
        logits, _, _ = model(x)
        probs = torch.softmax(logits[:, -1, :] / 0.9, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

# 解码
itos = {v: k for k, v in stoi.items()}
smiles = ''.join([itos[int(t)] for t in x[0]])
smiles = smiles.replace('<', '')
print(smiles)
```

---

## 9. 性能总结

| 指标 | 值 |
|------|-----|
| 模型参数量 | ~5.2M |
| 词汇表大小 | 63 tokens |
| 训练Epochs | 50 |
| 最佳Validity | **100%** |
| PPLM Validity | 99-100% |
| 唯一性 (Uniqueness) | 100% |
| 新颖性 (Novelty) | 99.8-100% |
| 多样性 (Diversity) | 86-87% |
| LogP>2.5 (Random) | 54.2% |
| LogP>2.5 (Single-PPLM) | **57.3%** (+3.1%) |
| QED>0.6 (Random) | 83.4% |
| QED>0.6 (PPLM) | **86.5%** (+3.1%) |

---

## 10. 未来工作

1. **更多属性优化:** 添加SA Score, MW等属性
2. **条件生成:** 基于scaffold的分子生成
3. **EDL不确定性:** 集成Evidential Deep Learning
4. **更大规模训练:** 使用更多数据和更大模型

---

*报告生成日期: 2026-02-02*

*模型: fullvocab_gpt.pt (Epoch 35, 100% Validity)*
