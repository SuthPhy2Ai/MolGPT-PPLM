# MolGPT with PPLM

Molecular generation using GPT with Plug-and-Play Language Model (PPLM) for guided generation.

## Project Structure

```
molgpt/
├── train/
│   ├── model.py               # GPT model
│   ├── trainer.py             # Training loop
│   ├── dataset.py             # SMILES dataset loader
│   ├── utils.py               # Utility functions
│   │
│   ├── train.py               # Main training script
│   ├── train_standard.py      # Quick training script
│   │
│   ├── pplm.py                # PPLM implementation
│   ├── pplm_demo.py           # PPLM demo (LogP)
│   ├── pplm_advanced_demo.py  # Novelty + Fragment demo
│   ├── pplm_al_demo.py        # Active Learning + Pareto demo
│   │
│   └── uncertainty_methods.py # Uncertainty estimation
│
├── datasets/                   # Data files
├── cond_gpt/weights/          # Model weights
└── moses2_stoi.json           # Vocabulary
```

## Installation

```bash
pip install -r requirements_edl.txt
```

## Quick Start

### 1. Train Model

```bash
cd train
python train_standard.py
```

### 2. PPLM Guided Generation

```bash
# LogP guidance
python pplm_demo.py

# Novelty + Fragment
python pplm_advanced_demo.py

# Active Learning + Multi-objective
python pplm_al_demo.py
```

---

## PPLM Modes

### Mode 1: Property Guidance

```python
from pplm import PPLMGenerator, AttributeClassifier

pplm = PPLMGenerator(
    model=model,
    classifier=classifier,
    stepsize=0.02,
    num_iterations=5,
    gm_scale=0.8
)

generated, _ = pplm.generate_with_pplm(
    input_ids=x,
    target_class=1,  # high LogP
    temperature=0.8
)
```

### Mode 2: Novelty Guidance

```python
from pplm import create_novelty_labels

labels = create_novelty_labels(
    smiles_list,
    reference_smiles,
    similarity_threshold=0.6
)
# Use target_class=0 for novel molecules
```

### Mode 3: Fragment Constraint

```python
from pplm import FragmentConstrainedGenerator

frag_gen = FragmentConstrainedGenerator(model, stoi)

# Prefix mode
results = frag_gen.generate_with_prefix("c1ccccc1")

# Constraint mode
results = frag_gen.generate_with_constraint("c1ccccc1")
```

---

## Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| stepsize | 0.01-0.05 | Gradient step |
| num_iterations | 3-10 | Iterations/token |
| gm_scale | 0.5-1.0 | Perturbation |
| temperature | 0.7-1.2 | Sampling temp |
| top_k | 10-50 | Top-k filter |

---

## Advanced Features

### Mode 4: Multi-Objective Pareto Optimization

Optimize multiple properties simultaneously using Pareto-based gradient aggregation.

```python
from pplm import MultiObjectiveClassifier, MultiObjectivePPLM

# Create multi-objective classifier
mo_classifier = MultiObjectiveClassifier(
    hidden_size=192,
    property_names=['LogP', 'QED'],
    hidden_dim=128
)

# Create multi-objective PPLM
mo_pplm = MultiObjectivePPLM(
    model=model,
    classifier=mo_classifier,
    stepsize=0.02,
    gm_scale=0.8
)

# Generate with multiple objectives
generated, scores = mo_pplm.generate_pareto_optimal(
    input_ids=x,
    max_length=72,
    target_classes={'LogP': 1, 'QED': 1},
    weights={'LogP': 0.5, 'QED': 0.5}
)
```

### Mode 5: Active Learning Loop

Iteratively improve generation through active learning.

```python
from pplm import ActiveLearningPPLM, MultiObjectiveClassifier

al_pplm = ActiveLearningPPLM(
    model=model,
    classifier=mo_classifier,
    stoi=stoi,
    property_names=['LogP', 'QED']
)

results = al_pplm.run_active_learning_loop(
    initial_smiles=train_smiles[:100],
    thresholds={'LogP': 2.5, 'QED': 0.6},
    n_iterations=5,
    n_generate=100,
    n_select=20,
    acquisition='uncertainty'  # or 'diversity', 'pareto'
)
```

### Pareto Front Analysis

```python
from pplm import compute_pareto_front, compute_crowding_distance

# Compute Pareto front
pareto_front, pareto_idx = compute_pareto_front(objectives)

# Compute crowding distance for diversity
crowding = compute_crowding_distance(pareto_front)
```
