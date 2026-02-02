#!/usr/bin/env python3
"""
Active Learning + Multi-Objective Pareto PPLM Demo

Demonstrates:
1. Multi-objective optimization (LogP + QED)
2. Pareto-based sample selection
3. Active learning loop for iterative improvement
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import numpy as np
from tqdm import tqdm

from model import GPT, GPTConfig
from pplm import (
    MultiObjectiveClassifier,
    MultiObjectivePPLM,
    ActiveLearningPPLM,
    compute_pareto_front,
    compute_crowding_distance,
    compute_molecular_properties,
    create_binary_labels
)


def load_model(weights_path, vocab_size=64, block_size=71, use_robust=False):
    """Load pretrained GPT model."""
    if use_robust:
        # Robust model config (larger)
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=8,
            n_head=8,
            n_embd=256,
            num_props=0,
            scaffold=False,
            scaffold_maxlen=0,
            lstm=False,
        )
    else:
        # Standard model config
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=6,
            n_head=4,
            n_embd=192,
            num_props=0,
            scaffold=False,
            scaffold_maxlen=0,
            lstm=False,
        )
    model = GPT(config)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found, using random weights")

    return model, config


def demo_multi_objective_pplm(model, config, stoi, train_smiles, device):
    """Demo: Multi-objective PPLM with LogP + QED."""
    print("\n" + "="*60)
    print("MODE 1: MULTI-OBJECTIVE PPLM")
    print("Goal: Optimize LogP AND QED simultaneously")
    print("="*60)

    property_names = ['LogP', 'QED']

    # Create multi-objective classifier
    mo_classifier = MultiObjectiveClassifier(
        hidden_size=config.n_embd,
        property_names=property_names,
        hidden_dim=128
    ).to(device)

    # Create labels for training
    print("\nPreparing training data...")
    sample_smiles = train_smiles[:500]

    logp_labels = create_binary_labels(sample_smiles, 'LogP', threshold=2.5)
    qed_labels = create_binary_labels(sample_smiles, 'QED', threshold=0.6)

    print(f"LogP labels: {sum(logp_labels)} high, {len(logp_labels)-sum(logp_labels)} low")
    print(f"QED labels: {sum(qed_labels)} high, {len(qed_labels)-sum(qed_labels)} low")

    return mo_classifier, property_names, sample_smiles, logp_labels, qed_labels


def demo_pareto_analysis(generated_smiles):
    """Demo: Pareto front analysis of generated molecules."""
    print("\n" + "="*60)
    print("MODE 2: PARETO FRONT ANALYSIS")
    print("="*60)

    from rdkit import Chem

    # Compute properties for all valid molecules
    objectives = []
    valid_smiles = []

    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            props = compute_molecular_properties(smi)
            if 'LogP' in props and 'QED' in props:
                objectives.append([props['LogP'], props['QED']])
                valid_smiles.append(smi)

    if len(objectives) < 2:
        print("Not enough valid molecules for Pareto analysis")
        return

    objectives = np.array(objectives)
    print(f"\nAnalyzing {len(objectives)} valid molecules...")

    # Compute Pareto front
    pareto_front, pareto_idx = compute_pareto_front(objectives)
    print(f"Pareto front size: {len(pareto_idx)}")

    # Compute crowding distance
    if len(pareto_front) > 2:
        crowding = compute_crowding_distance(pareto_front)
        print(f"Crowding distances: {crowding[:5]}...")

    # Show Pareto-optimal molecules
    print("\nPareto-optimal molecules:")
    for i, idx in enumerate(pareto_idx[:5]):
        smi = valid_smiles[idx]
        logp, qed = objectives[idx]
        print(f"  {i+1}. LogP={logp:.2f}, QED={qed:.3f}: {smi[:40]}...")

    return pareto_front, pareto_idx


def demo_active_learning(model, config, stoi, train_smiles, device):
    """Demo: Active learning loop with multi-objective optimization."""
    print("\n" + "="*60)
    print("MODE 3: ACTIVE LEARNING LOOP")
    print("Goal: Iteratively improve generation quality")
    print("="*60)

    property_names = ['LogP', 'QED']
    thresholds = {'LogP': 2.5, 'QED': 0.6}

    # Create classifier
    mo_classifier = MultiObjectiveClassifier(
        hidden_size=config.n_embd,
        property_names=property_names,
        hidden_dim=128
    ).to(device)

    # Create active learning system
    al_pplm = ActiveLearningPPLM(
        model=model,
        classifier=mo_classifier,
        stoi=stoi,
        property_names=property_names,
        device=device
    )

    # Run active learning
    print("\nRunning active learning loop...")
    results = al_pplm.run_active_learning_loop(
        initial_smiles=train_smiles[:100],
        thresholds=thresholds,
        n_iterations=3,
        n_generate=50,
        n_select=10,
        acquisition='uncertainty',
        target_classes={'LogP': 1, 'QED': 1},
        temperature=0.85,   # Slightly lower for more conservative sampling
        top_k=30,           # Moderate top_k (vocab_size=64)
        verbose=True
    )

    return results


def main():
    """Main demo function."""
    # Paths
    data_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/datasets"
    weights_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/cond_gpt/weights"

    # Load vocabulary - use full vocabulary for high validity
    stoi_path = "../moses2_stoi_full.json"
    with open(stoi_path, 'r') as f:
        stoi = json.load(f)
    print(f"Vocabulary size: {len(stoi)}")

    # Load training data
    print("Loading training data...")
    import pandas as pd
    data_file = data_dir + "/moses2.csv"
    df = pd.read_csv(data_file)
    train_smiles = df['smiles'].tolist()[:1000]
    print(f"Loaded {len(train_smiles)} molecules")

    # Load model - use fullvocab model (100% validity)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights_path = weights_dir + "/fullvocab_gpt.pt"
    model, config = load_model(weights_path, vocab_size=len(stoi), use_robust=True)
    model = model.to(device)

    # Demo 1: Multi-objective setup
    mo_classifier, prop_names, sample_smiles, logp_labels, qed_labels = \
        demo_multi_objective_pplm(model, config, stoi, train_smiles, device)

    # Demo 2: Active Learning
    results = demo_active_learning(model, config, stoi, train_smiles, device)

    # Summary
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nKey innovations implemented:")
    print("1. Multi-objective PPLM with Pareto gradient aggregation")
    print("2. Active learning loop with uncertainty acquisition")
    print("3. Pareto front analysis for multi-objective evaluation")


if __name__ == "__main__":
    main()
