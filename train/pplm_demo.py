#!/usr/bin/env python3
"""
PPLM Demo: Guided Molecular Generation

This script demonstrates how to use PPLM for controlled molecular generation.
Example: Generate molecules with high LogP (lipophilicity)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import re
from tqdm import tqdm

from model import GPT, GPTConfig
from pplm import (
    AttributeClassifier,
    PPLMGenerator,
    train_attribute_classifier,
    create_binary_labels,
    compute_molecular_properties
)


def load_model(weights_path, vocab_size=64, block_size=71):
    """Load pretrained GPT model."""
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
        use_edl=False
    )
    model = GPT(config)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found, using random weights")

    return model, config


def main():
    """Main demo function."""
    # Paths
    data_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/datasets"
    weights_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/cond_gpt/weights"

    # Load vocabulary
    stoi_path = "../moses2_stoi.json"
    with open(stoi_path, 'r') as f:
        stoi = json.load(f)
    itos = {v: k for k, v in stoi.items()}

    # Load training data
    print("Loading training data...")
    import pandas as pd
    data_file = data_dir + "/moses2.csv"
    df = pd.read_csv(data_file)
    smiles_list = df['smiles'].tolist()[:1000]

    print(f"Loaded {len(smiles_list)} molecules")

    # Create binary labels based on LogP
    print("Computing molecular properties...")
    labels = create_binary_labels(smiles_list, property_name='LogP')
    print(f"Label distribution: {sum(labels)} high LogP, {len(labels)-sum(labels)} low LogP")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights_path = weights_dir + "/standard_gpt.pt"
    model, config = load_model(weights_path, vocab_size=len(stoi))
    model = model.to(device)

    # Create and train attribute classifier
    print("\n" + "="*50)
    print("Training Attribute Classifier")
    print("="*50)

    classifier = AttributeClassifier(
        hidden_size=config.n_embd,
        num_classes=2,
        hidden_dim=128
    )

    classifier = train_attribute_classifier(
        model=model,
        classifier=classifier,
        smiles_list=smiles_list,
        labels=labels,
        stoi=stoi,
        epochs=5,
        batch_size=32,
        device=device
    )

    # Create PPLM generator
    print("\n" + "="*50)
    print("PPLM Guided Generation")
    print("="*50)

    pplm = PPLMGenerator(
        model=model,
        classifier=classifier,
        stepsize=0.02,  # Increased for stronger guidance
        num_iterations=5,  # More iterations
        gm_scale=0.8,   # Stronger perturbation
        device=device
    )

    # Generate molecules
    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    # Start with 'C' as seed
    context = "C"
    x = torch.tensor(
        [[stoi.get(s, 0) for s in regex.findall(context)]],
        dtype=torch.long, device=device
    ).repeat(50, 1)  # Generate 50 molecules for better statistics

    # Generate with PPLM (target high LogP)
    print("\nGenerating with PPLM (target: high LogP)...")
    generated, uncertainties = pplm.generate_with_pplm(
        input_ids=x,
        max_length=72,
        target_class=1,  # 1 = high LogP
        temperature=0.8,
        top_k=20  # Increased from 10 for more conservative sampling
    )

    # Decode generated molecules
    print("\nGenerated Molecules (PPLM - High LogP):")
    print("-" * 60)

    from rdkit import Chem
    import numpy as np

    valid_count = 0
    high_logp_count = 0
    logp_values = []

    for i, gen in enumerate(generated):
        smiles = ''.join([itos.get(int(t), '') for t in gen])
        smiles = smiles.replace('<', '').strip()

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            valid_count += 1
            props = compute_molecular_properties(smiles)
            logp = props.get('LogP', 0)
            logp_values.append(logp)
            if logp > 2.5:
                high_logp_count += 1
            if i < 10:  # Only print first 10
                print(f"{i+1}. {smiles[:45]}... LogP={logp:.2f}")

    print("-" * 60)
    print(f"PPLM Results: Valid={valid_count}/50 ({valid_count*2}%), High LogP={high_logp_count}/{valid_count}")
    if logp_values:
        print(f"LogP: mean={np.mean(logp_values):.2f}, std={np.std(logp_values):.2f}")

    # Baseline comparison (no PPLM)
    print("\n" + "="*50)
    print("Baseline Generation (No PPLM)")
    print("="*50)

    from utils import sample
    x_baseline = torch.tensor(
        [[stoi.get(s, 0) for s in regex.findall(context)]],
        dtype=torch.long, device=device
    ).repeat(50, 1)

    baseline_gen = sample(model, x_baseline, 71, temperature=0.8, sample=True, top_k=20)

    baseline_valid = 0
    baseline_high = 0
    baseline_logp = []

    for i, gen in enumerate(baseline_gen):
        smiles = ''.join([itos.get(int(t), '') for t in gen])
        smiles = smiles.replace('<', '').strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            baseline_valid += 1
            props = compute_molecular_properties(smiles)
            logp = props.get('LogP', 0)
            baseline_logp.append(logp)
            if logp > 2.5:
                baseline_high += 1

    print(f"Baseline Results: Valid={baseline_valid}/50 ({baseline_valid*2}%), High LogP={baseline_high}/{baseline_valid}")
    if baseline_logp:
        print(f"LogP: mean={np.mean(baseline_logp):.2f}, std={np.std(baseline_logp):.2f}")

    # Summary
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Method':<15} {'Valid%':<10} {'HighLogP%':<12} {'MeanLogP':<10}")
    print("-"*50)
    pplm_high_pct = high_logp_count/valid_count*100 if valid_count > 0 else 0
    base_high_pct = baseline_high/baseline_valid*100 if baseline_valid > 0 else 0
    pplm_mean = np.mean(logp_values) if logp_values else 0
    base_mean = np.mean(baseline_logp) if baseline_logp else 0
    print(f"{'PPLM':<15} {valid_count*2:<10} {pplm_high_pct:<12.1f} {pplm_mean:<10.2f}")
    print(f"{'Baseline':<15} {baseline_valid*2:<10} {base_high_pct:<12.1f} {base_mean:<10.2f}")
    print("="*50)


if __name__ == "__main__":
    main()
