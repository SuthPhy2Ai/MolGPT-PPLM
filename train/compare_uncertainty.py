#!/usr/bin/env python3
"""
Compare Uncertainty Estimation Methods for Autoregressive Models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import re

from model import GPT, GPTConfig
from dataset import SmileDataset
from uncertainty_methods import (
    MCDropoutUncertainty,
    EntropyUncertainty,
    TemperatureScaling,
    EDLUncertainty
)


def load_model(weights_path, vocab_size=64, block_size=72, use_edl=False):
    """Load trained model."""
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
        use_edl=use_edl
    )
    model = GPT(config)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found, using random weights")

    return model


def evaluate_uncertainty(model, test_smiles, stoi, device='cuda'):
    """Evaluate uncertainty quality with simple input."""
    model = model.to(device)
    model.eval()

    results = {
        'entropy': [],
        'mc_dropout': [],
        'temp_0.5': [],
        'temp_1.0': [],
        'temp_2.0': []
    }

    # Initialize estimators
    entropy_est = EntropyUncertainty(model)
    mc_est = MCDropoutUncertainty(model, n_samples=5)
    temp_ests = {
        'temp_0.5': TemperatureScaling(model, 0.5),
        'temp_1.0': TemperatureScaling(model, 1.0),
        'temp_2.0': TemperatureScaling(model, 2.0)
    }

    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    print("Evaluating uncertainty methods...")
    for smiles in tqdm(test_smiles[:100]):
        try:
            tokens = regex.findall(smiles)
            x = torch.tensor([stoi.get(s, 0) for s in tokens], dtype=torch.long)
            x = x.unsqueeze(0).to(device)

            # Entropy
            res = entropy_est.estimate(x)
            results['entropy'].append(res['uncertainty'].mean().item())

            # MC Dropout
            res = mc_est.estimate(x)
            results['mc_dropout'].append(res['uncertainty'].mean().item())

            # Temperature scaling
            for name, est in temp_ests.items():
                res = est.estimate(x)
                results[name].append(res['uncertainty'].mean().item())
        except Exception as e:
            continue

    return results


def print_comparison(results):
    """Print comparison table."""
    print("\n" + "="*60)
    print("UNCERTAINTY METHOD COMPARISON")
    print("="*60)

    for method, values in results.items():
        mean_unc = np.mean(values)
        std_unc = np.std(values)
        print(f"{method:15s}: mean={mean_unc:.4f}, std={std_unc:.4f}")

    print("="*60)


def main():
    """Main comparison function."""
    # Paths
    data_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/datasets"
    weights_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/cond_gpt/weights"

    # Load vocabulary
    stoi_path = "../moses2_stoi.json"
    with open(stoi_path, 'r') as f:
        stoi = json.load(f)

    # Load test data
    print("Loading test data...")
    test_file = data_dir + "/zinc_250k.txt"
    with open(test_file, 'r') as f:
        test_smiles = [line.strip() for line in f.readlines()[:200]]

    # Load standard model (no EDL)
    weights_path = weights_dir + "/moses2.pt"
    model = load_model(weights_path, use_edl=False)

    # Evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = evaluate_uncertainty(model, test_smiles, stoi, device)

    # Print results
    print_comparison(results)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv('../logs/uncertainty_comparison.csv', index=False)
    print("Results saved to ../logs/uncertainty_comparison.csv")


if __name__ == "__main__":
    main()
