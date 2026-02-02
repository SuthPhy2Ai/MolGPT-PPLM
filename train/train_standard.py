#!/usr/bin/env python3
"""
Quick training script for standard (non-EDL) GPT model.
This creates a baseline model for PPLM testing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from model import GPT, GPTConfig


class SimpleSmileDataset(Dataset):
    """Simplified SMILES dataset for quick training."""

    def __init__(self, smiles_list, stoi, block_size):
        self.smiles_list = smiles_list
        self.stoi = stoi
        self.block_size = block_size
        self.pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx].strip()

        # Tokenize
        tokens = self.regex.findall(smiles)

        # Pad with '<' to block_size
        tokens = tokens + ['<'] * (self.block_size - len(tokens))
        tokens = tokens[:self.block_size]

        # Convert to indices
        ids = [self.stoi.get(t, 0) for t in tokens]

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        # Dummy prop and scaffold
        prop = torch.zeros(1)
        scaffold = torch.zeros(1, dtype=torch.long)

        return x, y, prop, scaffold


def main():
    # Paths
    data_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/datasets"
    weights_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/cond_gpt/weights"

    # Load vocabulary
    stoi_path = "../moses2_stoi.json"
    with open(stoi_path, 'r') as f:
        stoi = json.load(f)
    itos = {v: k for k, v in stoi.items()}

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_dir + "/moses2.csv")
    train_smiles = df[df['split'] == 'train']['smiles'].tolist()
    test_smiles = df[df['split'] == 'test']['smiles'].tolist()

    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")

    # Create datasets
    train_dataset = SimpleSmileDataset(train_smiles, stoi, 72)
    test_dataset = SimpleSmileDataset(test_smiles, stoi, 72)

    # Model config (standard, no EDL)
    config = GPTConfig(
        vocab_size=len(stoi),
        block_size=71,  # block_size - 1 for x
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    best_loss = float('inf')
    epochs = 30

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, y, prop, scaffold in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, loss, _ = model(x, y, None, None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Eval
        model.eval()
        test_losses = []
        with torch.no_grad():
            for x, y, prop, scaffold in test_loader:
                x, y = x.to(device), y.to(device)
                logits, loss, _ = model(x, y, None, None)
                test_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        print(f"Epoch {epoch+1}: train={train_loss:.4f}, test={test_loss:.4f}")

        # Save best
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), weights_dir + "/standard_gpt.pt")
            print(f"  Saved best model (test_loss={test_loss:.4f})")

    print(f"\nTraining complete. Best test loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
