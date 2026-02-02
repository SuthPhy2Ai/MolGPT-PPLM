#!/usr/bin/env python3
"""
Extended training for high validity molecular generation.
Key: More epochs with standard cross-entropy loss.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from model import GPT, GPTConfig


class SimpleSmileDataset(Dataset):
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
        tokens = self.regex.findall(smiles)
        tokens = tokens + ['<'] * (self.block_size - len(tokens))
        tokens = tokens[:self.block_size]
        ids = [self.stoi.get(t, 0) for t in tokens]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def validate_generation(model, stoi, itos, device, n_samples=200):
    from rdkit import Chem
    model.eval()
    start_token = stoi.get('C', 0)
    x = torch.tensor([[start_token]], dtype=torch.long, device=device)
    x = x.repeat(n_samples, 1)

    with torch.no_grad():
        for _ in range(70):
            logits, _, _ = model(x)
            probs = torch.softmax(logits[:, -1, :] / 0.9, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    valid = 0
    for i in range(n_samples):
        smiles = ''.join([itos.get(int(t), '') for t in x[i]])
        smiles = smiles.replace('<', '').strip()
        if Chem.MolFromSmiles(smiles) is not None:
            valid += 1
    return valid / n_samples * 100


def main():
    data_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/datasets"
    weights_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/cond_gpt/weights"

    with open("../moses2_stoi.json", 'r') as f:
        stoi = json.load(f)
    itos = {v: k for k, v in stoi.items()}

    print("Loading data...")
    df = pd.read_csv(data_dir + "/moses2.csv")
    train_smiles = df[df['split'] == 'train']['smiles'].tolist()
    test_smiles = df[df['split'] == 'test']['smiles'].tolist()
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")

    train_dataset = SimpleSmileDataset(train_smiles, stoi, 72)
    test_dataset = SimpleSmileDataset(test_smiles, stoi, 72)

    # Larger model
    config = GPTConfig(
        vocab_size=len(stoi), block_size=71,
        n_layer=8, n_head=8, n_embd=256,
        num_props=0, scaffold=False, scaffold_maxlen=0, lstm=False
    )

    model = GPT(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Extended training
    epochs = 100
    batch_size = 128
    lr = 3e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    best_validity = 0
    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss, _ = model(x, y, None, None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        # Eval every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_losses = []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    _, loss, _ = model(x, y, None, None)
                    test_losses.append(loss.item())

            validity = validate_generation(model, stoi, itos, device)
            print(f"Epoch {epoch+1}: train={np.mean(train_losses):.4f}, "
                  f"test={np.mean(test_losses):.4f}, validity={validity:.1f}%")

            if validity > best_validity:
                best_validity = validity
                torch.save(model.state_dict(), weights_dir + "/extended_gpt.pt")
                print(f"  -> Saved (validity={validity:.1f}%)")

    print(f"\nBest validity: {best_validity:.1f}%")


if __name__ == "__main__":
    main()
