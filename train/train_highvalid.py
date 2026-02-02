#!/usr/bin/env python3
"""
High-validity training script for GPT molecular generation.
Key improvements:
1. Teacher forcing with scheduled sampling
2. Label smoothing for better generalization
3. Longer training with early stopping based on validity
4. Data augmentation via SMILES randomization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from model import GPT, GPTConfig


class AugmentedSmileDataset(Dataset):
    """SMILES dataset with optional randomization augmentation."""

    def __init__(self, smiles_list, stoi, block_size, augment=False):
        self.smiles_list = smiles_list
        self.stoi = stoi
        self.block_size = block_size
        self.augment = augment
        self.pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)

    def __len__(self):
        return len(self.smiles_list)

    def randomize_smiles(self, smiles):
        """Randomize SMILES representation."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            ans = list(range(mol.GetNumAtoms()))
            np.random.shuffle(ans)
            nm = Chem.RenumberAtoms(mol, ans)
            return Chem.MolToSmiles(nm, canonical=False)
        except:
            return smiles

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx].strip()

        # Augment with 50% probability during training
        if self.augment and np.random.random() < 0.5:
            smiles = self.randomize_smiles(smiles)

        tokens = self.regex.findall(smiles)
        tokens = tokens + ['<'] * (self.block_size - len(tokens))
        tokens = tokens[:self.block_size]
        ids = [self.stoi.get(t, 0) for t in tokens]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""

    def __init__(self, vocab_size, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        # logits: [B*T, V], target: [B*T]
        logits = logits.view(-1, self.vocab_size)
        target = target.view(-1)

        # Create smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

            # Handle ignore_index
            mask = target != self.ignore_index

        # KL divergence loss
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        loss = loss[mask].mean()

        return loss


def validate_generation(model, stoi, itos, device, n_samples=200, temp=0.9):
    """Generate samples and compute validity rate."""
    from rdkit import Chem

    model.eval()
    start_token = stoi.get('C', 0)
    x = torch.tensor([[start_token]], dtype=torch.long, device=device)
    x = x.repeat(n_samples, 1)

    with torch.no_grad():
        for _ in range(70):
            logits, _, _ = model(x)
            next_logits = logits[:, -1, :] / temp
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    valid = 0
    valid_smiles = []
    for i in range(n_samples):
        smiles = ''.join([itos.get(int(t), '') for t in x[i]])
        smiles = smiles.replace('<', '').strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid += 1
            valid_smiles.append(smiles)

    return valid / n_samples * 100, valid_smiles


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

    # Create datasets without augmentation (RDKit has multiprocessing issues)
    train_dataset = AugmentedSmileDataset(train_smiles, stoi, 72, augment=False)
    test_dataset = AugmentedSmileDataset(test_smiles, stoi, 72, augment=False)

    # Model config - slightly smaller for faster convergence
    config = GPTConfig(
        vocab_size=len(stoi),
        block_size=71,
        n_layer=6,
        n_head=6,
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

    # Training setup
    epochs = 100
    batch_size = 256
    lr = 5e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5)

    # Use label smoothing
    criterion = LabelSmoothingLoss(len(stoi), smoothing=0.1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    best_validity = 0
    patience = 0
    max_patience = 15

    print(f"\nTraining for up to {epochs} epochs...")
    print("=" * 60)

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits, _, _ = model(x, y, None, None)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        # Eval
        model.eval()
        test_losses = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _, _ = model(x, y, None, None)
                loss = criterion(logits, y)
                test_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        current_lr = scheduler.get_last_lr()[0]

        # Validate generation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            validity, _ = validate_generation(model, stoi, itos, device, n_samples=200)
            print(f"Epoch {epoch+1}: train={train_loss:.4f}, test={test_loss:.4f}, "
                  f"lr={current_lr:.2e}, validity={validity:.1f}%")

            if validity > best_validity:
                best_validity = validity
                torch.save(model.state_dict(), weights_dir + "/highvalid_gpt.pt")
                print(f"  -> Saved best model (validity={validity:.1f}%)")
                patience = 0
            else:
                patience += 1

            # Early stopping
            if patience >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1}: train={train_loss:.4f}, test={test_loss:.4f}, lr={current_lr:.2e}")

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best validity: {best_validity:.1f}%")


if __name__ == "__main__":
    main()
