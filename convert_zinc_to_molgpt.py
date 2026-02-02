#!/usr/bin/env python
"""
Convert ZINC-250K dataset to MolGPT format
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles):
    """Get Murcko scaffold from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None

def clean_smiles(smiles):
    """Clean SMILES string by removing newlines and extra spaces"""
    if isinstance(smiles, str):
        return smiles.replace('\n', '').replace('\r', '').strip()
    return smiles

print("Loading ZINC-250K dataset...")
# Read the dataset with proper handling of quoted fields
df = pd.read_csv('datasets/zinc_250k.txt', quotechar='"')

print(f"Original dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Clean SMILES strings
print("Cleaning SMILES strings...")
df['smiles'] = df['smiles'].apply(clean_smiles)

# Remove invalid SMILES
print("Removing invalid SMILES...")
valid_mask = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None if isinstance(x, str) else False)
df = df[valid_mask].reset_index(drop=True)
print(f"Valid molecules: {len(df)}")

# Calculate scaffold SMILES
print("Calculating scaffold SMILES...")
df['scaffold_smiles'] = df['smiles'].apply(get_scaffold)

# Remove molecules without valid scaffolds
df = df[df['scaffold_smiles'].notna()].reset_index(drop=True)
print(f"Molecules with valid scaffolds: {len(df)}")

# Add split column (80% train, 10% test, 10% val)
print("Creating train/test/val splits...")
np.random.seed(42)
n = len(df)
indices = np.random.permutation(n)

train_size = int(0.8 * n)
test_size = int(0.1 * n)

df['split'] = 'train'
df.loc[indices[train_size:train_size+test_size], 'split'] = 'test'
df.loc[indices[train_size+test_size:], 'split'] = 'val'

print(f"Train: {(df['split']=='train').sum()}")
print(f"Test: {(df['split']=='test').sum()}")
print(f"Val: {(df['split']=='val').sum()}")

# Rename columns to match MolGPT format
# Keep: smiles, logP, qed, SAS, scaffold_smiles, split
df.columns = df.columns.str.lower()

# Save as moses2.csv (compatible with MolGPT training scripts)
output_file = 'datasets/moses2.csv'
print(f"\nSaving to {output_file}...")
df.to_csv(output_file, index=False)

print(f"\nDataset conversion complete!")
print(f"Final shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nYou can now use this dataset with MolGPT training scripts:")
print(f"  ./train_moses.sh")
