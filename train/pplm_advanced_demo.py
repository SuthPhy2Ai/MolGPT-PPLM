#!/usr/bin/env python3
"""
Advanced PPLM Demo: Novelty Guidance & Fragment Constraint

Two new generation modes:
1. Novelty Guidance: Generate molecules different from training set
2. Fragment Constraint: Keep specific substructures during generation
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
    NoveltyClassifier,
    FragmentConstrainedGenerator,
    train_attribute_classifier,
    create_novelty_labels,
    compute_fingerprint_similarity,
    compute_max_similarity,
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


def demo_novelty_guidance(model, config, stoi, train_smiles, device):
    """Demo: Generate novel molecules different from training set."""
    print("\n" + "="*60)
    print("MODE 1: NOVELTY GUIDANCE")
    print("Goal: Generate molecules DIFFERENT from training set")
    print("="*60)

    itos = {v: k for k, v in stoi.items()}

    # Create novelty classifier
    print("\nTraining novelty classifier...")
    novelty_classifier = NoveltyClassifier(
        hidden_size=config.n_embd,
        hidden_dim=128
    )

    # Create labels: 1=similar to training, 0=novel
    # Use subset for efficiency
    sample_smiles = train_smiles[:500]
    labels = create_novelty_labels(
        sample_smiles,
        train_smiles[:200],
        similarity_threshold=0.6
    )
    print(f"Labels: {sum(labels)} similar, {len(labels)-sum(labels)} novel")

    # Train classifier
    novelty_classifier = train_attribute_classifier(
        model=model,
        classifier=novelty_classifier,
        smiles_list=sample_smiles,
        labels=labels,
        stoi=stoi,
        epochs=5,
        batch_size=32,
        device=device
    )

    # Create PPLM generator targeting class 0 (novel)
    pplm = PPLMGenerator(
        model=model,
        classifier=novelty_classifier,
        stepsize=0.02,
        num_iterations=5,
        gm_scale=0.8,
        device=device
    )

    # Generate with novelty guidance
    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    context = "C"
    x = torch.tensor(
        [[stoi.get(s, 0) for s in regex.findall(context)]],
        dtype=torch.long, device=device
    ).repeat(30, 1)

    print("\nGenerating with novelty guidance (target: novel molecules)...")
    generated, _ = pplm.generate_with_pplm(
        input_ids=x,
        max_length=72,
        target_class=0,  # 0 = novel
        temperature=0.9,
        top_k=20
    )

    # Evaluate novelty
    from rdkit import Chem
    print("\nNovelty-Guided Results:")
    print("-" * 60)

    valid_count = 0
    novel_count = 0
    similarities = []

    for i, gen in enumerate(generated):
        smiles = ''.join([itos.get(int(t), '') for t in gen])
        smiles = smiles.replace('<', '').strip()

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            valid_count += 1
            max_sim = compute_max_similarity(smiles, train_smiles[:200])
            similarities.append(max_sim)
            if max_sim < 0.6:
                novel_count += 1
            if i < 5:
                print(f"{i+1}. {smiles[:40]}... MaxSim={max_sim:.3f}")

    print("-" * 60)
    print(f"Valid: {valid_count}/30, Novel (sim<0.6): {novel_count}/{valid_count}")
    if similarities:
        import numpy as np
        print(f"Similarity: mean={np.mean(similarities):.3f}, min={np.min(similarities):.3f}")

    return similarities


def demo_fragment_constraint(model, stoi, device):
    """Demo: Generate molecules with specific fragments."""
    print("\n" + "="*60)
    print("MODE 2: FRAGMENT CONSTRAINT")
    print("Goal: Generate molecules containing specific substructures")
    print("="*60)

    from rdkit import Chem

    # Create fragment-constrained generator
    frag_gen = FragmentConstrainedGenerator(
        model=model,
        stoi=stoi,
        device=device
    )

    # Test fragments
    fragments = [
        ("c1ccccc1", "Benzene ring"),
        ("C(=O)O", "Carboxylic acid"),
        ("N", "Nitrogen atom"),
    ]

    for fragment, name in fragments:
        print(f"\n--- Fragment: {name} ({fragment}) ---")

        # Method 1: Prefix generation
        print("\n[Prefix Mode] Starting with fragment:")
        prefix_results = frag_gen.generate_with_prefix(
            fragment=fragment,
            num_samples=10,
            max_length=72,
            temperature=0.8,
            top_k=20
        )

        valid_prefix = 0
        contains_prefix = 0
        for i, smi in enumerate(prefix_results[:5]):
            mol = Chem.MolFromSmiles(smi)
            is_valid = mol is not None
            has_frag = frag_gen.contains_fragment(smi, fragment) if is_valid else False
            if is_valid:
                valid_prefix += 1
            if has_frag:
                contains_prefix += 1
            status = "Valid+Frag" if has_frag else ("Valid" if is_valid else "Invalid")
            print(f"  {i+1}. {smi[:45]}... [{status}]")

        print(f"  Prefix: Valid={valid_prefix}/10, Contains={contains_prefix}/10")

        # Method 2: Constraint generation (rejection sampling)
        print("\n[Constraint Mode] Rejection sampling:")
        constraint_results = frag_gen.generate_with_constraint(
            fragment=fragment,
            num_samples=5,
            max_length=72,
            temperature=0.8,
            top_k=20,
            max_attempts=200
        )

        for i, smi in enumerate(constraint_results[:5]):
            mol = Chem.MolFromSmiles(smi)
            is_valid = mol is not None
            status = "Valid" if is_valid else "Invalid"
            print(f"  {i+1}. {smi[:45]}... [{status}]")

        print(f"  Constraint: Found {len(constraint_results)} molecules with fragment")


def main():
    """Main demo function."""
    # Paths
    data_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/datasets"
    weights_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/cond_gpt/weights"

    # Load vocabulary
    stoi_path = "../moses2_stoi.json"
    with open(stoi_path, 'r') as f:
        stoi = json.load(f)

    # Load training data
    print("Loading training data...")
    import pandas as pd
    data_file = data_dir + "/moses2.csv"
    df = pd.read_csv(data_file)
    train_smiles = df['smiles'].tolist()[:1000]
    print(f"Loaded {len(train_smiles)} molecules")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights_path = weights_dir + "/standard_gpt.pt"
    model, config = load_model(weights_path, vocab_size=len(stoi))
    model = model.to(device)

    # Demo 1: Novelty Guidance
    demo_novelty_guidance(model, config, stoi, train_smiles, device)

    # Demo 2: Fragment Constraint
    demo_fragment_constraint(model, stoi, device)

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
