#!/usr/bin/env python3
"""Run remaining experiments (Exp2 baseline comparison)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/train')

import torch
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from model import GPT, GPTConfig
from pplm import (
    AttributeClassifier,
    PPLMGenerator,
    MultiObjectiveClassifier,
    MultiObjectivePPLM,
    compute_molecular_properties,
)


def load_model_and_vocab():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base_dir, "moses2_stoi_full.json"), 'r') as f:
        stoi = json.load(f)
    itos = {v: k for k, v in stoi.items()}

    config = GPTConfig(
        vocab_size=len(stoi), block_size=71,
        n_layer=8, n_head=8, n_embd=256,
        num_props=0, scaffold=False, scaffold_maxlen=0, lstm=False
    )
    model = GPT(config)
    weights_path = os.path.join(base_dir, "cond_gpt/weights/fullvocab_gpt.pt")
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return model, config, stoi, itos


def compute_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def compute_metrics(smiles_list, train_smiles_set):
    """Compute comprehensive metrics."""
    valid_smiles = []
    valid_fps = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
            valid_fps.append(compute_fingerprint(smi))

    n_total = len(smiles_list)
    n_valid = len(valid_smiles)
    validity = n_valid / n_total if n_total > 0 else 0

    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / n_valid if n_valid > 0 else 0

    novel_smiles = [s for s in unique_smiles if s not in train_smiles_set]
    novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0

    # Internal diversity
    internal_div = 0.0
    if len(valid_fps) > 1:
        sims = []
        for i in range(min(100, len(valid_fps))):
            for j in range(i+1, min(100, len(valid_fps))):
                if valid_fps[i] and valid_fps[j]:
                    sims.append(DataStructs.TanimotoSimilarity(valid_fps[i], valid_fps[j]))
        internal_div = 1 - np.mean(sims) if sims else 0

    # Properties
    logp_values, qed_values = [], []
    for smi in valid_smiles:
        props = compute_molecular_properties(smi)
        if 'LogP' in props:
            logp_values.append(props['LogP'])
        if 'QED' in props:
            qed_values.append(props['QED'])

    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'internal_diversity': internal_div,
        'n_valid': n_valid,
        'logp_mean': np.mean(logp_values) if logp_values else 0,
        'logp_std': np.std(logp_values) if logp_values else 0,
        'qed_mean': np.mean(qed_values) if qed_values else 0,
        'qed_std': np.std(qed_values) if qed_values else 0,
        'logp_above_2.5': sum(1 for v in logp_values if v > 2.5) / len(logp_values) if logp_values else 0,
        'qed_above_0.6': sum(1 for v in qed_values if v > 0.6) / len(qed_values) if qed_values else 0,
    }


def generate_baseline(model, stoi, itos, device, n_samples=500):
    """Generate without PPLM."""
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

    smiles_list = []
    for i in range(n_samples):
        smi = ''.join([itos.get(int(t), '') for t in x[i]])
        smi = smi.replace('<', '').strip()
        smiles_list.append(smi)
    return smiles_list


def run_single_pplm(model, config, stoi, itos, train_smiles, device, n_samples=500):
    """Single-objective PPLM for LogP."""
    import re
    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    classifier = AttributeClassifier(hidden_size=config.n_embd).to(device)

    # Train classifier
    sample_smiles = train_smiles[:500]
    labels, hidden_list = [], []

    model.eval()
    with torch.no_grad():
        for smi in sample_smiles:
            props = compute_molecular_properties(smi)
            labels.append(1 if props.get('LogP', 0) > 2.5 else 0)

            tokens = regex.findall(smi)
            ids = [stoi.get(t, 0) for t in tokens] or [0]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            b, t = x.size()
            tok_emb = model.tok_emb(x)
            pos_emb = model.pos_emb[:, :t, :]
            type_emb = model.type_emb(torch.ones((b, t), dtype=torch.long, device=device))
            h = model.drop(tok_emb + pos_emb + type_emb)
            for layer in model.blocks:
                h, _ = layer(h)
            hidden_list.append(h.mean(dim=1).squeeze(0))

    hidden = torch.stack(hidden_list)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    classifier.train()
    for _ in range(20):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(classifier(hidden), labels_t)
        loss.backward()
        optimizer.step()
    classifier.eval()

    # Generate
    pplm = PPLMGenerator(model=model, classifier=classifier,
                         stepsize=0.02, num_iterations=3, gm_scale=0.7, device=device)

    start_token = stoi.get('C', 0)
    x = torch.tensor([[start_token]], dtype=torch.long, device=device).repeat(n_samples, 1)
    generated, _ = pplm.generate_with_pplm(x, max_length=72, target_class=1,
                                            temperature=0.85, top_k=30)

    smiles_list = []
    for i in range(n_samples):
        smi = ''.join([itos.get(int(t), '') for t in generated[i]]).replace('<', '').strip()
        smiles_list.append(smi)
    return smiles_list


def run_multi_pplm(model, config, stoi, itos, train_smiles, device, n_samples=500):
    """Multi-objective PPLM for LogP + QED."""
    import re
    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    property_names = ['LogP', 'QED']
    thresholds = {'LogP': 2.5, 'QED': 0.6}

    mo_classifier = MultiObjectiveClassifier(
        hidden_size=config.n_embd, property_names=property_names, hidden_dim=128
    ).to(device)

    sample_smiles = train_smiles[:500]
    labels = {name: [] for name in property_names}
    hidden_list = []

    model.eval()
    with torch.no_grad():
        for smi in sample_smiles:
            props = compute_molecular_properties(smi)
            for name in property_names:
                labels[name].append(1 if props.get(name, 0) > thresholds[name] else 0)

            tokens = regex.findall(smi)
            ids = [stoi.get(t, 0) for t in tokens] or [0]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            b, t = x.size()
            tok_emb = model.tok_emb(x)
            pos_emb = model.pos_emb[:, :t, :]
            type_emb = model.type_emb(torch.ones((b, t), dtype=torch.long, device=device))
            h = model.drop(tok_emb + pos_emb + type_emb)
            for layer in model.blocks:
                h, _ = layer(h)
            hidden_list.append(h.mean(dim=1).squeeze(0))

    hidden = torch.stack(hidden_list)

    optimizer = torch.optim.Adam(mo_classifier.parameters(), lr=1e-3)
    mo_classifier.train()
    for _ in range(20):
        optimizer.zero_grad()
        logits_dict = mo_classifier(hidden)
        total_loss = sum(
            torch.nn.functional.cross_entropy(
                logits_dict[name],
                torch.tensor(labels[name], dtype=torch.long, device=device)
            ) for name in property_names
        )
        total_loss.backward()
        optimizer.step()
    mo_classifier.eval()

    mo_pplm = MultiObjectivePPLM(
        model=model, classifier=mo_classifier,
        stepsize=0.02, num_iterations=3, gm_scale=0.7, device=device
    )

    start_token = stoi.get('C', 0)
    x = torch.tensor([[start_token]], dtype=torch.long, device=device).repeat(n_samples, 1)
    generated, _ = mo_pplm.generate_pareto_optimal(
        x, max_length=72, target_classes={'LogP': 1, 'QED': 1},
        temperature=0.85, top_k=30
    )

    smiles_list = []
    for i in range(n_samples):
        smi = ''.join([itos.get(int(t), '') for t in generated[i]]).replace('<', '').strip()
        smiles_list.append(smi)
    return smiles_list


def main():
    print("="*60)
    print("EXPERIMENT 2: BASELINE COMPARISON")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, config, stoi, itos = load_model_and_vocab()
    model = model.to(device)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_dir, "datasets/moses2.csv"))
    train_smiles = df[df['split'] == 'train']['smiles'].tolist()[:5000]
    train_smiles_set = set(train_smiles)

    results = {}
    n_samples = 500

    print("\n[1/3] Random Generation...")
    random_smi = generate_baseline(model, stoi, itos, device, n_samples)
    results['random'] = compute_metrics(random_smi, train_smiles_set)
    print(f"  Validity: {results['random']['validity']*100:.1f}%")
    print(f"  LogP>2.5: {results['random']['logp_above_2.5']*100:.1f}%")

    print("\n[2/3] Single-Objective PPLM (LogP)...")
    single_smi = run_single_pplm(model, config, stoi, itos, train_smiles, device, n_samples)
    results['single_logp'] = compute_metrics(single_smi, train_smiles_set)
    print(f"  Validity: {results['single_logp']['validity']*100:.1f}%")
    print(f"  LogP>2.5: {results['single_logp']['logp_above_2.5']*100:.1f}%")

    print("\n[3/3] Multi-Objective PPLM (LogP+QED)...")
    multi_smi = run_multi_pplm(model, config, stoi, itos, train_smiles, device, n_samples)
    results['multi_obj'] = compute_metrics(multi_smi, train_smiles_set)
    print(f"  Validity: {results['multi_obj']['validity']*100:.1f}%")
    print(f"  LogP>2.5: {results['multi_obj']['logp_above_2.5']*100:.1f}%")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"{results_dir}/exp2_baseline_{ts}.pkl", 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for m, r in results.items():
        print(f"\n{m}:")
        print(f"  Validity:   {r['validity']*100:.1f}%")
        print(f"  Uniqueness: {r['uniqueness']*100:.1f}%")
        print(f"  Novelty:    {r['novelty']*100:.1f}%")
        print(f"  Diversity:  {r['internal_diversity']*100:.1f}%")
        print(f"  LogP>2.5:   {r['logp_above_2.5']*100:.1f}%")
        print(f"  QED>0.6:    {r['qed_above_0.6']*100:.1f}%")


if __name__ == "__main__":
    main()
