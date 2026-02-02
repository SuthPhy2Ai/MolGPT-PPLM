#!/usr/bin/env python3
"""
Complete experiments for paper submission.
Includes: scaled experiments, baseline comparisons, diversity/novelty metrics.
"""

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
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs

from model import GPT, GPTConfig
from pplm import (
    MultiObjectiveClassifier,
    MultiObjectivePPLM,
    ActiveLearningPPLM,
    compute_pareto_front,
    compute_molecular_properties,
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model_and_vocab():
    """Load pretrained model and vocabulary."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "cond_gpt/weights")

    with open(os.path.join(base_dir, "moses2_stoi_full.json"), 'r') as f:
        stoi = json.load(f)
    itos = {v: k for k, v in stoi.items()}

    config = GPTConfig(
        vocab_size=len(stoi), block_size=71,
        n_layer=8, n_head=8, n_embd=256,
        num_props=0, scaffold=False, scaffold_maxlen=0, lstm=False
    )

    model = GPT(config)
    weights_path = os.path.join(weights_dir, "fullvocab_gpt.pt")
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    return model, config, stoi, itos


def load_train_data(n_samples=10000):
    """Load training SMILES for reference."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "datasets/moses2.csv")
    df = pd.read_csv(data_path)
    train_smiles = df[df['split'] == 'train']['smiles'].tolist()[:n_samples]
    return train_smiles


def compute_fingerprint(smiles):
    """Compute Morgan fingerprint for a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def compute_tanimoto_similarity(fp1, fp2):
    """Compute Tanimoto similarity between two fingerprints."""
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_metrics(smiles_list, train_smiles_set, train_fps):
    """Compute comprehensive metrics for generated molecules."""
    valid_smiles = []
    valid_fps = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
            valid_fps.append(compute_fingerprint(smi))

    n_total = len(smiles_list)
    n_valid = len(valid_smiles)

    # Validity
    validity = n_valid / n_total if n_total > 0 else 0

    # Uniqueness
    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / n_valid if n_valid > 0 else 0

    # Novelty (not in training set)
    novel_smiles = [s for s in unique_smiles if s not in train_smiles_set]
    novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0

    # Internal Diversity
    internal_div = 0.0
    if len(valid_fps) > 1:
        sims = []
        for i in range(min(100, len(valid_fps))):
            for j in range(i+1, min(100, len(valid_fps))):
                if valid_fps[i] and valid_fps[j]:
                    sims.append(compute_tanimoto_similarity(valid_fps[i], valid_fps[j]))
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
        'n_total': n_total,
        'n_valid': n_valid,
        'n_unique': len(unique_smiles),
        'n_novel': len(novel_smiles),
        'logp_mean': np.mean(logp_values) if logp_values else 0,
        'logp_std': np.std(logp_values) if logp_values else 0,
        'qed_mean': np.mean(qed_values) if qed_values else 0,
        'qed_std': np.std(qed_values) if qed_values else 0,
        'logp_above_2.5': sum(1 for v in logp_values if v > 2.5) / len(logp_values) if logp_values else 0,
        'qed_above_0.6': sum(1 for v in qed_values if v > 0.6) / len(qed_values) if qed_values else 0,
    }


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_baseline(model, stoi, itos, device, n_samples=200, temperature=0.9):
    """Generate molecules without PPLM (baseline)."""
    import re
    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    model.eval()
    start_token = stoi.get('C', 0)
    x = torch.tensor([[start_token]], dtype=torch.long, device=device)
    x = x.repeat(n_samples, 1)

    with torch.no_grad():
        for _ in range(70):
            logits, _, _ = model(x)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    smiles_list = []
    for i in range(n_samples):
        smi = ''.join([itos.get(int(t), '') for t in x[i]])
        smi = smi.replace('<', '').strip()
        smiles_list.append(smi)

    return smiles_list


# ============================================================================
# EXPERIMENT 1: SCALED ACTIVE LEARNING
# ============================================================================

def run_scaled_experiment(model, config, stoi, train_smiles, device,
                          n_iterations=10, n_generate=200, n_runs=3):
    """Run scaled active learning experiment with multiple runs."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: SCALED ACTIVE LEARNING")
    print(f"Config: {n_iterations} iterations, {n_generate} molecules/iter, {n_runs} runs")
    print("="*70)

    all_results = []
    property_names = ['LogP', 'QED']
    thresholds = {'LogP': 2.5, 'QED': 0.6}

    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")

        mo_classifier = MultiObjectiveClassifier(
            hidden_size=config.n_embd,
            property_names=property_names,
            hidden_dim=128
        ).to(device)

        al_pplm = ActiveLearningPPLM(
            model=model,
            classifier=mo_classifier,
            stoi=stoi,
            property_names=property_names,
            device=device
        )

        results = al_pplm.run_active_learning_loop(
            initial_smiles=train_smiles[:200],
            thresholds=thresholds,
            n_iterations=n_iterations,
            n_generate=n_generate,
            n_select=30,
            acquisition='uncertainty',
            target_classes={'LogP': 1, 'QED': 1},
            temperature=0.85,
            top_k=30,
            verbose=True
        )

        all_results.append(results)

    return all_results


# ============================================================================
# EXPERIMENT 2: BASELINE COMPARISON
# ============================================================================

def run_baseline_comparison(model, config, stoi, itos, train_smiles, device,
                            train_smiles_set, train_fps, n_samples=1000):
    """Compare different generation methods."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: BASELINE COMPARISON")
    print("="*70)

    results = {}

    # 1. Random generation (no PPLM)
    print("\n[1/3] Random Generation (No PPLM)...")
    random_smiles = generate_baseline(model, stoi, itos, device, n_samples)
    results['random'] = compute_metrics(random_smiles, train_smiles_set, train_fps)
    results['random']['smiles'] = random_smiles
    print(f"  Validity: {results['random']['validity']*100:.1f}%")
    print(f"  LogP>2.5: {results['random']['logp_above_2.5']*100:.1f}%")

    # 2. Single-objective PPLM (LogP only)
    print("\n[2/3] Single-Objective PPLM (LogP)...")
    single_smiles = run_single_objective_pplm(
        model, config, stoi, itos, train_smiles, device, n_samples, target='LogP'
    )
    results['single_logp'] = compute_metrics(single_smiles, train_smiles_set, train_fps)
    results['single_logp']['smiles'] = single_smiles
    print(f"  Validity: {results['single_logp']['validity']*100:.1f}%")
    print(f"  LogP>2.5: {results['single_logp']['logp_above_2.5']*100:.1f}%")

    # 3. Multi-objective PPLM (LogP + QED)
    print("\n[3/3] Multi-Objective PPLM (LogP + QED)...")
    multi_smiles = run_multi_objective_pplm(
        model, config, stoi, itos, train_smiles, device, n_samples
    )
    results['multi_obj'] = compute_metrics(multi_smiles, train_smiles_set, train_fps)
    results['multi_obj']['smiles'] = multi_smiles
    print(f"  Validity: {results['multi_obj']['validity']*100:.1f}%")
    print(f"  LogP>2.5: {results['multi_obj']['logp_above_2.5']*100:.1f}%")

    return results


def run_single_objective_pplm(model, config, stoi, itos, train_smiles, device,
                               n_samples, target='LogP'):
    """Run single-objective PPLM."""
    import re
    from pplm import AttributeClassifier, PPLMGenerator

    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    # Create and train classifier
    classifier = AttributeClassifier(hidden_size=config.n_embd).to(device)

    # Get hidden states for training
    threshold = 2.5 if target == 'LogP' else 0.6
    sample_smiles = train_smiles[:500]
    labels = []
    hidden_list = []

    model.eval()
    with torch.no_grad():
        for smi in sample_smiles:
            props = compute_molecular_properties(smi)
            if target in props:
                labels.append(1 if props[target] > threshold else 0)
            else:
                labels.append(0)

            tokens = regex.findall(smi)
            ids = [stoi.get(t, 0) for t in tokens]
            if len(ids) == 0:
                ids = [0]
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

    # Train classifier
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    classifier.train()
    for _ in range(20):
        optimizer.zero_grad()
        logits = classifier(hidden)
        loss = torch.nn.functional.cross_entropy(logits, labels_t)
        loss.backward()
        optimizer.step()
    classifier.eval()

    # Generate with PPLM
    pplm = PPLMGenerator(
        model=model, classifier=classifier,
        stepsize=0.02, num_iterations=3, gm_scale=0.7, device=device
    )

    start_token = stoi.get('C', 0)
    x = torch.tensor([[start_token]], dtype=torch.long, device=device)
    x = x.repeat(n_samples, 1)

    generated, _ = pplm.generate_with_pplm(
        x, max_length=72, target_class=1, temperature=0.85, top_k=30
    )

    smiles_list = []
    for i in range(n_samples):
        smi = ''.join([itos.get(int(t), '') for t in generated[i]])
        smi = smi.replace('<', '').strip()
        smiles_list.append(smi)

    return smiles_list


def run_multi_objective_pplm(model, config, stoi, itos, train_smiles, device, n_samples):
    """Run multi-objective PPLM (LogP + QED)."""
    import re
    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    property_names = ['LogP', 'QED']
    thresholds = {'LogP': 2.5, 'QED': 0.6}

    mo_classifier = MultiObjectiveClassifier(
        hidden_size=config.n_embd,
        property_names=property_names,
        hidden_dim=128
    ).to(device)

    # Train classifier
    sample_smiles = train_smiles[:500]
    labels = {name: [] for name in property_names}
    hidden_list = []

    model.eval()
    with torch.no_grad():
        for smi in sample_smiles:
            props = compute_molecular_properties(smi)
            for name in property_names:
                if name in props:
                    labels[name].append(1 if props[name] > thresholds[name] else 0)
                else:
                    labels[name].append(0)

            tokens = regex.findall(smi)
            ids = [stoi.get(t, 0) for t in tokens]
            if len(ids) == 0:
                ids = [0]
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

    # Train
    optimizer = torch.optim.Adam(mo_classifier.parameters(), lr=1e-3)
    mo_classifier.train()
    for _ in range(20):
        optimizer.zero_grad()
        logits_dict = mo_classifier(hidden)
        total_loss = 0
        for name in property_names:
            target = torch.tensor(labels[name], dtype=torch.long, device=device)
            total_loss += torch.nn.functional.cross_entropy(logits_dict[name], target)
        total_loss.backward()
        optimizer.step()
    mo_classifier.eval()

    # Generate
    mo_pplm = MultiObjectivePPLM(
        model=model, classifier=mo_classifier,
        stepsize=0.02, num_iterations=3, gm_scale=0.7, device=device
    )

    start_token = stoi.get('C', 0)
    x = torch.tensor([[start_token]], dtype=torch.long, device=device)
    x = x.repeat(n_samples, 1)

    generated, _ = mo_pplm.generate_pareto_optimal(
        x, max_length=72,
        target_classes={'LogP': 1, 'QED': 1},
        temperature=0.85, top_k=30
    )

    smiles_list = []
    for i in range(n_samples):
        smi = ''.join([itos.get(int(t), '') for t in generated[i]])
        smi = smi.replace('<', '').strip()
        smiles_list.append(smi)

    return smiles_list


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run all experiments."""
    print("="*70)
    print("COMPLETE EXPERIMENTS FOR PAPER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\nLoading model and data...")
    model, config, stoi, itos = load_model_and_vocab()
    model = model.to(device)

    train_smiles = load_train_data(n_samples=10000)
    train_smiles_set = set(train_smiles)
    print(f"Training data: {len(train_smiles)} molecules")

    print("Computing training fingerprints...")
    train_fps = [compute_fingerprint(s) for s in tqdm(train_smiles[:1000])]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    all_results = {}

    # Experiment 1: Scaled Active Learning
    exp1_results = run_scaled_experiment(
        model, config, stoi, train_smiles, device,
        n_iterations=10, n_generate=200, n_runs=3
    )
    all_results['exp1_scaled_al'] = exp1_results
    with open(f"{results_dir}/exp1_scaled_al_{timestamp}.pkl", 'wb') as f:
        pickle.dump(exp1_results, f)
    print(f"Saved: exp1_scaled_al_{timestamp}.pkl")

    # Experiment 2: Baseline Comparison
    exp2_results = run_baseline_comparison(
        model, config, stoi, itos, train_smiles, device,
        train_smiles_set, train_fps, n_samples=1000
    )
    all_results['exp2_baseline'] = exp2_results
    with open(f"{results_dir}/exp2_baseline_{timestamp}.pkl", 'wb') as f:
        pickle.dump(exp2_results, f)
    print(f"Saved: exp2_baseline_{timestamp}.pkl")

    # Print summary
    print_summary(all_results)

    # Save all results
    with open(f"{results_dir}/all_results_{timestamp}.pkl", 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def print_summary(all_results):
    """Print experiment summary."""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    if 'exp2_baseline' in all_results:
        print("\nBaseline Comparison:")
        print("-"*50)
        for method, metrics in all_results['exp2_baseline'].items():
            if method != 'smiles' and isinstance(metrics, dict):
                print(f"\n{method}:")
                print(f"  Validity:   {metrics.get('validity', 0)*100:.1f}%")
                print(f"  Uniqueness: {metrics.get('uniqueness', 0)*100:.1f}%")
                print(f"  Novelty:    {metrics.get('novelty', 0)*100:.1f}%")
                print(f"  Diversity:  {metrics.get('internal_diversity', 0)*100:.1f}%")
                print(f"  LogP>2.5:   {metrics.get('logp_above_2.5', 0)*100:.1f}%")
                print(f"  QED>0.6:    {metrics.get('qed_above_0.6', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
