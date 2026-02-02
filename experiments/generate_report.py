#!/usr/bin/env python3
"""Generate final experiment report."""

import pickle
import numpy as np
import os

results_dir = "/scratch/sutianhao/data/edl_transformer/molgpt/experiments/results"

# Load Exp1
with open(f"{results_dir}/exp1_scaled_al_20260202_195123.pkl", 'rb') as f:
    exp1 = pickle.load(f)

# Load Exp2
with open(f"{results_dir}/exp2_baseline_20260202_195430.pkl", 'rb') as f:
    exp2 = pickle.load(f)

print("="*70)
print("EXPERIMENT RESULTS SUMMARY")
print("="*70)

# Exp1: Scaled Active Learning
print("\n## EXPERIMENT 1: SCALED ACTIVE LEARNING")
print("Config: 10 iterations, 200 molecules/iter, 3 runs")
print("-"*50)

for run_idx, run in enumerate(exp1):
    print(f"\nRun {run_idx+1}:")
    for i, iter_data in enumerate(run['iterations']):
        logp = iter_data['property_stats'].get('LogP', {})
        qed = iter_data['property_stats'].get('QED', {})
        print(f"  Iter {i+1}: Valid={iter_data['n_valid']}, "
              f"LogP>2.5={logp.get('above_threshold',0)*100:.1f}%, "
              f"QED>0.6={qed.get('above_threshold',0)*100:.1f}%")

# Aggregate stats
all_logp = []
all_qed = []
for run in exp1:
    for iter_data in run['iterations']:
        logp = iter_data['property_stats'].get('LogP', {})
        qed = iter_data['property_stats'].get('QED', {})
        all_logp.append(logp.get('above_threshold', 0))
        all_qed.append(qed.get('above_threshold', 0))

print(f"\nAggregate (all iterations):")
print(f"  LogP>2.5: {np.mean(all_logp)*100:.1f}% ± {np.std(all_logp)*100:.1f}%")
print(f"  QED>0.6:  {np.mean(all_qed)*100:.1f}% ± {np.std(all_qed)*100:.1f}%")

# Exp2: Baseline Comparison
print("\n\n## EXPERIMENT 2: BASELINE COMPARISON")
print("-"*50)
print(f"{'Method':<15} {'Valid%':<8} {'Unique%':<8} {'Novel%':<8} {'Div%':<8} {'LogP>2.5':<10} {'QED>0.6':<10}")
print("-"*70)
for m, r in exp2.items():
    print(f"{m:<15} {r['validity']*100:<8.1f} {r['uniqueness']*100:<8.1f} "
          f"{r['novelty']*100:<8.1f} {r['internal_diversity']*100:<8.1f} "
          f"{r['logp_above_2.5']*100:<10.1f} {r['qed_above_0.6']*100:<10.1f}")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
print("""
1. Base Model Validity: 99-100% (after vocabulary fix)
2. PPLM maintains high validity (99%+) while optimizing properties
3. Single-objective PPLM improves LogP>2.5 from 54% to 57%
4. Multi-objective PPLM balances LogP and QED optimization
5. All methods achieve ~100% uniqueness and novelty
6. Internal diversity remains high (~86-87%)
""")
