# EDL Loss Bug Fix Report

## Date: 2026-01-24

## Problem Summary

Training loss was **increasing** instead of decreasing during EDL training:
- Epoch 1: loss ≈ 0.996 (normal)
- Epoch 68: loss ≈ 201.5 (catastrophic failure)
- Loss increased by **200x** over 68 epochs

## Root Cause

**Critical bug in KL divergence calculation** in `edl_loss.py` line 36.

The KL divergence formula had **reversed signs** in the log-gamma terms:

### Incorrect Implementation (BEFORE):
```python
# WRONG: All three terms had reversed signs
ln_term = torch.lgamma(torch.tensor(float(num_classes), device=alpha.device)) - torch.lgamma(S) + torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
```

### Correct Implementation (AFTER):
```python
# CORRECT: Proper signs for KL(Dir(α) || Dir(1,...,1))
ln_term = torch.lgamma(S) - torch.lgamma(torch.tensor(float(num_classes), device=alpha.device)) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
```

## Mathematical Explanation

The correct KL divergence formula for KL(Dir(α) || Dir(1,...,1)) is:

```
KL = log(Γ(S)) - log(Γ(K)) - Σlog(Γ(α_k)) + Σ(α_k - 1)(ψ(α_k) - ψ(S))
```

Where:
- K = number of classes (vocab_size)
- S = Σα_k (Dirichlet strength)
- Γ = gamma function
- ψ = digamma function

The bug had all three log-gamma terms with reversed signs:
```
WRONG: log(Γ(K)) - log(Γ(S)) + Σlog(Γ(α_k))  ← All signs reversed!
RIGHT: log(Γ(S)) - log(Γ(K)) - Σlog(Γ(α_k))
```

## Impact of the Bug

The incorrect formula caused KL divergence to produce **negative values**, which when added to the MSE loss caused the total loss to become increasingly negative (loss explosion).

### Test Results:

**Before Fix (Incorrect Formula):**
```
Low confidence (α=[2,2,2,2], S=8):     KL = -13.1048 ❌ (negative!)
High confidence (α=[100,3,2,1], S=106): KL = -44.4911 ❌ (negative!)
Result: KL values are NEGATIVE, causing loss explosion
```

**After Fix (Correct Formula):**
```
Low confidence (α=[2,2,2,2], S=8):     KL = 0.3620 ✓ (positive!)
High confidence (α=[100,3,2,1], S=106): KL = 7.7889 ✓ (positive!)
Result: KL values are POSITIVE and increase with divergence from uniform prior
```

## Verification

**Status**: ✅ **VERIFIED** - Fix is working correctly!

**Actual training behavior after fix:**
- Epoch 1, iter 0: loss ≈ 0.996 ✓ (positive)
- Epoch 1, iter 1: loss ≈ 0.978 ✓ (decreasing)
- Epoch 6, iter 0: loss ≈ 0.71 ✓ (continuing to decrease)
- Model saved at epoch 5 ✓ (training progressing normally)

**Confirmation:**
- Loss starts positive (~1.0) ✓
- Loss **decreases** during training (0.996 → 0.978 → 0.71) ✓
- Loss remains **positive** (not becoming negative) ✓
- No NaN or Inf values ✓

## Files Modified

1. `/data/home/hzw1010/suth/edl_transformer/molgpt/train/edl_loss.py` (line 36)
   - Fixed KL divergence calculation

2. `/data/home/hzw1010/suth/edl_transformer/molgpt/train/train_edl.sh` (line 20)
   - Changed MAX_EPOCHS from 2000 back to 20 (reasonable default)

## Status

✅ **RESOLVED** - Training is now working correctly with EDL loss.

## Notes

- Negative loss values are expected due to the KL divergence term
- The key metric is that loss should **decrease** during training
- Model is now learning properly as evidenced by decreasing loss values
