"""
Uncertainty Estimation Methods for Autoregressive Models

Comparison of different uncertainty estimation approaches:
1. MC Dropout - Multiple forward passes with dropout enabled
2. Entropy-based - Use output entropy as uncertainty
3. Temperature Scaling - Calibrate softmax temperature
4. EDL - Evidential Deep Learning (for reference)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty estimation.

    Key idea: Run multiple forward passes with dropout enabled,
    use variance of predictions as uncertainty.
    """

    def __init__(self, model, n_samples: int = 10):
        self.model = model
        self.n_samples = n_samples

    def enable_dropout(self):
        """Enable dropout layers during inference"""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def estimate(self, x, prop=None, scaffold=None) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty using MC Dropout.

        Returns:
            mean_probs: Mean predicted probabilities [B, T, K]
            uncertainty: Predictive uncertainty [B, T]
            epistemic: Model uncertainty (variance) [B, T]
        """
        self.model.eval()
        self.enable_dropout()

        all_probs = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                logits, _, _ = self.model(x, targets=None, prop=prop, scaffold=scaffold)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        # Stack: [n_samples, B, T, K]
        all_probs = torch.stack(all_probs, dim=0)

        # Mean prediction
        mean_probs = all_probs.mean(dim=0)  # [B, T, K]

        # Epistemic uncertainty (variance across samples)
        variance = all_probs.var(dim=0)  # [B, T, K]
        epistemic = variance.sum(dim=-1)  # [B, T]

        # Total uncertainty (entropy of mean)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)

        return {
            'mean_probs': mean_probs,
            'uncertainty': entropy,
            'epistemic': epistemic,
            'method': 'mc_dropout'
        }


class EntropyUncertainty:
    """
    Entropy-based uncertainty estimation.

    Key idea: High entropy = high uncertainty
    Simple, no extra computation needed.
    """

    def __init__(self, model):
        self.model = model

    def estimate(self, x, prop=None, scaffold=None) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty using output entropy.
        """
        self.model.eval()

        with torch.no_grad():
            logits, _, _ = self.model(x, targets=None, prop=prop, scaffold=scaffold)
            probs = F.softmax(logits, dim=-1)

        # Entropy as uncertainty
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        # Max probability (confidence)
        max_prob, _ = probs.max(dim=-1)

        return {
            'probs': probs,
            'uncertainty': entropy,
            'confidence': max_prob,
            'method': 'entropy'
        }


class TemperatureScaling:
    """
    Temperature Scaling for calibrated uncertainty.

    Key idea: Learn optimal temperature to calibrate softmax.
    """

    def __init__(self, model, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature

    def estimate(self, x, prop=None, scaffold=None) -> Dict[str, torch.Tensor]:
        """Estimate with temperature-scaled softmax."""
        self.model.eval()

        with torch.no_grad():
            logits, _, _ = self.model(x, targets=None, prop=prop, scaffold=scaffold)
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=-1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_prob, _ = probs.max(dim=-1)

        return {
            'probs': probs,
            'uncertainty': entropy,
            'confidence': max_prob,
            'temperature': self.temperature,
            'method': 'temperature_scaling'
        }


class EDLUncertainty:
    """EDL-based uncertainty (wrapper for comparison)."""

    def __init__(self, model):
        self.model = model

    def estimate(self, x, prop=None, scaffold=None) -> Dict[str, torch.Tensor]:
        self.model.eval()

        with torch.no_grad():
            alpha, _, _ = self.model(x, targets=None, prop=prop, scaffold=scaffold)

        S = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / S
        K = alpha.size(-1)

        # Vacuity (epistemic)
        vacuity = K / S.squeeze(-1)
        # Dissonance
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        return {
            'probs': probs,
            'uncertainty': vacuity,
            'entropy': entropy,
            'alpha': alpha,
            'method': 'edl'
        }


def compare_methods(model, x, prop=None, scaffold=None, use_edl=False):
    """Compare all uncertainty methods on same input."""
    results = {}

    # Entropy (baseline)
    entropy_est = EntropyUncertainty(model)
    results['entropy'] = entropy_est.estimate(x, prop, scaffold)

    # MC Dropout
    mc_est = MCDropoutUncertainty(model, n_samples=10)
    results['mc_dropout'] = mc_est.estimate(x, prop, scaffold)

    # Temperature Scaling
    for temp in [0.5, 1.0, 2.0]:
        temp_est = TemperatureScaling(model, temperature=temp)
        results[f'temp_{temp}'] = temp_est.estimate(x, prop, scaffold)

    # EDL (if model supports)
    if use_edl:
        edl_est = EDLUncertainty(model)
        results['edl'] = edl_est.estimate(x, prop, scaffold)

    return results
