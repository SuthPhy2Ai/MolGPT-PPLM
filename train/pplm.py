"""
PPLM (Plug and Play Language Model) for Molecular Generation

Implementation of gradient-based guided decoding for controlling molecular properties.
Based on: "Plug and Play Language Models: A Simple Approach to Controlled Text Generation"
https://arxiv.org/abs/1912.02164

Key idea:
- Train a small attribute classifier on hidden states
- During generation, compute gradient of attribute w.r.t. hidden states
- Update hidden states to increase probability of desired attribute
- Generate with modified hidden states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class AttributeClassifier(nn.Module):
    """
    Small classifier that predicts molecular properties from GPT hidden states.

    This classifier is trained separately on hidden states extracted from the GPT model.
    During PPLM generation, we compute gradients through this classifier to guide generation.
    """

    def __init__(self, hidden_size: int, num_classes: int = 2, hidden_dim: int = 128):
        """
        Args:
            hidden_size: Dimension of GPT hidden states (n_embd)
            num_classes: Number of attribute classes (e.g., 2 for binary classification)
            hidden_dim: Hidden dimension of classifier MLP
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, H] or [B, H] hidden states from GPT

        Returns:
            logits: [B, T, num_classes] or [B, num_classes] classification logits
        """
        return self.classifier(hidden_states)


class PPLMGenerator:
    """
    PPLM-based guided generation for molecular GPT.

    Algorithm:
    1. Forward pass through GPT to get hidden states H
    2. Compute attribute gradient: grad = ∇_H log p(attribute | H)
    3. Update hidden states: H' = H + stepsize * grad
    4. Generate next token using modified hidden states
    5. Repeat for each generation step
    """

    def __init__(
        self,
        model: nn.Module,
        classifier: AttributeClassifier,
        stepsize: float = 0.03,
        num_iterations: int = 3,
        kl_scale: float = 0.01,
        gm_scale: float = 0.9,
        grad_length: int = 10000,
        device: str = 'cuda'
    ):
        """
        Args:
            model: GPT model for generation
            classifier: Trained attribute classifier
            stepsize: Step size for gradient update
            num_iterations: Number of gradient update iterations per token
            kl_scale: Weight for KL divergence loss (keeps generation fluent)
            gm_scale: Weight for gradient modification
            grad_length: Maximum sequence length for gradient computation
            device: Device to run on
        """
        self.model = model
        self.classifier = classifier
        self.stepsize = stepsize
        self.num_iterations = num_iterations
        self.kl_scale = kl_scale
        self.gm_scale = gm_scale
        self.grad_length = grad_length
        self.device = device

        self.model.to(device)
        self.classifier.to(device)

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        prop: Optional[torch.Tensor] = None,
        scaffold: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract hidden states from GPT model.

        Returns:
            hidden_states: [B, T, H] hidden states before final layer norm
            logits: [B, T, V] output logits
        """
        self.model.eval()

        b, t = input_ids.size()
        config = self.model.config

        # Get embeddings
        token_embeddings = self.model.tok_emb(input_ids)
        position_embeddings = self.model.pos_emb[:, :t, :]
        type_embeddings = self.model.type_emb(
            torch.ones((b, t), dtype=torch.long, device=input_ids.device)
        )
        x = self.model.drop(token_embeddings + position_embeddings + type_embeddings)

        # Handle conditional inputs
        if config.num_props and prop is not None:
            type_embd = self.model.type_emb(
                torch.zeros((b, 1), dtype=torch.long, device=input_ids.device)
            )
            if prop.ndim == 2:
                p = self.model.prop_nn(prop.unsqueeze(1))
            else:
                p = self.model.prop_nn(prop)
            p += type_embd
            x = torch.cat([p, x], 1)

        if config.scaffold and scaffold is not None:
            type_embd = self.model.type_emb(
                torch.zeros((b, 1), dtype=torch.long, device=input_ids.device)
            )
            scaffold_embeds = self.model.tok_emb(scaffold)
            if config.lstm:
                scaffold_embeds = self.model.lstm(scaffold_embeds.permute(1, 0, 2))[1][0]
                scaffold_embeds = scaffold_embeds.permute(1, 0, 2)
            scaffold_embeds += type_embd
            x = torch.cat([scaffold_embeds, x], 1)

        # Forward through transformer blocks
        for layer in self.model.blocks:
            x, _ = layer(x)

        hidden_states = x  # Before final layer norm

        # Get logits
        x = self.model.ln_f(x)
        logits = self.model.head(x)

        # Crop conditional tokens
        if config.num_props and config.scaffold:
            num = int(bool(config.num_props)) + int(config.scaffold_maxlen)
        elif config.num_props:
            num = int(bool(config.num_props))
        elif config.scaffold:
            num = int(config.scaffold_maxlen)
        else:
            num = 0

        hidden_states = hidden_states[:, num:, :]
        logits = logits[:, num:, :]

        return hidden_states, logits

    def perturb_hidden(
        self,
        hidden_states: torch.Tensor,
        original_logits: torch.Tensor,
        target_class: int = 1
    ) -> torch.Tensor:
        """
        Perturb hidden states using gradient from attribute classifier.

        Args:
            hidden_states: [B, T, H] hidden states to perturb
            original_logits: [B, T, V] original logits for KL constraint
            target_class: Target attribute class to optimize for

        Returns:
            perturbed_hidden: [B, T, H] perturbed hidden states
        """
        # Clone and enable gradients
        perturbed = hidden_states.clone().detach()
        perturbed.requires_grad_(True)

        # Accumulate gradients over iterations
        grad_accumulator = torch.zeros_like(perturbed)

        for _ in range(self.num_iterations):
            if perturbed.grad is not None:
                perturbed.grad.zero_()

            # Get classifier prediction on last token (or mean pooled)
            # Use mean pooling for sequence-level attribute
            pooled = perturbed.mean(dim=1)  # [B, H]
            attr_logits = self.classifier(pooled)  # [B, num_classes]

            # Compute loss: maximize log p(target_class | hidden)
            attr_loss = -F.log_softmax(attr_logits, dim=-1)[:, target_class].mean()

            # Compute gradient
            attr_loss.backward(retain_graph=True)

            if perturbed.grad is not None:
                # Accumulate normalized gradient
                grad = perturbed.grad.clone()
                grad_norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
                grad = grad / grad_norm
                grad_accumulator += grad

        # Apply perturbation
        perturbed_hidden = hidden_states + self.stepsize * self.gm_scale * grad_accumulator

        return perturbed_hidden.detach()

    def generate_with_pplm(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        target_class: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        prop: Optional[torch.Tensor] = None,
        scaffold: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Generate molecules with PPLM guidance.

        Args:
            input_ids: [B, T] input token ids
            max_length: Maximum generation length
            target_class: Target attribute class
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = no top-k)
            prop: Optional property conditioning
            scaffold: Optional scaffold conditioning
            eos_token_id: End of sequence token id

        Returns:
            generated_ids: [B, max_length] generated token ids
            uncertainties: List of uncertainty scores per step
        """
        self.model.eval()
        self.classifier.eval()

        device = input_ids.device
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        uncertainties = []

        for step in range(max_length - input_ids.size(1)):
            # Get hidden states and logits
            hidden_states, logits = self.get_hidden_states(generated, prop, scaffold)

            # Perturb hidden states using classifier gradient
            perturbed_hidden = self.perturb_hidden(
                hidden_states, logits, target_class
            )

            # Get new logits from perturbed hidden states
            perturbed_logits = self._hidden_to_logits(perturbed_hidden)

            # Get logits for next token (last position)
            next_logits = perturbed_logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                next_logits = self._top_k_filtering(next_logits, top_k)

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Compute uncertainty (entropy)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            uncertainties.append(entropy.mean().item())

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        return generated, uncertainties

    def _hidden_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convert hidden states to logits using model's head."""
        x = self.model.ln_f(hidden_states)
        logits = self.model.head(x)
        return logits

    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """Filter logits to keep only top-k tokens."""
        if top_k > 0:
            # Ensure top_k doesn't exceed vocab size
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_values,
                torch.full_like(logits, float('-inf')),
                logits
            )
        return logits


def train_attribute_classifier(
    model: nn.Module,
    classifier: AttributeClassifier,
    smiles_list: List[str],
    labels: List[int],
    stoi: Dict[str, int],
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cuda'
) -> AttributeClassifier:
    """
    Train attribute classifier on hidden states from GPT model.
    """
    import re
    from tqdm import tqdm

    pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    model.to(device)
    model.eval()
    classifier.to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Extract hidden states
    print("Extracting hidden states...")
    all_hidden = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size)):
            batch_smiles = smiles_list[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            # Tokenize
            batch_tokens = []
            valid_labels = []
            max_len = 0

            for smi, lbl in zip(batch_smiles, batch_labels):
                tokens = regex.findall(smi)
                if len(tokens) > 0:
                    ids = [stoi.get(t, 0) for t in tokens]
                    batch_tokens.append(ids)
                    valid_labels.append(lbl)
                    max_len = max(max_len, len(ids))

            if len(batch_tokens) == 0:
                continue

            # Pad sequences
            padded = []
            for ids in batch_tokens:
                padded.append(ids + [0] * (max_len - len(ids)))

            x = torch.tensor(padded, dtype=torch.long, device=device)

            # Get hidden states using a temporary generator
            pplm_gen = PPLMGenerator(model, classifier, device=device)
            hidden, _ = pplm_gen.get_hidden_states(x)

            # Mean pool over sequence
            pooled = hidden.mean(dim=1)
            all_hidden.append(pooled.cpu())
            all_labels.extend(valid_labels)

    # Concatenate all hidden states
    all_hidden = torch.cat(all_hidden, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    print(f"Training classifier on {len(all_labels)} samples...")

    # Training loop
    dataset = torch.utils.data.TensorDataset(all_hidden, all_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for hidden_batch, label_batch in loader:
            hidden_batch = hidden_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = classifier(hidden_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == label_batch).sum().item()
            total += label_batch.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/len(loader):.4f}, Acc={acc:.1f}%")

    return classifier


def compute_molecular_properties(smiles: str) -> Dict[str, float]:
    """
    Compute molecular properties for PPLM attribute classification.

    Returns dict with: MW, LogP, TPSA, QED, num_rings, etc.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED as QEDModule
    except ImportError:
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'QED': QEDModule.qed(mol),
        'NumRings': Descriptors.RingCount(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol)
    }


def create_binary_labels(
    smiles_list: List[str],
    property_name: str = 'LogP',
    threshold: float = None
) -> List[int]:
    """
    Create binary labels based on molecular property threshold.

    Args:
        smiles_list: List of SMILES
        property_name: Property to use (LogP, QED, MW, etc.)
        threshold: Threshold for binary classification (default: median)

    Returns:
        List of binary labels (0 or 1)
    """
    properties = []
    for smi in smiles_list:
        props = compute_molecular_properties(smi)
        if property_name in props:
            properties.append(props[property_name])
        else:
            properties.append(None)

    # Filter valid properties
    valid_props = [p for p in properties if p is not None]

    if threshold is None:
        threshold = np.median(valid_props)

    labels = []
    for p in properties:
        if p is None:
            labels.append(0)
        else:
            labels.append(1 if p > threshold else 0)

    return labels


# ============================================================================
# NOVELTY GUIDANCE: Generate molecules different from training set
# ============================================================================

class NoveltyClassifier(nn.Module):
    """
    Classifier that predicts whether a molecule is similar to training set.
    Used for novelty guidance - we want to generate molecules that are DIFFERENT.
    """

    def __init__(self, hidden_size: int, hidden_dim: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 0=novel, 1=similar to training
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden_states)


def compute_fingerprint_similarity(smiles1: str, smiles2: str) -> float:
    """Compute Tanimoto similarity between two molecules."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0


def compute_max_similarity(smiles: str, reference_smiles: List[str],
                          sample_size: int = 100) -> float:
    """Compute max similarity to reference set (sampled for efficiency)."""
    if len(reference_smiles) > sample_size:
        import random
        sampled = random.sample(reference_smiles, sample_size)
    else:
        sampled = reference_smiles

    max_sim = 0.0
    for ref in sampled:
        sim = compute_fingerprint_similarity(smiles, ref)
        max_sim = max(max_sim, sim)
    return max_sim


def create_novelty_labels(
    smiles_list: List[str],
    reference_smiles: List[str],
    similarity_threshold: float = 0.7
) -> List[int]:
    """
    Create labels for novelty classification.
    1 = similar to reference (high similarity), 0 = novel (low similarity)
    """
    labels = []
    for smi in smiles_list:
        max_sim = compute_max_similarity(smi, reference_smiles)
        labels.append(1 if max_sim > similarity_threshold else 0)
    return labels


# ============================================================================
# FRAGMENT CONSTRAINT: Keep specific substructures during generation
# ============================================================================

class FragmentConstrainedGenerator:
    """
    Generator that ensures specific fragments/substructures are preserved.

    Two modes:
    1. Prefix mode: Start with fragment tokens, continue generation
    2. Constraint mode: Penalize generations that don't contain fragment
    """

    def __init__(
        self,
        model: nn.Module,
        stoi: Dict[str, int],
        device: str = 'cuda'
    ):
        self.model = model
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
        self.device = device
        self.model.to(device)

        import re
        self.pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)

    def tokenize(self, smiles: str) -> List[int]:
        """Tokenize SMILES string."""
        tokens = self.regex.findall(smiles)
        return [self.stoi.get(t, 0) for t in tokens]

    def decode(self, ids: torch.Tensor) -> str:
        """Decode token ids to SMILES."""
        return ''.join([self.itos.get(int(t), '') for t in ids])

    def contains_fragment(self, smiles: str, fragment: str) -> bool:
        """Check if molecule contains fragment as substructure."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            frag = Chem.MolFromSmarts(fragment)
            if mol is None or frag is None:
                return fragment in smiles
            return mol.HasSubstructMatch(frag)
        except:
            return fragment in smiles

    def generate_with_prefix(
        self,
        fragment: str,
        num_samples: int = 10,
        max_length: int = 72,
        temperature: float = 1.0,
        top_k: int = 0
    ) -> List[str]:
        """
        Generate molecules starting with fragment as prefix.
        The fragment tokens are fixed, model continues from there.
        """
        self.model.eval()

        # Tokenize fragment
        fragment_ids = self.tokenize(fragment)
        x = torch.tensor([fragment_ids], dtype=torch.long, device=self.device)
        x = x.repeat(num_samples, 1)

        # Generate continuation
        with torch.no_grad():
            for _ in range(max_length - len(fragment_ids)):
                logits, _, _ = self.model(x)
                next_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    v, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < v[:, -1:]] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=1)

        # Decode results
        results = []
        for i in range(num_samples):
            smiles = self.decode(x[i]).replace('<', '').strip()
            results.append(smiles)

        return results

    def generate_with_constraint(
        self,
        fragment: str,
        num_samples: int = 10,
        max_length: int = 72,
        temperature: float = 1.0,
        top_k: int = 0,
        max_attempts: int = 100
    ) -> List[str]:
        """
        Generate molecules that must contain the fragment.
        Uses rejection sampling - generates many, keeps those with fragment.
        """
        self.model.eval()
        results = []
        attempts = 0

        # Start with single token
        start_token = self.stoi.get('C', 0)

        while len(results) < num_samples and attempts < max_attempts:
            batch_size = min(50, max_attempts - attempts)
            x = torch.tensor([[start_token]], dtype=torch.long, device=self.device)
            x = x.repeat(batch_size, 1)

            with torch.no_grad():
                for _ in range(max_length - 1):
                    logits, _, _ = self.model(x)
                    next_logits = logits[:, -1, :] / temperature

                    if top_k > 0:
                        v, _ = torch.topk(next_logits, top_k)
                        next_logits[next_logits < v[:, -1:]] = float('-inf')

                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    x = torch.cat([x, next_token], dim=1)

            # Check which contain fragment
            for i in range(batch_size):
                smiles = self.decode(x[i]).replace('<', '').strip()
                if self.contains_fragment(smiles, fragment):
                    results.append(smiles)
                    if len(results) >= num_samples:
                        break

            attempts += batch_size

        return results


# ============================================================================
# MULTI-OBJECTIVE PARETO OPTIMIZATION
# ============================================================================

class MultiObjectiveClassifier(nn.Module):
    """
    Multi-head classifier for multiple molecular properties.
    Each head predicts one property (binary classification).
    """

    def __init__(
        self,
        hidden_size: int,
        property_names: List[str],
        hidden_dim: int = 128
    ):
        super().__init__()
        self.property_names = property_names
        self.num_objectives = len(property_names)

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Separate head for each property
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2)
            )
            for name in property_names
        })

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_features = self.shared(hidden_states)
        return {
            name: head(shared_features)
            for name, head in self.heads.items()
        }

    def get_probabilities(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(hidden_states)
        return {
            name: F.softmax(logit, dim=-1)[:, 1]
            for name, logit in logits.items()
        }


def pareto_dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """Check if obj1 Pareto-dominates obj2."""
    return np.all(obj1 >= obj2) and np.any(obj1 > obj2)


def compute_pareto_front(objectives: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Compute Pareto front from objective vectors.
    Returns pareto_front array and indices of Pareto-optimal points.
    """
    n = len(objectives)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if pareto_dominates(objectives[j], objectives[i]):
                is_pareto[i] = False
                break

    pareto_indices = np.where(is_pareto)[0].tolist()
    pareto_front = objectives[is_pareto]
    return pareto_front, pareto_indices


def compute_crowding_distance(pareto_front: np.ndarray) -> np.ndarray:
    """Compute crowding distance for diversity preservation."""
    n, m = pareto_front.shape
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)

    for obj_idx in range(m):
        sorted_indices = np.argsort(pareto_front[:, obj_idx])
        obj_range = (pareto_front[sorted_indices[-1], obj_idx] -
                     pareto_front[sorted_indices[0], obj_idx])

        if obj_range == 0:
            continue

        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        for i in range(1, n - 1):
            distances[sorted_indices[i]] += (
                pareto_front[sorted_indices[i + 1], obj_idx] -
                pareto_front[sorted_indices[i - 1], obj_idx]
            ) / obj_range

    return distances


class MultiObjectivePPLM(PPLMGenerator):
    """
    Multi-objective PPLM using Pareto-based gradient aggregation.

    Key innovation: Instead of optimizing single objective, we compute
    gradients for multiple objectives and aggregate them using Pareto weights.
    """

    def __init__(
        self,
        model: nn.Module,
        classifier: MultiObjectiveClassifier,
        stepsize: float = 0.03,
        num_iterations: int = 3,
        gm_scale: float = 0.9,
        aggregation: str = 'weighted_sum',
        device: str = 'cuda'
    ):
        self.model = model
        self.mo_classifier = classifier
        self.stepsize = stepsize
        self.num_iterations = num_iterations
        self.gm_scale = gm_scale
        self.aggregation = aggregation
        self.device = device
        self.property_names = classifier.property_names

        self.model.to(device)
        self.mo_classifier.to(device)

    def compute_multi_objective_gradient(
        self,
        hidden_states: torch.Tensor,
        target_classes: Dict[str, int],
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute aggregated gradient from multiple objectives.

        Args:
            hidden_states: [B, T, H] hidden states
            target_classes: Dict mapping property name to target class (0 or 1)
            weights: Optional weights for each objective

        Returns:
            Aggregated gradient tensor
        """
        if weights is None:
            weights = {name: 1.0 / len(self.property_names)
                       for name in self.property_names}

        perturbed = hidden_states.clone().detach()
        perturbed.requires_grad_(True)

        gradients = {}

        for name in self.property_names:
            if perturbed.grad is not None:
                perturbed.grad.zero_()

            pooled = perturbed.mean(dim=1)
            logits = self.mo_classifier.heads[name](
                self.mo_classifier.shared(pooled)
            )

            target = target_classes.get(name, 1)
            loss = -F.log_softmax(logits, dim=-1)[:, target].mean()
            loss.backward(retain_graph=True)

            if perturbed.grad is not None:
                grad = perturbed.grad.clone()
                grad_norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
                gradients[name] = grad / grad_norm

        # Aggregate gradients
        aggregated = torch.zeros_like(hidden_states)
        for name, grad in gradients.items():
            aggregated += weights[name] * grad

        return aggregated

    def generate_pareto_optimal(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        target_classes: Dict[str, int],
        weights: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
        top_k: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Generate molecules optimizing multiple objectives."""
        self.model.eval()
        self.mo_classifier.eval()

        generated = input_ids.clone()
        objective_scores = {name: [] for name in self.property_names}

        for step in range(max_length - input_ids.size(1)):
            # Get hidden states
            hidden, logits = self.get_hidden_states(generated)

            # Compute multi-objective gradient
            grad = self.compute_multi_objective_gradient(
                hidden, target_classes, weights
            )

            # Apply perturbation
            perturbed = hidden + self.stepsize * self.gm_scale * grad

            # Get logits from perturbed hidden
            next_logits = self._hidden_to_logits(perturbed)[:, -1, :] / temperature

            if top_k > 0:
                next_logits = self._top_k_filtering(next_logits, top_k)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Track objective scores
            with torch.no_grad():
                pooled = perturbed.mean(dim=1)
                probs_dict = self.mo_classifier.get_probabilities(pooled)
                for name in self.property_names:
                    objective_scores[name].append(
                        probs_dict[name].mean().item()
                    )

        return generated, objective_scores


# ============================================================================
# ACTIVE LEARNING LOOP
# ============================================================================

class ActiveLearningPPLM:
    """
    Active Learning loop for molecular generation with PPLM.

    Key innovation: Iteratively improve generation by:
    1. Generate molecules with current classifier
    2. Evaluate generated molecules (oracle/simulation)
    3. Select informative samples for classifier update
    4. Retrain classifier and repeat

    Acquisition functions for sample selection:
    - Uncertainty: Select samples where classifier is uncertain
    - Diversity: Select samples that are diverse from current training set
    - Pareto: Select samples on Pareto front of objectives
    """

    def __init__(
        self,
        model: nn.Module,
        classifier: nn.Module,
        stoi: Dict[str, int],
        property_names: List[str],
        device: str = 'cuda'
    ):
        self.model = model
        self.classifier = classifier
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
        self.property_names = property_names
        self.device = device

        # Training data buffer
        self.smiles_buffer: List[str] = []
        self.labels_buffer: Dict[str, List[int]] = {
            name: [] for name in property_names
        }
        self.hidden_buffer: List[torch.Tensor] = []

        import re
        self.pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)

    def tokenize(self, smiles: str) -> List[int]:
        tokens = self.regex.findall(smiles)
        return [self.stoi.get(t, 0) for t in tokens]

    def decode(self, ids: torch.Tensor) -> str:
        return ''.join([self.itos.get(int(t), '') for t in ids])

    def compute_uncertainty(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction uncertainty (entropy) for acquisition."""
        self.classifier.eval()
        with torch.no_grad():
            if isinstance(self.classifier, MultiObjectiveClassifier):
                probs = self.classifier.get_probabilities(hidden_states)
                uncertainties = []
                for name, p in probs.items():
                    ent = -p * torch.log(p + 1e-10) - (1-p) * torch.log(1-p + 1e-10)
                    uncertainties.append(ent)
                return torch.stack(uncertainties, dim=-1).mean(dim=-1)
            else:
                logits = self.classifier(hidden_states)
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                return entropy

    def select_samples(
        self,
        smiles_list: List[str],
        hidden_states: torch.Tensor,
        n_select: int,
        strategy: str = 'uncertainty'
    ) -> List[int]:
        """
        Select informative samples using acquisition function.

        Args:
            smiles_list: Candidate SMILES
            hidden_states: [N, H] hidden states
            n_select: Number of samples to select
            strategy: 'uncertainty', 'diversity', or 'pareto'
        """
        n = len(smiles_list)
        n_select = min(n_select, n)

        if strategy == 'uncertainty':
            scores = self.compute_uncertainty(hidden_states)
            _, indices = torch.topk(scores, n_select)
            return indices.cpu().tolist()

        elif strategy == 'diversity':
            selected = [0]
            remaining = list(range(1, n))

            while len(selected) < n_select and remaining:
                max_min_dist = -1
                best_idx = remaining[0]

                for idx in remaining:
                    min_dist = float('inf')
                    for sel_idx in selected:
                        dist = compute_fingerprint_similarity(
                            smiles_list[idx], smiles_list[sel_idx]
                        )
                        min_dist = min(min_dist, 1 - dist)

                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = idx

                selected.append(best_idx)
                remaining.remove(best_idx)

            return selected

        elif strategy == 'pareto':
            if isinstance(self.classifier, MultiObjectiveClassifier):
                with torch.no_grad():
                    probs = self.classifier.get_probabilities(hidden_states)
                    obj_values = np.array([
                        probs[name].cpu().numpy()
                        for name in self.property_names
                    ]).T
                    _, pareto_idx = compute_pareto_front(obj_values)
                    if len(pareto_idx) >= n_select:
                        return pareto_idx[:n_select]
                    return pareto_idx + list(range(n_select - len(pareto_idx)))
            return list(range(n_select))

        return list(range(n_select))

    def evaluate_with_oracle(
        self,
        smiles_list: List[str],
        thresholds: Dict[str, float]
    ) -> Dict[str, List[int]]:
        """Evaluate molecules using oracle (RDKit properties)."""
        labels = {name: [] for name in self.property_names}

        for smi in smiles_list:
            props = compute_molecular_properties(smi)
            for name in self.property_names:
                if name in props and name in thresholds:
                    labels[name].append(
                        1 if props[name] > thresholds[name] else 0
                    )
                else:
                    labels[name].append(0)

        return labels

    def update_classifier(
        self,
        hidden_states: torch.Tensor,
        labels: Dict[str, List[int]],
        epochs: int = 5,
        lr: float = 1e-3
    ):
        """Update classifier with new labeled samples."""
        if not isinstance(self.classifier, MultiObjectiveClassifier):
            return

        self.classifier.train()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            total_loss = 0

            logits_dict = self.classifier(hidden_states)
            for name in self.property_names:
                target = torch.tensor(
                    labels[name], dtype=torch.long, device=self.device
                )
                loss = F.cross_entropy(logits_dict[name], target)
                total_loss += loss

            total_loss.backward()
            optimizer.step()

        self.classifier.eval()

    def run_active_learning_loop(
        self,
        initial_smiles: List[str],
        thresholds: Dict[str, float],
        n_iterations: int = 5,
        n_generate: int = 100,
        n_select: int = 20,
        acquisition: str = 'uncertainty',
        target_classes: Optional[Dict[str, int]] = None,
        temperature: float = 0.9,
        top_k: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Run the active learning loop.
        """
        from rdkit import Chem

        if target_classes is None:
            target_classes = {name: 1 for name in self.property_names}

        results = {
            'iterations': [],
            'pareto_fronts': [],
            'best_molecules': []
        }

        # Initialize with initial data
        if verbose:
            print("Initializing with seed data...")

        init_labels = self.evaluate_with_oracle(initial_smiles, thresholds)
        self._add_to_buffer(initial_smiles, init_labels)
        self._train_initial_classifier(epochs=10)

        # Create PPLM generator with conservative parameters for better validity
        if isinstance(self.classifier, MultiObjectiveClassifier):
            pplm = MultiObjectivePPLM(
                model=self.model,
                classifier=self.classifier,
                stepsize=0.01,      # Reduced from 0.02 for gentler guidance
                num_iterations=3,   # Reduced from 5 for less perturbation
                gm_scale=0.5,       # Reduced from 0.8 for weaker perturbation
                device=self.device
            )
        else:
            pplm = PPLMGenerator(
                model=self.model,
                classifier=self.classifier,
                stepsize=0.01,
                num_iterations=3,
                gm_scale=0.5,
                device=self.device
            )

        # Active learning iterations
        for iteration in range(n_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

            generated_smiles, hidden_states = self._generate_batch(
                pplm, n_generate, target_classes, temperature, top_k
            )

            valid_smiles, valid_hidden = [], []
            for i, smi in enumerate(generated_smiles):
                if Chem.MolFromSmiles(smi) is not None:
                    valid_smiles.append(smi)
                    valid_hidden.append(hidden_states[i])

            if verbose:
                print(f"Generated: {len(generated_smiles)}, Valid: {len(valid_smiles)}")

            if len(valid_smiles) == 0:
                continue

            valid_hidden = torch.stack(valid_hidden)
            selected_idx = self.select_samples(
                valid_smiles, valid_hidden, n_select, acquisition
            )
            selected_smiles = [valid_smiles[i] for i in selected_idx]
            selected_hidden = valid_hidden[selected_idx]

            new_labels = self.evaluate_with_oracle(selected_smiles, thresholds)
            self.update_classifier(selected_hidden, new_labels, epochs=5)
            self._add_to_buffer(selected_smiles, new_labels)

            iter_results = self._compute_iteration_metrics(valid_smiles, thresholds)
            results['iterations'].append(iter_results)

            if verbose:
                self._print_iteration_summary(iter_results)

        return results

    def _add_to_buffer(
        self,
        smiles_list: List[str],
        labels: Dict[str, List[int]]
    ):
        """Add samples to training buffer."""
        self.smiles_buffer.extend(smiles_list)
        for name in self.property_names:
            self.labels_buffer[name].extend(labels[name])

    def _train_initial_classifier(self, epochs: int = 10):
        """Train classifier on initial buffer data."""
        if len(self.smiles_buffer) == 0:
            return

        # Extract hidden states
        hidden_list = []
        self.model.eval()

        with torch.no_grad():
            for smi in self.smiles_buffer:
                ids = self.tokenize(smi)
                if len(ids) == 0:
                    continue
                x = torch.tensor([ids], dtype=torch.long, device=self.device)

                # Simple forward to get hidden
                b, t = x.size()
                tok_emb = self.model.tok_emb(x)
                pos_emb = self.model.pos_emb[:, :t, :]
                type_emb = self.model.type_emb(torch.ones((b, t), dtype=torch.long, device=self.device))
                h = self.model.drop(tok_emb + pos_emb + type_emb)

                for layer in self.model.blocks:
                    h, _ = layer(h)

                hidden_list.append(h.mean(dim=1).squeeze(0))

        if len(hidden_list) == 0:
            return

        hidden = torch.stack(hidden_list)
        self.update_classifier(hidden, self.labels_buffer, epochs=epochs)

    def _generate_batch(
        self,
        pplm,
        n_generate: int,
        target_classes: Dict[str, int],
        temperature: float,
        top_k: int
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate a batch of molecules."""
        start_token = self.stoi.get('C', 0)
        x = torch.tensor([[start_token]], dtype=torch.long, device=self.device)
        x = x.repeat(n_generate, 1)

        if isinstance(pplm, MultiObjectivePPLM):
            generated, _ = pplm.generate_pareto_optimal(
                x, max_length=72, target_classes=target_classes,
                temperature=temperature, top_k=top_k
            )
        else:
            generated, _ = pplm.generate_with_pplm(
                x, max_length=72, target_class=1,
                temperature=temperature, top_k=top_k
            )

        # Decode and extract hidden
        smiles_list = []
        hidden_list = []

        self.model.eval()
        with torch.no_grad():
            for i in range(n_generate):
                smi = self.decode(generated[i]).replace('<', '').strip()
                smiles_list.append(smi)

                ids = self.tokenize(smi)
                if len(ids) > 0:
                    seq = torch.tensor([ids], dtype=torch.long, device=self.device)
                    b, t = seq.size()
                    tok_emb = self.model.tok_emb(seq)
                    pos_emb = self.model.pos_emb[:, :t, :]
                    type_emb = self.model.type_emb(
                        torch.ones((b, t), dtype=torch.long, device=self.device)
                    )
                    h = self.model.drop(tok_emb + pos_emb + type_emb)
                    for layer in self.model.blocks:
                        h, _ = layer(h)
                    hidden_list.append(h.mean(dim=1).squeeze(0))
                else:
                    hidden_list.append(torch.zeros(
                        self.model.config.n_embd, device=self.device
                    ))

        return smiles_list, hidden_list

    def _compute_iteration_metrics(
        self,
        smiles_list: List[str],
        thresholds: Dict[str, float]
    ) -> Dict:
        """Compute metrics for an iteration."""
        metrics = {
            'n_valid': len(smiles_list),
            'property_stats': {}
        }

        for name in self.property_names:
            values = []
            above_threshold = 0
            for smi in smiles_list:
                props = compute_molecular_properties(smi)
                if name in props:
                    values.append(props[name])
                    if name in thresholds and props[name] > thresholds[name]:
                        above_threshold += 1

            if values:
                metrics['property_stats'][name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'above_threshold': above_threshold / len(smiles_list)
                }

        return metrics

    def _print_iteration_summary(self, metrics: Dict):
        """Print iteration summary."""
        print(f"Valid molecules: {metrics['n_valid']}")
        for name, stats in metrics['property_stats'].items():
            print(f"  {name}: mean={stats['mean']:.3f}, "
                  f"std={stats['std']:.3f}, "
                  f"above_thresh={stats['above_threshold']*100:.1f}%")
