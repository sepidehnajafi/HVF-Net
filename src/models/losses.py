"""Confidence-adaptive supervised contrastive loss.

Implements the focal-style adaptive weighting described in Section 3.8:
samples the model is already confident about contribute less to the
contrastive term, focusing gradient signal on harder, ambiguous examples.

    lambda(t) = lambda_max * (1 - C(t))^gamma

where C(t) is a running estimate of classification confidence for the batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Standard supervised contrastive loss (SupCon) over L2-normalized
    embeddings, with class labels defining positive pairs.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, dim), not necessarily normalized.
            labels: (batch,) integer class labels.

        Returns:
            Scalar SupCon loss.
        """
        device = embeddings.device
        z = F.normalize(embeddings, dim=-1)
        sim = z @ z.T / self.temperature  # (batch, batch)

        batch_size = z.shape[0]
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float().to(device)
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask  # exclude self-similarity

        # For numerical stability, subtract the row-wise max before exponentiating.
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        num_positives = positive_mask.sum(dim=1)
        # Samples with no positive pair in-batch contribute zero loss.
        valid = num_positives > 0
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / num_positives[valid]
        return -mean_log_prob_pos.mean()


class AdaptiveContrastiveWeight(nn.Module):
    """Computes the confidence-adaptive weight lambda(t) for a batch.

    Args:
        lambda_max: maximum contrastive loss weight (Table 2: 4.0).
        gamma: focusing exponent (Table 2: 2.0).
    """

    def __init__(self, lambda_max: float = 4.0, gamma: float = 2.0):
        super().__init__()
        self.lambda_max = lambda_max
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) classification logits.
            labels: (batch,) integer class labels.

        Returns:
            Scalar weight lambda(t) for the current batch, in [0, lambda_max].
        """
        probs = F.softmax(logits, dim=-1)
        confidence = probs.gather(1, labels.view(-1, 1)).squeeze(1).mean()
        return self.lambda_max * (1.0 - confidence).clamp(min=0.0) ** self.gamma


class TotalLoss(nn.Module):
    """Cross-entropy classification loss plus an adaptively-weighted
    supervised contrastive term.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_max: float = 4.0,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.contrastive = SupervisedContrastiveLoss(temperature)
        self.adaptive_weight = AdaptiveContrastiveWeight(lambda_max, gamma)

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys ``total``, ``ce``, ``contrastive``, ``lambda`` for
            logging.
        """
        ce_loss = self.ce(logits, labels)
        contrastive_loss = self.contrastive(embeddings, labels)
        lam = self.adaptive_weight(logits.detach(), labels)
        total = ce_loss + lam * contrastive_loss
        return {
            "total": total,
            "ce": ce_loss.detach(),
            "contrastive": contrastive_loss.detach(),
            "lambda": lam.detach(),
        }
