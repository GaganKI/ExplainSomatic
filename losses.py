"""
losses.py
----------
VAF-aware loss: standard BCE-with-logits, except examples with a true VAF
below 5% get their loss multiplied by `low_vaf_weight` (default 5x), so the
optimizer can't just ignore them as statistically rare noise. This is the
concrete mechanism behind Objective 2.
"""

import torch
import torch.nn as nn


class VAFAwareLoss(nn.Module):
    def __init__(self, low_vaf_weight=5.0, vaf_threshold=0.05):
        super().__init__()
        self.low_vaf_weight = low_vaf_weight
        self.vaf_threshold = vaf_threshold
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, labels, vaf):
        per_example_loss = self.bce(logits, labels)
        is_low_vaf_positive = (labels > 0.5) & (vaf < self.vaf_threshold)
        weights = torch.where(is_low_vaf_positive,
                               torch.full_like(per_example_loss, self.low_vaf_weight),
                               torch.ones_like(per_example_loss))
        return (per_example_loss * weights).mean()


class PlainBCELoss(nn.Module):
    """The 'no VAF-awareness' baseline, used to isolate how much the
    weighting mechanism itself is contributing (separate from architecture)."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, vaf=None):
        return self.bce(logits, labels)