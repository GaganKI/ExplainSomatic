"""
gradcam.py
-----------
Grad-CAM over Stream A's last convolutional feature map. This is the
Objective 3 explainability mechanism: for a given call, it produces a
(reads x positions) heatmap showing which reads/positions in the pileup
most influenced the model's decision -- the thing a clinician would
actually look at instead of trusting a bare "variant" label.
"""

import torch
import torch.nn.functional as F


def grad_cam(model, pileup, ctx_tokens, target_class=1):
    """
    model: an ExplainSomaticModel (or CNNOnlyModel) instance
    pileup, ctx_tokens: a SINGLE example, batch dimension = 1
    Returns: heatmap of shape (reads, width), values in [0, 1]
    """
    model.eval()
    pileup = pileup.clone().requires_grad_(False)

    logit = model(pileup, ctx_tokens)  # forward pass populates stream_a.last_feature_map
    feat_map = model.stream_a.last_feature_map  # (1, 64, reads, width)
    feat_map.retain_grad()

    score = logit.sum()
    model.zero_grad()
    score.backward(retain_graph=False)

    grads = feat_map.grad  # (1, 64, reads, width)
    weights = grads.mean(dim=(2, 3), keepdim=True)  # global-average-pool the gradients -> per-channel importance
    cam = F.relu((weights * feat_map).sum(dim=1))   # (1, reads, width)
    cam = cam[0]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.detach()


if __name__ == "__main__":
    from data_sim import SomaticSimDataset
    from models import ExplainSomaticModel

    ds = SomaticSimDataset(1, seed=3, homopolymer_prob=0.0)
    pileup, ctx, label, vaf, hp = ds[0]
    pileup = pileup.unsqueeze(0)
    ctx = ctx.unsqueeze(0)

    model = ExplainSomaticModel(transformer_layers=2)
    heatmap = grad_cam(model, pileup, ctx)
    print("heatmap shape:", heatmap.shape, "min/max:", heatmap.min().item(), heatmap.max().item())