"""Train the gating network g_theta (and its ablation variants) on precomputed maps.

    python -m caxgate.train_gate

Loss (Eq. loss_total of the manuscript):
    L = L_faith + beta_1 L_clinical + beta_2 L_sparsity + beta_3 L_cost
L_faith uses a sigmoid relaxation of top-k masking so that gradients reach alpha:
    m_k = sigmoid((Phi - q_k) / temp), q_k = (1-k) quantile of Phi (detached),
    x_k = (1 - m_k) x + m_k blur(x),  L_faith = mean_k relu(p(y*|x_k) - (1-k) p(y*|x)).
The backbone is frozen.
"""
import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .config import CLASSES, POLYP_CLASS, get_config
from .data import prepare
from .fusion import GATE_VARIANTS, Gate, blend_torch
from .models import checkpoint_path, load_classifier
from .precompute import load_maps
from .utils import load_tau, save_json, seed_everything

POLYP_IDX = CLASSES.index(POLYP_CLASS)


def gate_path(cfg, seed, variant):
    return cfg.sub("gates") / f"seed{seed}_{variant}.pt"


def _blur_torch(x, sigma):
    import torchvision.transforms.functional as TF
    k = int(2 * round(3 * sigma) + 1)
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def gate_losses(cfg, model, tau, gate, inputs, use_log, batch, t_ratio):
    x, lime_m, shap_m, logm, masks, z = batch
    lam = cfg.lambda_struct if use_log else 0.0
    alpha = gate(z[:, inputs])
    phi = blend_torch(alpha, lime_m, shap_m, logm, lam)

    with torch.no_grad():
        p0 = F.softmax(model(x).float() / tau, 1)[:, POLYP_IDX]
        base = _blur_torch(x, cfg.blur_sigma)
    faith = 0.0
    flat = phi.flatten(1)
    for k in cfg.faith_fractions:
        q = torch.quantile(flat.detach(), 1 - k, dim=1).view(-1, 1, 1)
        m = torch.sigmoid((phi - q) / cfg.faith_soft_temp).unsqueeze(1)
        xk = (1 - m) * x + m * base
        pk = F.softmax(model(xk).float() / tau, 1)[:, POLYP_IDX]
        faith = faith + F.relu(pk - (1 - k) * p0).mean()
    faith = faith / len(cfg.faith_fractions)

    inter = (phi * masks).flatten(1).sum(1)
    union = phi.flatten(1).sum(1) + masks.flatten(1).sum(1) - inter
    clinical = (1 - inter / (union + 1e-6)).mean()
    sparsity = phi.abs().mean()
    cost = ((1 - alpha) * t_ratio).mean()
    total = faith + cfg.beta_clinical * clinical + cfg.beta_sparsity * sparsity + cfg.beta_cost * cost
    return total, {"faith": float(faith), "clinical": float(clinical), "sparsity": float(sparsity),
                   "cost": float(cost), "alpha_mean": float(alpha.mean())}


def train_variant(cfg, seed, variant, maps, data, splits):
    out = gate_path(cfg, seed, variant)
    if out.exists():
        print(f"[skip] gate seed {seed} {variant}")
        return
    seed_everything(seed)
    device = cfg.device
    spec = GATE_VARIANTS[variant]
    inputs = spec["inputs"]
    model = load_classifier(checkpoint_path(cfg, "proposed", seed), len(CLASSES), device)
    for p in model.parameters():
        p.requires_grad_(False)
    tau = load_tau(cfg, "proposed", seed)

    pos = {int(j): i for i, j in enumerate(maps["idx"])}
    tr = np.array([pos[j] for j in splits["gate"]["train"]])
    va = np.array([pos[j] for j in splits["gate"]["val"]])
    z = maps["z"]
    z_mean, z_std = z[tr][:, inputs].mean(0), z[tr][:, inputs].std(0)
    gate = Gate(len(inputs), cfg.gate_hidden, z_mean, z_std).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=cfg.gate_lr)
    t_ratio = float(maps["t_shap"].mean() / (maps["t_lime"].mean() + maps["t_shap"].mean()))

    def batches(rows, shuffle):
        rows = np.random.permutation(rows) if shuffle else rows
        for b in range(0, len(rows), cfg.gate_batch):
            r = rows[b:b + cfg.gate_batch]
            src = maps["idx"][r]
            x = torch.from_numpy(data["seg_images"][src].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            yield tuple(t.to(device) for t in (
                x, torch.from_numpy(maps["lime"][r]), torch.from_numpy(maps["shap"][r]),
                torch.from_numpy(maps["logm"][r]), torch.from_numpy(data["seg_masks"][src].astype(np.float32)),
                torch.from_numpy(z[r])))

    best, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(cfg.gate_epochs):
        gate.train()
        for batch in batches(tr, True):
            loss, _ = gate_losses(cfg, model, tau, gate, inputs, spec["use_log"], batch, t_ratio)
            opt.zero_grad()
            loss.backward()
            opt.step()
        gate.eval()
        tot, n, parts = 0.0, 0, {}
        for batch in batches(va, False):
            with torch.no_grad():
                loss, comp = gate_losses(cfg, model, tau, gate, inputs, spec["use_log"], batch, t_ratio)
            k = len(batch[0])
            tot += float(loss) * k
            n += k
            for key, v in comp.items():
                parts[key] = parts.get(key, 0.0) + v * k
        val = tot / max(n, 1)
        history.append({"epoch": epoch + 1, "val_total": val, **{k: v / max(n, 1) for k, v in parts.items()}})
        print(f"gate seed {seed} {variant} epoch {epoch + 1}: val {val:.4f}")
        if val < best - 1e-5:
            best, best_state, wait = val, copy.deepcopy(gate.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.gate_patience:
                break
    torch.save({"state_dict": best_state, "inputs": inputs, "use_log": spec["use_log"],
                "z_mean": z_mean, "z_std": z_std, "t_ratio": t_ratio, "history": history}, out)


def load_gate(cfg, seed, variant):
    ck = torch.load(gate_path(cfg, seed, variant), map_location="cpu", weights_only=False)
    gate = Gate(len(ck["inputs"]), cfg.gate_hidden, ck["z_mean"], ck["z_std"])
    gate.load_state_dict(ck["state_dict"])
    return gate.eval(), ck


@torch.no_grad()
def predict_alpha(gate, ck, z):
    return gate(torch.from_numpy(np.asarray(z, np.float32)[:, ck["inputs"]])).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=list(GATE_VARIANTS))
    args = ap.parse_args()
    cfg = get_config(args.smoke)
    data, splits = prepare(cfg, verbose=False)
    for seed in (args.seeds if args.seeds is not None else cfg.seeds_explain):
        maps = load_maps(cfg, "gate", seed)
        pos = {int(j): i for i, j in enumerate(maps["idx"])}
        tr = [pos[j] for j in splits["gate"]["train"]]
        u = maps["u_aleatoric"][tr]
        save_json(cfg.sub("gates") / f"seed{seed}_entropy_tertiles.json",
                  {"thresholds": [float(np.quantile(u, 1 / 3)), float(np.quantile(u, 2 / 3))]})
        for variant in args.variants:
            train_variant(cfg, seed, variant, maps, data, splits)


if __name__ == "__main__":
    main()
