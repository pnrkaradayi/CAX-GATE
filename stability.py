"""Attribution stability under stochastic input perturbations (Table and Figure on stability).

    python -m caxgate.stability
For a fixed random subset of CVC-ClinicDB frames, each frame is perturbed cfg.stability_copies
times (contrast jitter and Gaussian noise). Every method is recomputed on each copy and compared
with the reference map of the clean frame (L2 fidelity). The per-frame variance across copies,
averaged over frames, is the "mean fidelity variance" of one seed.
"""
import argparse

import numpy as np
import pandas as pd

from .config import CLASSES, POLYP_CLASS, get_config
from .data import prepare
from .explainers import ModelFn, gradcam_map, ig_map, lime_map, shap_map
from .fusion import blend_np, gamma_np, log_mask
from .metrics import agreement
from .models import checkpoint_path, load_classifier
from .precompute import load_maps
from .train_gate import load_gate, predict_alpha
from .uncertainty import context_vectors
from .utils import load_tau

POLYP_IDX = CLASSES.index(POLYP_CLASS)


def perturb(img01, rng, cfg):
    m = img01.mean()
    f = rng.uniform(1 - cfg.stability_contrast, 1 + cfg.stability_contrast)
    x = m + f * (img01 - m) + rng.normal(0.0, cfg.stability_noise_sigma, img01.shape)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def run(cfg, seed, data):
    out = cfg.sub("results", "stability") / f"seed{seed}.csv"
    done = set()
    rows = []
    if out.exists():
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = set(prev.frame.unique())
    maps = load_maps(cfg, "cvc", seed)
    model = load_classifier(checkpoint_path(cfg, "proposed", seed), len(CLASSES), cfg.device)
    tau = load_tau(cfg, "proposed", seed)
    fn = ModelFn(model, tau, cfg.device, batch=cfg.shap_batch)
    gate, ck = load_gate(cfg, seed, "full")
    frames = np.random.default_rng(0).choice(len(data["cvc_images"]), cfg.stability_frames, replace=False)
    for n_done, i in enumerate(frames):
        if int(i) in done:
            continue
        rng = np.random.default_rng(1000 * seed + int(i))
        clean = data["cvc_images"][i].astype(np.float32) / 255.0
        ref = maps["ref"][i]
        for c in range(cfg.stability_copies):
            x = perturb(clean, rng, cfg)
            L, _ = lime_map(fn, x, POLYP_IDX, cfg, seed=seed)
            S, _ = shap_map(fn, x, POLYP_IDX, cfg)
            M = log_mask(x, cfg.log_sigma)
            z = context_vectors(model, x[None], tau, cfg)["z"]
            a = float(predict_alpha(gate, ck, z)[0])
            gc, _ = gradcam_map(model, x, POLYP_IDX, cfg.device)
            ig, _ = ig_map(model, x, POLYP_IDX, cfg)
            sal = {"LIME": L, "Partition SHAP": S, "Static fusion": 0.5 * L + 0.5 * S,
                   "CAX-Gate": blend_np(L, S, gamma_np(a, M, cfg.lambda_struct)),
                   "Grad-CAM": gc, "Integrated Gradients": ig}
            for name, m in sal.items():
                rows.append({"seed": seed, "frame": int(i), "copy": c, "method": name,
                             "l2_fidelity": agreement(m, ref)["l2_fidelity"]})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"stability seed {seed}: frame {n_done + 1}/{len(frames)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = get_config(args.smoke)
    data, _ = prepare(cfg, verbose=False)
    for seed in (args.seeds if args.seeds is not None else cfg.seeds_explain):
        run(cfg, seed, data)


if __name__ == "__main__":
    main()
