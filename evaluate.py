"""Evaluate every attribution method on CVC-ClinicDB (Tables 4-8, Figures 5-6 of the manuscript).

    python -m caxgate.evaluate
Writes, per seed: results/explain/seed{s}_frames.csv (one row per frame and method),
results/explain/seed{s}_summary.json (alpha statistics, curves, fast-path share) and, for the
first seed, results/explain/qualitative.npz (maps of one frame per uncertainty tier).
"""
import argparse

import numpy as np
import pandas as pd

from .config import CLASSES, POLYP_CLASS, get_config
from .data import prepare
from .explainers import ModelFn, blurred
from .fusion import GATE_VARIANTS, alpha_entropy_rule, alpha_tertile_rule, blend_np, gamma_np
from .metrics import agreement, deletion_insertion, mass_in_mask, pointing_game
from .models import checkpoint_path, load_classifier
from .precompute import load_maps
from .stats import spearman
from .train_gate import load_gate, predict_alpha
from .utils import load_json, load_tau, save_json

POLYP_IDX = CLASSES.index(POLYP_CLASS)
GATE_LABELS = {
    "entropy_only": "Gate: entropy only",
    "posterior": "Gate: posterior inputs",
    "posterior_epistemic": "Gate: posterior + epistemic",
    "full_no_log": "CAX-Gate without LoG",
    "full": "CAX-Gate",
}
MAIN_METHODS = ["LIME", "Partition SHAP", "Static fusion", "CAX-Gate"]


def method_alphas(cfg, maps, seed):
    """alpha per frame for every alpha-driven method, and whether it uses the LoG modulation."""
    u = maps["u_aleatoric"]
    thr = load_json(cfg.sub("gates") / f"seed{seed}_entropy_tertiles.json")["thresholds"]
    out = {
        "Fixed rule (1-H)": (alpha_entropy_rule(u), True),
        "Fixed tertile rule": (alpha_tertile_rule(u, thr, cfg.tertile_alphas), True),
    }
    for variant, label in gate_variants_labelled():
        gate, ck = load_gate(cfg, seed, variant)
        out[label] = (predict_alpha(gate, ck, maps["z"]), bool(ck["use_log"]))
    return out


def gate_variants_labelled():
    return [(v, GATE_LABELS[v]) for v in GATE_VARIANTS]


def frame_maps(cfg, maps, alphas, i):
    L, S, M = maps["lime"][i], maps["shap"][i], maps["logm"][i]
    out = {"LIME": L, "Partition SHAP": S, "Static fusion": 0.5 * L + 0.5 * S,
           "Grad-CAM": maps["gradcam"][i], "Integrated Gradients": maps["ig"][i]}
    for name, (a, use_log) in alphas.items():
        lam = cfg.lambda_struct if use_log else 0.0
        out[name] = blend_np(L, S, gamma_np(float(a[i]), M, lam))
    return out


def tiers_of(composite):
    t1, t2 = np.quantile(composite, [1 / 3, 2 / 3])
    return np.where(composite <= t1, 0, np.where(composite <= t2, 1, 2)), [float(t1), float(t2)]


def run(cfg, seed, data):
    out_dir = cfg.sub("results", "explain")
    frames_csv = out_dir / f"seed{seed}_frames.csv"
    if frames_csv.exists():
        print(f"[skip] evaluate seed {seed}")
        return
    maps = load_maps(cfg, "cvc", seed)
    model = load_classifier(checkpoint_path(cfg, "proposed", seed), len(CLASSES), cfg.device)
    fn = ModelFn(model, load_tau(cfg, "proposed", seed), cfg.device, batch=cfg.shap_batch)
    alphas = method_alphas(cfg, maps, seed)
    tier, thresholds = tiers_of(maps["composite"])
    imgs, masks = data["cvc_images"], data["cvc_masks"]

    rows, curves = [], {}
    for i in range(len(imgs)):
        img = imgs[i].astype(np.float32) / 255.0
        base = blurred(img, cfg.blur_sigma)
        for name, sal in frame_maps(cfg, maps, alphas, i).items():
            fr, d, ins, d_auc, i_auc = deletion_insertion(fn, img, sal, POLYP_IDX, base, cfg.curve_step)
            c = curves.setdefault(name, {"deletion": np.zeros_like(d), "insertion": np.zeros_like(ins)})
            c["deletion"] += d
            c["insertion"] += ins
            rows.append({"seed": seed, "frame": i, "method": name,
                         "alpha": float(alphas[name][0][i]) if name in alphas else np.nan,
                         "pointing_game": pointing_game(sal, masks[i]),
                         "mass_in_mask": mass_in_mask(sal, masks[i], cfg.mass_top_fraction),
                         "deletion_auc": d_auc, "insertion_auc": i_auc,
                         **agreement(sal, maps["ref"][i]),
                         "composite": float(maps["composite"][i]), "u_aleatoric": float(maps["u_aleatoric"][i]),
                         "tier": int(tier[i])})
        if (i + 1) % 50 == 0:
            print(f"evaluate seed {seed}: {i + 1}/{len(imgs)} frames")
    df = pd.DataFrame(rows)

    a = alphas["CAX-Gate"][0]
    fast = a >= cfg.alpha_fast
    lime_pg = df[df.method == "LIME"].sort_values("frame")["pointing_game"].to_numpy()
    rho, p = spearman(maps["composite"], a)
    summary = {
        "seed": seed, "tier_thresholds": thresholds,
        "tier_sizes": [int((tier == t).sum()) for t in range(3)],
        "alpha_by_tier": [describe(a[tier == t]) for t in range(3)],
        "spearman_composite_alpha": {"rho": rho, "p": p},
        "fast_path_fraction": float(fast.mean()),
        "fast_path_frames_lime_pointing_game": float(lime_pg[fast].mean()) if fast.any() else None,
        "curve_fractions": fr.tolist(),
        "curves": {k: {kk: (vv / len(imgs)).tolist() for kk, vv in v.items()} for k, v in curves.items()},
        "times_seconds": {k[2:]: float(maps[k].mean()) for k in maps if k.startswith("t_")},
    }
    save_json(out_dir / f"seed{seed}_summary.json", summary)
    df.to_csv(frames_csv, index=False)

    if seed == cfg.seeds_explain[0]:
        pick = [int(np.argmin(np.abs(maps["composite"] - np.median(maps["composite"][tier == t])) + (tier != t) * 1e9))
                for t in range(3)]
        q = {"frames": np.array(pick), "images": imgs[pick], "masks": masks[pick],
             "alpha": a[pick], "composite": maps["composite"][pick]}
        for name in ["LIME", "Partition SHAP", "Static fusion", "CAX-Gate"]:
            q[name] = np.stack([frame_maps(cfg, maps, alphas, i)[name] for i in pick])
        np.savez_compressed(out_dir / "qualitative.npz", **{k.replace(" ", "_"): v for k, v in q.items()})


def describe(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "median": float(med),
            "iqr": float(q3 - q1), "min": float(x.min()), "max": float(x.max()), "n": int(len(x))}


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
