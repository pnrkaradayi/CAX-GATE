"""Aggregate all outputs into the tables and figures of the manuscript.

    python -m caxgate.report
Writes results/report/ (CSV + LaTeX rows for every table, PNG/PDF figures, summary.json) and
results_report.zip in the output folder. Send that zip file back to update the manuscript.
"""
import argparse
import json
import shutil
from importlib import metadata

import numpy as np
import pandas as pd

from .config import CLASSES, get_config
from .stats import bonferroni, bootstrap_ci, cohens_d_paired, mcnemar_exact, paired_t, stars, wilson
from .utils import load_json, save_json

PRIMARY_ORDER = ["LIME", "Partition SHAP", "Static fusion", "Grad-CAM", "Integrated Gradients", "CAX-Gate"]
ABLATION_ORDER = ["Static fusion", "Fixed rule (1-H)", "Fixed tertile rule", "Gate: entropy only",
                  "Gate: posterior inputs", "Gate: posterior + epistemic", "CAX-Gate without LoG", "CAX-Gate"]
TIER_METHODS = ["LIME", "Partition SHAP", "Static fusion", "CAX-Gate"]
MODEL_LABELS = {"svm_hog": "SVM (HOG features)", "rf_hog": "Random Forest (HOG)", "resnet50": "ResNet-50",
                "efficientnet_b0": "EfficientNet-B0", "proposed": "Proposed CNN (ConvNeXt-T)"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_table(rdir, name, df, latex_cols=None):
    df.to_csv(rdir / f"{name}.csv", index=False)
    cols = latex_cols or list(df.columns)
    lines = [" & ".join(str(v) for v in row) + r" \\" for row in df[cols].astype(str).values]
    (rdir / f"{name}.tex").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ classification
def classification_table(cfg, rdir):
    rows, preds = [], {}
    labels = np.load(cfg.out / "cache" / "hk_labels.npy")
    for split in ("grouped", "image_level"):
        d = cfg.out / "results" / "classification" / split
        if not d.exists():
            continue
        for name in ["svm_hog", "rf_hog", "resnet50", "efficientnet_b0", "proposed"]:
            res = [load_json(p) for p in sorted(d.glob(f"{name}_seed*.json"))]
            if not res:
                continue
            for p in sorted(d.glob(f"{name}_seed*_preds.npz")):
                z = np.load(p)
                preds.setdefault((split, name), []).append(z)
            acc = np.array([r["test"]["accuracy"] for r in res])
            rec = np.array([r["cvc_polyp_recall"] for r in res])
            n_test, n_cvc = res[0]["n_test"], res[0]["n_cvc"]
            row = {"split": split, "model": MODEL_LABELS[name] + (" (image-level split)" if split == "image_level" else ""),
                   "key": name, "seeds": len(res), "n_test": n_test,
                   "accuracy": 100 * acc.mean(), "accuracy_sd": 100 * acc.std(ddof=1) if len(acc) > 1 else 0.0,
                   "accuracy_ci": wilson(acc.mean(), n_test)}
            for k in ("balanced_accuracy", "precision_macro", "f1_macro", "auc_macro_ovr"):
                v = np.array([r["test"][k] for r in res])
                scale = 1 if k == "auc_macro_ovr" else 100
                row[k], row[k + "_sd"] = scale * v.mean(), scale * (v.std(ddof=1) if len(v) > 1 else 0.0)
            row.update({"cvc_polyp_recall": 100 * rec.mean(), "cvc_polyp_recall_ci": wilson(rec.mean(), n_cvc),
                        "n_cvc": n_cvc, "n_parameters": res[0].get("n_parameters"), "tau_mean": np.mean([r.get("tau", np.nan) for r in res])})
            rows.append(row)
    df = pd.DataFrame(rows)
    # McNemar vs ResNet-50 on per-image majority correctness across seeds (grouped split)
    def majority_correct(key, which):
        zs = preds.get(("grouped", key), [])
        if not zs:
            return None
        if which == "test":
            y = labels[zs[0]["test_idx"]]
            return (np.mean([z["test_probs"].argmax(1) == y for z in zs], axis=0) >= 0.5)
        return (np.mean([z["cvc_probs"].argmax(1) == CLASSES.index("polyps") for z in zs], axis=0) >= 0.5)
    sig = []
    others = [k for k in ["svm_hog", "rf_hog", "efficientnet_b0", "proposed"] if ("grouped", k) in preds]
    for _, r in df.iterrows():
        entry = {}
        if r["split"] == "grouped" and r["key"] in others and ("grouped", "resnet50") in preds:
            for which in ("test", "cvc"):
                p = mcnemar_exact(majority_correct(r["key"], which), majority_correct("resnet50", which))
                entry[which] = bonferroni(p, len(others))
        sig.append(entry)
    df["p_vs_resnet50_bonferroni"] = sig
    df["latex"] = [f"{r.model} & \\makecell{{{fmt(r.accuracy)}{stars(r.p_vs_resnet50_bonferroni.get('test', 1)) and '$^{' + stars(r.p_vs_resnet50_bonferroni.get('test', 1)) + '}$'}\\\\{{}}[{fmt(r.accuracy_ci[0])}, {fmt(r.accuracy_ci[1])}]}} & "
                   f"{fmt(r.f1_macro)} $\\pm$ {fmt(r.f1_macro_sd, 1)} & {fmt(r.auc_macro_ovr, 3)} $\\pm$ {fmt(r.auc_macro_ovr_sd, 3)}"
                   for r in df.itertuples()]
    write_table(rdir, "table_classification", df.drop(columns=["latex"]))
    (rdir / "table_classification.tex").write_text("\n".join(l + r" \\" for l in df["latex"]) + "\n")
    return df


# ------------------------------------------------------------------ explanation tables
def load_frames(cfg):
    files = sorted((cfg.out / "results" / "explain").glob("seed*_frames.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def load_clusters(cfg, n_frames):
    p = cfg.cvc_dir / cfg.cvc_sequences_csv
    if p.exists():
        m = pd.read_csv(p)
        names = load_json(cfg.out / "cache" / "cvc_names.json")
        seq = dict(zip(m.iloc[:, 0].astype(str), m.iloc[:, 1]))
        return np.array([seq.get(str(nm), i) for i, nm in enumerate(names)])
    return None


def per_frame(df, method, metric):
    """Per-frame value averaged over seeds, ordered by frame."""
    s = df[df.method == method].groupby("frame")[metric].mean().sort_index()
    return s.to_numpy()


def method_summary(cfg, df, methods, reference, clusters, n_tests):
    rows = []
    n = df.frame.nunique()
    ref_pg = per_frame(df, reference, "pointing_game") if reference in methods else None
    for m in methods:
        sub = df[df.method == m]
        if sub.empty:
            continue
        by_seed = sub.groupby("seed").mean(numeric_only=True)
        row = {"method": m, "seeds": len(by_seed), "n": n}
        pg = by_seed["pointing_game"].mean()
        row["pointing_game"], row["pointing_game_ci"] = 100 * pg, wilson(pg, n)
        for k in ("mass_in_mask", "deletion_auc", "insertion_auc", "l2_fidelity", "cosine", "ssim"):
            scale = 100 if k == "mass_in_mask" else 1
            row[k] = scale * by_seed[k].mean()
            row[k + "_sd"] = scale * (by_seed[k].std(ddof=1) if len(by_seed) > 1 else 0.0)
            ci = bootstrap_ci(per_frame(df, m, k), cfg.bootstrap_resamples, 0, clusters)
            row[k + "_ci"] = (scale * ci[0], scale * ci[1])
        if reference and m != reference:
            row["p_pg"] = bonferroni(mcnemar_exact(per_frame(df, m, "pointing_game") >= 0.5, ref_pg >= 0.5), n_tests)
            for k in ("mass_in_mask", "deletion_auc", "insertion_auc", "l2_fidelity", "cosine", "ssim"):
                row["p_" + k] = bonferroni(paired_t(per_frame(df, m, k), per_frame(df, reference, k))[1], n_tests)
        if "LIME" in methods and m != "LIME":
            row["cohens_d_mim_vs_lime"] = cohens_d_paired(per_frame(df, m, "mass_in_mask"), per_frame(df, "LIME", "mass_in_mask"))
        rows.append(row)
    return pd.DataFrame(rows)


def tier_table(df):
    rows = []
    for t in range(3):
        sub_t = df[df.tier == t]
        for m in TIER_METHODS:
            sub = sub_t[sub_t.method == m]
            if sub.empty:
                continue
            by_seed = sub.groupby("seed").mean(numeric_only=True)
            n_t = int(sub.groupby("seed").size().mean())
            pg = by_seed["pointing_game"].mean()
            rows.append({"tier": ["low", "medium", "high"][t], "method": m, "n": n_t,
                         "pointing_game": 100 * pg, "pointing_game_ci": wilson(pg, n_t),
                         "l2_fidelity": by_seed["l2_fidelity"].mean(),
                         "l2_fidelity_sd": by_seed["l2_fidelity"].std(ddof=1) if len(by_seed) > 1 else 0.0,
                         "cosine": by_seed["cosine"].mean(),
                         "cosine_sd": by_seed["cosine"].std(ddof=1) if len(by_seed) > 1 else 0.0})
    out = pd.DataFrame(rows)
    if not out.empty:
        lime = out[out.method == "LIME"].set_index("tier")["l2_fidelity"]
        out["gain_l2_vs_lime_pct"] = [100 * (r.l2_fidelity - lime[r.tier]) / lime[r.tier] if r.method != "LIME" else np.nan
                                      for r in out.itertuples()]
    return out


def alpha_table(cfg):
    sums = [load_json(p) for p in sorted((cfg.out / "results" / "explain").glob("seed*_summary.json"))]
    if not sums:
        return None, None
    rows = []
    for t in range(3):
        stats = [s["alpha_by_tier"][t] for s in sums if s["alpha_by_tier"][t]]
        rows.append({"tier": ["low", "medium", "high"][t], **{k: float(np.mean([s[k] for s in stats])) for k in
                                                             ("mean", "sd", "median", "iqr", "min", "max", "n")}})
    extra = {
        "spearman_rho_mean": float(np.mean([s["spearman_composite_alpha"]["rho"] for s in sums])),
        "spearman_p_max": float(np.max([s["spearman_composite_alpha"]["p"] for s in sums])),
        "fast_path_fraction_mean": float(np.mean([s["fast_path_fraction"] for s in sums])),
        "fast_path_frames_lime_pg_mean": float(np.nanmean([np.nan if s["fast_path_frames_lime_pointing_game"] is None
                                                           else s["fast_path_frames_lime_pointing_game"] for s in sums])),
        "mean_times_seconds": {k: float(np.mean([s["times_seconds"][k] for s in sums])) for k in sums[0]["times_seconds"]},
    }
    return pd.DataFrame(rows), extra


def stability_table(cfg):
    files = sorted((cfg.out / "results" / "stability").glob("seed*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files])
    var = df.groupby(["seed", "method", "frame"])["l2_fidelity"].var(ddof=1).groupby(["seed", "method"]).mean()
    rows = []
    for m in var.index.get_level_values("method").unique():
        v = var.xs(m, level="method").to_numpy()
        rows.append({"method": m, "mean_fidelity_variance": v.mean(),
                     "sd_across_seeds": v.std(ddof=1) if len(v) > 1 else 0.0,
                     "cv_pct": 100 * v.std(ddof=1) / v.mean() if len(v) > 1 and v.mean() > 0 else np.nan,
                     "frames": int(df[df.method == m].frame.nunique()), "seeds": len(v)})
    out = pd.DataFrame(rows)
    lime = out.set_index("method").loc["LIME", "mean_fidelity_variance"] if "LIME" in set(out.method) else np.nan
    out["reduction_vs_lime_pct"] = 100 * (lime - out["mean_fidelity_variance"]) / lime
    return out


def latency_table(cfg):
    p = cfg.out / "results" / "latency" / "latency.csv"
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    g = df.groupby("method")["seconds"]
    out = pd.DataFrame({"mean_ms": 1000 * g.mean(), "sd_ms": 1000 * g.std(ddof=1),
                        "min_ms": 1000 * g.min(), "max_ms": 1000 * g.max()}).reset_index()
    out["fps"] = 1000 / out["mean_ms"]
    shap_ms = out.set_index("method").loc["Partition SHAP", "mean_ms"]
    out["speedup_vs_shap"] = shap_ms / out["mean_ms"]
    return out, load_json(cfg.out / "results" / "latency" / "hardware.json")


def hyperparameter_table(cfg, rdir):
    keys = ["image_size", "split_folds", "phash_max_hamming", "proposed_arch", "head_dropout", "batch_size", "lr",
            "weight_decay", "max_epochs", "patience", "mc_passes", "gamma_risk", "lime_samples", "lime_kernel_width",
            "shap_max_evals", "shap_blur", "ref_slic_segments", "ref_nsamples", "ig_steps", "blur_sigma", "log_sigma",
            "lambda_struct", "alpha_fast", "gate_hidden", "gate_lr", "gate_epochs", "gate_batch", "beta_clinical",
            "beta_sparsity", "beta_cost", "faith_fractions", "faith_soft_temp", "tertile_alphas", "mass_top_fraction",
            "curve_step", "stability_frames", "stability_copies", "stability_noise_sigma", "stability_contrast",
            "bootstrap_resamples", "seeds_classifier", "seeds_explain"]
    d = cfg.as_dict()
    df = pd.DataFrame([{"parameter": k, "value": str(d[k])} for k in keys])
    write_table(rdir, "table_hyperparameters", df)
    return df


# ------------------------------------------------------------------ figures
def figures(cfg, rdir, frames_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})
    sums = [load_json(p) for p in sorted((cfg.out / "results" / "explain").glob("seed*_summary.json"))]
    if sums:
        fr = np.array(sums[0]["curve_fractions"]) * 100
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, kind in zip(axes, ("deletion", "insertion")):
            for c, m in zip(COLORS, PRIMARY_ORDER):
                curves = [np.array(s["curves"][m][kind]) for s in sums if m in s["curves"]]
                if curves:
                    mean = np.mean(curves, axis=0)
                    trapz = getattr(np, "trapezoid", None) or np.trapz
                    auc = float(np.mean([float(trapz(cv, fr / 100)) for cv in curves]))
                    ax.plot(fr, mean, color=c, lw=2, label=f"{m} (AUC {auc:.3f})")
            ax.set_xlabel(f"Pixels {'removed' if kind == 'deletion' else 'inserted'} (%)")
            ax.set_ylabel("p(polyp | x)")
            ax.set_title(f"({'a' if kind == 'deletion' else 'b'}) {kind.capitalize()}")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(rdir / f"fig_deletion_insertion.{ext}", dpi=200)
        plt.close(fig)

    if frames_df is not None:
        s0 = frames_df[(frames_df.method == "CAX-Gate") & (frames_df.seed == frames_df.seed.min())]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for t, c, lab in zip(range(3), COLORS, ("low", "medium", "high")):
            sub = s0[s0.tier == t]
            ax.scatter(sub.composite, sub.alpha, s=14, color=c, alpha=0.7, label=f"{lab} uncertainty")
        if len(s0) > 1:
            k, b = np.polyfit(s0.composite, s0.alpha, 1)
            xs = np.linspace(s0.composite.min(), s0.composite.max(), 50)
            ax.plot(xs, k * xs + b, color="#333333", lw=1.5, label="least-squares fit")
        ax.set_xlabel("Composite uncertainty $U_{ale}+U_{epi}$")
        ax.set_ylabel(r"$\alpha_{global}$")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(rdir / f"fig_alpha_vs_uncertainty.{ext}", dpi=200)
        plt.close(fig)

    q = cfg.out / "results" / "explain" / "qualitative.npz"
    if q.exists():
        z = np.load(q)
        cols = [("images", "Input"), ("masks", "Expert mask"), ("LIME", "LIME"), ("Partition_SHAP", "Partition SHAP"),
                ("Static_fusion", "Static fusion"), ("CAX-Gate", "CAX-Gate")]
        fig, axes = plt.subplots(3, 6, figsize=(13, 6.6))
        for r in range(3):
            for c, (key, title) in enumerate(cols):
                ax = axes[r, c]
                if key == "images":
                    ax.imshow(z[key][r])
                elif key == "masks":
                    ax.imshow(z[key][r], cmap="gray")
                else:
                    ax.imshow(z["images"][r])
                    ax.imshow(z[key][r], cmap="jet", alpha=0.45, vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=11)
                if c == 0:
                    ax.set_ylabel(["Low", "Medium", "High"][r] + " uncertainty", fontsize=11)
                if key == "CAX-Gate":
                    ax.set_xlabel(rf"$\alpha$ = {z['alpha'][r]:.2f}", fontsize=10)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(rdir / f"fig_qualitative.{ext}", dpi=200)
        plt.close(fig)


def environment():
    out = {}
    for pkg in ("torch", "torchvision", "lime", "shap", "scikit-learn", "scikit-image", "numpy", "imagehash"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = get_config(args.smoke)
    rdir = cfg.sub("results", "report")
    summary = {"config": cfg.as_dict(), "environment": environment()}
    ds = cfg.out / "data_summary.json"
    if ds.exists():
        summary["data"] = load_json(ds)
    calib = [load_json(p) | {"file": p.name} for p in sorted((cfg.out / "calibration").rglob("*.json"))]
    summary["calibration"] = calib

    summary["classification"] = classification_table(cfg, rdir).to_dict("records")
    frames = load_frames(cfg)
    if frames is not None:
        clusters = load_clusters(cfg, frames.frame.nunique())
        summary["video_level_bootstrap"] = clusters is not None
        prim = method_summary(cfg, frames, PRIMARY_ORDER, "Static fusion", clusters, len(PRIMARY_ORDER) - 1)
        write_table(rdir, "table_primary", prim)
        abl = method_summary(cfg, frames, ABLATION_ORDER, "CAX-Gate", clusters, len(ABLATION_ORDER) - 1)
        write_table(rdir, "table_ablation", abl)
        tiers = tier_table(frames)
        write_table(rdir, "table_tiers", tiers)
        summary.update({"primary": prim.to_dict("records"), "ablation": abl.to_dict("records"),
                        "tiers": tiers.to_dict("records")})
    alpha, extra = alpha_table(cfg)
    if alpha is not None:
        write_table(rdir, "table_alpha", alpha)
        summary.update({"alpha_by_tier": alpha.to_dict("records"), "gating": extra})
    stab = stability_table(cfg)
    if stab is not None:
        write_table(rdir, "table_stability", stab)
        summary["stability"] = stab.to_dict("records")
    lat, hw = latency_table(cfg)
    if lat is not None:
        write_table(rdir, "table_latency", lat)
        summary.update({"latency": lat.to_dict("records"), "hardware": hw})
    hyperparameter_table(cfg, rdir)
    figures(cfg, rdir, frames)
    save_json(rdir / "summary.json", summary)
    zip_path = shutil.make_archive(str(cfg.out / "results_report"), "zip", rdir)
    print(f"Report written to {rdir}\nSend this file back: {zip_path}")


if __name__ == "__main__":
    main()
