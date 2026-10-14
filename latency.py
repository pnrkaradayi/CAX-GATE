"""Per-frame explanation latency (batch size 1, same GPU for every method).

    python -m caxgate.latency
Times cover the explanation only; the classifier's own prediction is not included. For CAX-Gate
they include the context vector (deterministic pass + MC dropout passes), the gate, the explainers,
the LoG mask and the blending. "CAX-Gate (routed)" applies the fast path whenever alpha >= alpha_fast,
which gives the average cost of the full pipeline on the evaluated frames.
"""
import argparse
import platform
import time

import numpy as np
import pandas as pd
import torch

from .config import CLASSES, POLYP_CLASS, get_config
from .data import prepare
from .explainers import ModelFn, gradcam_map, ig_map, lime_map, shap_map
from .fusion import blend_np, gamma_np, log_mask
from .models import checkpoint_path, load_classifier
from .train_gate import load_gate, predict_alpha
from .uncertainty import context_vectors
from .utils import load_tau, save_json

POLYP_IDX = CLASSES.index(POLYP_CLASS)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(cfg, data):
    seed = cfg.seeds_explain[0]
    model = load_classifier(checkpoint_path(cfg, "proposed", seed), len(CLASSES), cfg.device)
    tau = load_tau(cfg, "proposed", seed)
    fn = ModelFn(model, tau, cfg.device, batch=cfg.shap_batch)
    gate, ck = load_gate(cfg, seed, "full")

    def context_and_alpha(x):
        z = context_vectors(model, x[None], tau, cfg)["z"]
        return float(predict_alpha(gate, ck, z)[0])

    def cax(x, mode):
        a = context_and_alpha(x)
        L, _ = lime_map(fn, x, POLYP_IDX, cfg, seed=seed)
        if mode == "fast" or (mode == "routed" and a >= cfg.alpha_fast):
            return L
        S, _ = shap_map(fn, x, POLYP_IDX, cfg)
        return blend_np(L, S, gamma_np(a, log_mask(x, cfg.log_sigma), cfg.lambda_struct))

    methods = {
        "LIME": lambda x: lime_map(fn, x, POLYP_IDX, cfg, seed=seed),
        "Partition SHAP": lambda x: shap_map(fn, x, POLYP_IDX, cfg),
        "Static fusion": lambda x: 0.5 * lime_map(fn, x, POLYP_IDX, cfg, seed=seed)[0] + 0.5 * shap_map(fn, x, POLYP_IDX, cfg)[0],
        "Grad-CAM": lambda x: gradcam_map(model, x, POLYP_IDX, cfg.device),
        "Integrated Gradients": lambda x: ig_map(model, x, POLYP_IDX, cfg),
        "CAX-Gate (full blend)": lambda x: cax(x, "full"),
        "CAX-Gate (fast path)": lambda x: cax(x, "fast"),
        "CAX-Gate (routed)": lambda x: cax(x, "routed"),
    }
    n = min(cfg.latency_frames + cfg.latency_warmup, len(data["cvc_images"]))
    frames = [data["cvc_images"][i].astype(np.float32) / 255.0 for i in range(n)]
    rows = []
    for name, f in methods.items():
        for j, x in enumerate(frames):
            _sync()
            t0 = time.perf_counter()
            f(x)
            _sync()
            dt = time.perf_counter() - t0
            if j >= cfg.latency_warmup:
                rows.append({"method": name, "frame": j, "seconds": dt})
        print(f"latency {name}: {np.mean([r['seconds'] for r in rows if r['method'] == name]) * 1000:.1f} ms")
    df = pd.DataFrame(rows)
    out = cfg.sub("results", "latency")
    df.to_csv(out / "latency.csv", index=False)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    save_json(out / "hardware.json", {
        "gpu": gpu, "cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
        "torch": torch.__version__, "batch_size": 1, "frames": cfg.latency_frames,
        "lime_samples": cfg.lime_samples, "shap_max_evals": cfg.shap_max_evals, "mc_passes": cfg.mc_passes})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = get_config(args.smoke)
    data, _ = prepare(cfg, verbose=False)
    run(cfg, data)


if __name__ == "__main__":
    main()
