"""Precompute context vectors and attribution maps (resumable; saves every few images).

    python -m caxgate.precompute --set gate     # segmented Hyper-Kvasir polyps used to train the gate
    python -m caxgate.precompute --set cvc      # CVC-ClinicDB evaluation frames

All maps explain the polyp class (the true class of every frame in both sets), using the
"Proposed CNN" of the grouped split for each seed in cfg.seeds_explain.
"""
import argparse
import time

import numpy as np

from .config import CLASSES, POLYP_CLASS, get_config
from .data import prepare
from .explainers import ModelFn, gradcam_map, ig_map, lime_map, reference_map, shap_map
from .fusion import log_mask
from .models import checkpoint_path, load_classifier
from .uncertainty import context_vectors
from .utils import load_tau

POLYP_IDX = CLASSES.index(POLYP_CLASS)
SAVE_EVERY = 25


def maps_path(cfg, which, seed):
    return cfg.sub("maps", which) / f"seed{seed}.npz"


def _load_partial(path):
    if path.exists():
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}
    return None


def _save(path, store):
    tmp = path.with_name(path.stem + "_tmp.npz")
    np.savez_compressed(tmp, **store)
    tmp.replace(path)


def run(cfg, which, seed):
    data, splits = prepare(cfg, verbose=False)
    if which == "gate":
        idx = np.array(sorted(splits["gate"]["train"] + splits["gate"]["val"]))
        images = data["seg_images"][idx]
    else:
        idx = np.arange(len(data["cvc_images"]))
        images = data["cvc_images"]
    n = len(idx)
    s = cfg.image_size
    path = maps_path(cfg, which, seed)
    store = _load_partial(path)
    model = load_classifier(checkpoint_path(cfg, "proposed", seed), len(CLASSES), cfg.device)
    tau = load_tau(cfg, "proposed", seed)
    fn = ModelFn(model, tau, cfg.device, batch=cfg.shap_batch)
    extra = which == "cvc"
    names = ["lime", "shap", "logm"] + (["ref", "gradcam", "ig"] if extra else [])

    if store is None:
        ctx = context_vectors(model, images, tau, cfg)
        store = {"idx": idx, "z": ctx["z"], "probs": ctx["probs"], "pred": ctx["pred"],
                 "u_aleatoric": ctx["u_aleatoric"], "u_epistemic": ctx["u_epistemic"],
                 "composite": ctx["composite"], "done": np.zeros(n, dtype=bool)}
        for k in names:
            store[k] = np.zeros((n, s, s), dtype=np.float16)
            store["t_" + k] = np.zeros(n, dtype=np.float32)

    t_start = time.time()
    for i in range(n):
        if store["done"][i]:
            continue
        img = images[i].astype(np.float32) / 255.0
        lm, t = lime_map(fn, img, POLYP_IDX, cfg, seed=seed)
        store["lime"][i], store["t_lime"][i] = lm, t
        sm, t = shap_map(fn, img, POLYP_IDX, cfg)
        store["shap"][i], store["t_shap"][i] = sm, t
        t0 = time.perf_counter()
        store["logm"][i] = log_mask(img, cfg.log_sigma)
        store["t_logm"][i] = time.perf_counter() - t0
        if extra:
            rm, t = reference_map(fn, img, POLYP_IDX, cfg, seed=seed)
            store["ref"][i], store["t_ref"][i] = rm, t
            gm, t = gradcam_map(model, img, POLYP_IDX, cfg.device)
            store["gradcam"][i], store["t_gradcam"][i] = gm, t
            im, t = ig_map(model, img, POLYP_IDX, cfg)
            store["ig"][i], store["t_ig"][i] = im, t
        store["done"][i] = True
        if (i + 1) % SAVE_EVERY == 0 or i == n - 1:
            _save(path, store)
            done = int(store["done"].sum())
            rate = (time.time() - t_start) / max(1, done)
            print(f"{which} seed {seed}: {done}/{n} images ({rate:.1f} s/image this session)")
    _save(path, store)


def load_maps(cfg, which, seed):
    d = np.load(maps_path(cfg, which, seed))
    out = {k: d[k] for k in d.files}
    if not out["done"].all():
        raise RuntimeError(f"maps/{which}/seed{seed} incomplete: run precompute again")
    for k in ("lime", "shap", "logm", "ref", "gradcam", "ig"):
        if k in out:
            out[k] = out[k].astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set", choices=["gate", "cvc"], required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = get_config(args.smoke)
    for seed in (args.seeds if args.seeds is not None else cfg.seeds_explain):
        run(cfg, args.set, seed)


if __name__ == "__main__":
    main()
