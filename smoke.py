"""End-to-end smoke test on a tiny synthetic dataset (CPU, a few minutes).

    python -m caxgate.smoke [--workdir /tmp/caxgate_smoke]

Creates fake Hyper-Kvasir / CVC-ClinicDB folders with random images, then runs every stage with
the tiny budgets of config.smoke_overrides. It only checks that the pipeline runs; the numbers
it produces are meaningless.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASSES


def blob_image(rng, size, hue):
    y, x = np.mgrid[0:size, 0:size]
    cy, cx, r = rng.uniform(0.3, 0.7) * size, rng.uniform(0.3, 0.7) * size, rng.uniform(0.1, 0.25) * size
    mask = (y - cy) ** 2 + (x - cx) ** 2 < r ** 2
    img = rng.uniform(80, 160, (size, size, 3))
    img[..., hue % 3] += 40
    img[mask] = [200, 60 + 10 * hue, 60]
    return np.clip(img, 0, 255).astype(np.uint8), mask


def make_data(root: Path, rng):
    hk = root / "hyper-kvasir"
    for ci, c in enumerate(CLASSES):
        d = hk / "labeled-images" / "lower-gi-tract" / "group" / c
        d.mkdir(parents=True, exist_ok=True)
        for k in range(8):
            Image.fromarray(blob_image(rng, 96, ci)[0]).save(d / f"{c}_{k}.jpg")
    seg_i, seg_m = hk / "segmented-images" / "images", hk / "segmented-images" / "masks"
    seg_i.mkdir(parents=True, exist_ok=True)
    seg_m.mkdir(parents=True, exist_ok=True)
    for k in range(12):
        img, m = blob_image(rng, 96, CLASSES.index("polyps"))
        Image.fromarray(img).save(seg_i / f"s{k}.jpg")
        Image.fromarray((m * 255).astype(np.uint8)).save(seg_m / f"s{k}.jpg")
    cvc_i, cvc_m = root / "CVC-ClinicDB" / "Original", root / "CVC-ClinicDB" / "Ground Truth"
    cvc_i.mkdir(parents=True, exist_ok=True)
    cvc_m.mkdir(parents=True, exist_ok=True)
    for k in range(1, 13):
        img, m = blob_image(rng, 96, CLASSES.index("polyps"))
        Image.fromarray(img).save(cvc_i / f"{k}.png")
        Image.fromarray((m * 255).astype(np.uint8)).save(cvc_m / f"{k}.png")


STAGES = [
    ["caxgate.data"],
    ["caxgate.train_backbone"],
    ["caxgate.precompute", "--set", "gate"],
    ["caxgate.precompute", "--set", "cvc"],
    ["caxgate.train_gate"],
    ["caxgate.evaluate"],
    ["caxgate.stability"],
    ["caxgate.latency"],
    ["caxgate.report"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=str(Path.cwd() / "smoke_run"))
    args = ap.parse_args()
    work = Path(args.workdir)
    data, out = work / "data", work / "outputs"
    if not data.exists():
        make_data(data, np.random.default_rng(0))
    env = dict(os.environ, CAXGATE_DATA=str(data), CAXGATE_OUT=str(out), CAXGATE_SMOKE="1")
    for stage in STAGES:
        print(f"\n=== {' '.join(stage)} ===", flush=True)
        r = subprocess.run([sys.executable, "-m", *stage, "--smoke"], env=env)
        if r.returncode != 0:
            sys.exit(f"Stage failed: {' '.join(stage)}")
    print(f"\nSmoke test passed. Outputs in {out}")


if __name__ == "__main__":
    main()
