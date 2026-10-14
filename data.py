"""Datasets, image cache, near-duplicate grouping and train/validation/test splits.

Run once before training:
    python -m caxgate.data            # builds the cache, groups and splits, prints a summary
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CLASSES, EXPECTED_COUNTS, get_config

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ------------------------------------------------------------------ file discovery
def _images_in(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def find_class_dirs(hk_dir: Path) -> dict:
    """Map each of the 11 class names to its folder inside labeled-images/ (searched recursively)."""
    root = hk_dir / "labeled-images"
    if not root.exists():
        root = hk_dir
    found = {}
    for d in root.rglob("*"):
        if d.is_dir() and d.name in CLASSES:
            found[d.name] = d
    missing = [c for c in CLASSES if c not in found]
    if missing:
        raise FileNotFoundError(f"Hyper-Kvasir class folders not found under {root}: {missing}")
    return found


def _find_subdir(root: Path, keywords) -> Path:
    for d in sorted(root.rglob("*")):
        name = d.name.lower().replace("_", " ").replace("-", " ")
        if d.is_dir() and any(k in name for k in keywords) and _images_in(d):
            return d
    raise FileNotFoundError(f"No folder matching {keywords} with images under {root}")


def find_cvc(cvc_dir: Path):
    """Return matched (image_path, mask_path) pairs of CVC-ClinicDB (tif or png releases)."""
    img_dir = _find_subdir(cvc_dir, ["original"])
    mask_dir = _find_subdir(cvc_dir, ["ground truth", "groundtruth", "mask"])
    masks = {p.stem: p for p in _images_in(mask_dir)}
    pairs = [(p, masks[p.stem]) for p in _images_in(img_dir) if p.stem in masks]
    if not pairs:
        raise FileNotFoundError("No CVC-ClinicDB image/mask pairs with matching file names")
    pairs.sort(key=lambda t: (len(t[0].stem), t[0].stem))  # 1, 2, ..., 612
    return pairs


def find_segmented(hk_dir: Path):
    """Return (image_path, mask_path) pairs of the Hyper-Kvasir segmented polyp images."""
    seg = hk_dir / "segmented-images"
    img_dir, mask_dir = seg / "images", seg / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Expected {img_dir} and {mask_dir}")
    masks = {p.stem: p for p in _images_in(mask_dir)}
    return [(p, masks[p.stem]) for p in _images_in(img_dir) if p.stem in masks]


# ------------------------------------------------------------------ loading
def load_image(path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_mask(path, size: int) -> np.ndarray:
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(m) > 127


# ------------------------------------------------------------------ near-duplicate grouping
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash64(img: np.ndarray) -> np.uint64:
    import imagehash
    h = imagehash.phash(Image.fromarray(img), hash_size=8)
    return np.uint64(int(str(h), 16))


def hamming_to_all(h: np.uint64, others: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(others, h).view(np.uint8).reshape(-1, 8)
    return _POPCOUNT[x].sum(axis=1)


def group_near_duplicates(hashes: np.ndarray, max_ham: int) -> np.ndarray:
    """Union-find over pairs whose perceptual-hash Hamming distance is <= max_ham."""
    n = len(hashes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n - 1):
        close = np.nonzero(hamming_to_all(hashes[i], hashes[i + 1:]) <= max_ham)[0] + i + 1
        for j in close:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


# ------------------------------------------------------------------ splits
def _fold_assignment(labels, groups, k, seed):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    fold = np.empty(len(labels), dtype=int)
    if groups is None:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        it = splitter.split(np.zeros(len(labels)), labels)
    else:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        it = splitter.split(np.zeros(len(labels)), labels, groups)
    for f, (_, test_idx) in enumerate(it):
        fold[test_idx] = f
    return fold


def make_split(labels, groups, k, seed):
    fold = _fold_assignment(labels, groups, k, seed)
    n_test = max(1, round(0.2 * k))
    n_val = max(1, round(0.1 * k))
    test = np.nonzero(fold < n_test)[0]
    val = np.nonzero((fold >= n_test) & (fold < n_test + n_val))[0]
    train = np.nonzero(fold >= n_test + n_val)[0]
    return {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}


# ------------------------------------------------------------------ cache
def cache_dir(cfg) -> Path:
    return cfg.sub("cache")


def build_cache(cfg, verbose=True):
    """Resize every image once to image_size and store uint8 arrays (fast I/O on Colab)."""
    cdir = cache_dir(cfg)
    s = cfg.image_size

    if not (cdir / "hk_images.npy").exists():
        class_dirs = find_class_dirs(cfg.hk_dir)
        paths, labels = [], []
        for ci, c in enumerate(CLASSES):
            for p in _images_in(class_dirs[c]):
                paths.append(p)
                labels.append(ci)
        imgs = np.lib.format.open_memmap(cdir / "hk_images.npy", mode="w+", dtype=np.uint8, shape=(len(paths), s, s, 3))
        for i, p in enumerate(paths):
            imgs[i] = load_image(p, s)
            if verbose and i % 1000 == 0:
                print(f"  Hyper-Kvasir {i}/{len(paths)}")
        imgs.flush()
        np.save(cdir / "hk_labels.npy", np.array(labels, dtype=np.int64))
        (cdir / "hk_paths.json").write_text(json.dumps([str(p) for p in paths]))

    if not (cdir / "cvc_images.npy").exists():
        pairs = find_cvc(cfg.cvc_dir)
        np.save(cdir / "cvc_images.npy", np.stack([load_image(a, s) for a, _ in pairs]))
        np.save(cdir / "cvc_masks.npy", np.stack([load_mask(b, s) for _, b in pairs]))
        (cdir / "cvc_names.json").write_text(json.dumps([a.stem for a, _ in pairs]))

    if not (cdir / "seg_images.npy").exists():
        pairs = find_segmented(cfg.hk_dir)
        np.save(cdir / "seg_images.npy", np.stack([load_image(a, s) for a, _ in pairs]))
        np.save(cdir / "seg_masks.npy", np.stack([load_mask(b, s) for _, b in pairs]))
        (cdir / "seg_names.json").write_text(json.dumps([a.stem for a, _ in pairs]))


def load_cache(cfg):
    cdir = cache_dir(cfg)
    return {
        "hk_images": np.load(cdir / "hk_images.npy", mmap_mode="r"),
        "hk_labels": np.load(cdir / "hk_labels.npy"),
        "cvc_images": np.load(cdir / "cvc_images.npy"),
        "cvc_masks": np.load(cdir / "cvc_masks.npy"),
        "cvc_names": json.loads((cdir / "cvc_names.json").read_text()),
        "seg_images": np.load(cdir / "seg_images.npy"),
        "seg_masks": np.load(cdir / "seg_masks.npy"),
    }


def prepare(cfg, verbose=True):
    """Build cache, near-duplicate groups, splits and the gate-training subset. Idempotent."""
    build_cache(cfg, verbose)
    cdir = cache_dir(cfg)
    data = load_cache(cfg)
    labels = data["hk_labels"]

    if not (cdir / "hk_hashes.npy").exists():
        np.save(cdir / "hk_hashes.npy", np.array([phash64(im) for im in data["hk_images"]], dtype=np.uint64))
        np.save(cdir / "seg_hashes.npy", np.array([phash64(im) for im in data["seg_images"]], dtype=np.uint64))
    hk_h = np.load(cdir / "hk_hashes.npy")
    seg_h = np.load(cdir / "seg_hashes.npy")

    split_file = cfg.out / "splits.json"
    if not split_file.exists():
        groups = group_near_duplicates(hk_h, cfg.phash_max_hamming)
        splits = {
            "grouped": make_split(labels, groups, cfg.split_folds, cfg.split_seed),
            "image_level": make_split(labels, None, cfg.split_folds, cfg.split_seed),
            "n_groups": int(groups.max() + 1),
        }
        # Gate training uses segmented polyp images that are not near duplicates of any image
        # in the classifier's training split (their context vectors would be over-confident).
        train_set = set(splits["grouped"]["train"])
        train_h = hk_h[sorted(train_set)]
        keep = [i for i, h in enumerate(seg_h)
                if len(train_h) == 0 or hamming_to_all(h, train_h).min() > cfg.phash_max_hamming]
        rng = random.Random(cfg.split_seed)
        rng.shuffle(keep)
        n_val = max(1, int(round(cfg.gate_val_fraction * len(keep))))
        splits["gate"] = {"val": sorted(keep[:n_val]), "train": sorted(keep[n_val:])}
        split_file.write_text(json.dumps(splits))
    splits = json.loads(split_file.read_text())

    summary = {
        "class_counts": {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)},
        "total_images": int(len(labels)),
        "near_duplicate_groups": splits["n_groups"],
        "grouped_split_sizes": {k: len(v) for k, v in splits["grouped"].items()},
        "image_level_split_sizes": {k: len(v) for k, v in splits["image_level"].items()},
        "cvc_frames": int(len(data["cvc_images"])),
        "segmented_images": int(len(data["seg_images"])),
        "gate_images": {k: len(v) for k, v in splits["gate"].items()},
    }
    (cfg.out / "data_summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(json.dumps(summary, indent=2))
        if not cfg.smoke:
            diff = {c: (summary["class_counts"][c], EXPECTED_COUNTS[c]) for c in CLASSES
                    if summary["class_counts"][c] != EXPECTED_COUNTS[c]}
            print("Class counts match the official release." if not diff
                  else f"WARNING: counts differ from the official release (found, expected): {diff}")
    return data, splits


# ------------------------------------------------------------------ torch datasets
class ImageDataset:
    """Images as float tensors in [0, 1] (normalisation happens inside the model wrapper)."""

    def __init__(self, images, labels, indices, cfg, augment=False):
        self.images, self.labels, self.idx = images, labels, np.asarray(indices)
        self.cfg, self.augment = cfg, augment

    def __len__(self):
        return len(self.idx)

    def _augment(self, img):
        import torch
        import torchvision.transforms.functional as TF
        cfg = self.cfg
        if random.random() < cfg.aug_clahe_prob:
            img = clahe(img, cfg.aug_clahe_clip)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        if random.random() < 0.5:
            x = TF.hflip(x)
        if random.random() < 0.5:
            x = TF.vflip(x)
        angle = random.uniform(-cfg.aug_rotation_deg, cfg.aug_rotation_deg)
        scale = random.uniform(1 - cfg.aug_scale, 1 + cfg.aug_scale)
        x = TF.affine(x, angle=angle, translate=[0, 0], scale=scale, shear=[0.0])
        x = x + cfg.aug_noise_sigma * torch.randn_like(x)
        return x.clamp(0, 1)

    def __getitem__(self, i):
        import torch
        j = int(self.idx[i])
        img = np.asarray(self.images[j])
        if self.augment:
            x = self._augment(img)
        else:
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return x, int(self.labels[j])


def clahe(img: np.ndarray, clip: float) -> np.ndarray:
    """CLAHE on the L channel (OpenCV clip limit semantics). Returns the input if OpenCV is missing."""
    try:
        import cv2
    except ImportError:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def channel_stats(images, indices, max_images=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.asarray(indices)
    if len(idx) > max_images:
        idx = rng.choice(idx, max_images, replace=False)
    x = np.stack([np.asarray(images[int(i)]) for i in idx]).astype(np.float64) / 255.0
    return x.mean(axis=(0, 1, 2)).tolist(), x.std(axis=(0, 1, 2)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = get_config(args.smoke)
    prepare(cfg)


if __name__ == "__main__":
    main()
