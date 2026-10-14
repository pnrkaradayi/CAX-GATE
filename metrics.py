"""Attribution-quality metrics."""
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2.0 renamed trapz


def pointing_game(sal: np.ndarray, mask: np.ndarray) -> int:
    """1 if the maximum of the saliency map lies inside the expert mask (strict, no tolerance)."""
    return int(mask.flat[int(np.argmax(sal))])


def mass_in_mask(sal: np.ndarray, mask: np.ndarray, top_fraction: float) -> float:
    """Share of the attribution energy of the top `top_fraction` pixels that lies inside the mask."""
    flat = sal.ravel()
    k = max(1, int(round(top_fraction * flat.size)))
    idx = np.argpartition(-flat, k - 1)[:k]
    energy = flat[idx]
    total = energy.sum()
    if total <= 0:
        return 0.0
    return float(energy[mask.ravel()[idx]].sum() / total)


def deletion_insertion(fn, img01, sal, label, base, step):
    """Deletion and insertion curves (Petsiuk et al., 2018).

    Pixels are ordered by decreasing attribution. Deletion replaces the top fraction of pixels by
    `base` (a blurred copy); insertion starts from `base` and restores the top fraction.
    Returns (fractions, deletion curve, insertion curve, deletion AUC, insertion AUC).
    """
    h, w = sal.shape
    order = np.argsort(-sal.ravel(), kind="stable")
    fractions = np.round(np.arange(0.0, 1.0 + 1e-9, step), 6)
    del_imgs, ins_imgs = [], []
    for f in fractions:
        sel = np.zeros(h * w, dtype=bool)
        sel[order[: int(round(f * h * w))]] = True
        sel = sel.reshape(h, w, 1)
        del_imgs.append(np.where(sel, base, img01))
        ins_imgs.append(np.where(sel, img01, base))
    probs = fn(np.stack(del_imgs + ins_imgs).astype(np.float32))[:, int(label)]
    d, i = probs[: len(fractions)], probs[len(fractions):]
    return fractions, d, i, float(_trapz(d, fractions)), float(_trapz(i, fractions))


def agreement(sal: np.ndarray, ref: np.ndarray) -> dict:
    """Secondary diagnostic: agreement with the reference map (both in [0, 1])."""
    from skimage.metrics import structural_similarity
    diff = sal - ref
    l2 = 1.0 - float(np.sqrt(np.mean(diff ** 2)))           # 1 - RMS difference, in [0, 1]
    denom = np.linalg.norm(sal) * np.linalg.norm(ref)
    cos = float(np.dot(sal.ravel(), ref.ravel()) / denom) if denom > 0 else 0.0
    ssim = float(structural_similarity(sal, ref, data_range=1.0))
    return {"l2_fidelity": l2, "cosine": cos, "ssim": ssim}
