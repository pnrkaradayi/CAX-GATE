"""Attribution methods. Every map is returned on the 224x224 pixel grid and normalised the same way:
negative attributions are clipped to zero and the map is min-max scaled to [0, 1].

  LIME            lime_image, quickshift superpixels, mean-colour hiding, cfg.lime_samples samples
  Partition SHAP  shap Partition explainer with a blur masker, cfg.shap_max_evals evaluations
  Reference       KernelSHAP over SLIC superpixels with blurred absent segments (secondary diagnostic)
  Grad-CAM        last convolutional block of the backbone
  IG              Integrated Gradients from an all-black baseline, cfg.ig_steps steps
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def normalise_map(a) -> np.ndarray:
    a = np.clip(np.asarray(a, dtype=np.float64), 0, None)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def blurred(img01: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(img01, sigma=(sigma, sigma, 0))


class ModelFn:
    """Callable used by LIME and SHAP: (N,H,W,3) float images in [0,1] -> (N,C) probabilities."""

    def __init__(self, model, tau, device, batch=64):
        self.model, self.tau, self.device, self.batch = model.eval(), tau, device, batch

    @torch.no_grad()
    def __call__(self, imgs):
        imgs = np.asarray(imgs)
        out = []
        for i in range(0, len(imgs), self.batch):
            x = torch.from_numpy(np.ascontiguousarray(imgs[i:i + self.batch], dtype=np.float32))
            x = x.permute(0, 3, 1, 2).to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                logits = self.model(x)
            out.append(F.softmax(logits.float() / self.tau, dim=1).cpu().numpy())
        return np.concatenate(out)


def _timed(f):
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        m = f(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return m, time.perf_counter() - t0
    return wrapper


@_timed
def lime_map(fn: ModelFn, img01, label, cfg, seed=0):
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    seg = SegmentationAlgorithm("quickshift", kernel_size=cfg.lime_qs_kernel_size,
                                max_dist=cfg.lime_qs_max_dist, ratio=cfg.lime_qs_ratio, random_seed=seed)
    explainer = lime_image.LimeImageExplainer(kernel_width=cfg.lime_kernel_width, random_state=seed)
    exp = explainer.explain_instance(img01.astype(np.float64), fn, labels=(int(label),), top_labels=None,
                                     hide_color=None, num_samples=cfg.lime_samples, batch_size=cfg.shap_batch,
                                     segmentation_fn=seg, random_seed=seed, progress_bar=False)
    w = np.zeros(exp.segments.max() + 1)
    for s, v in exp.local_exp[int(label)]:
        w[s] = v
    return normalise_map(w[exp.segments])


@_timed
def shap_map(fn: ModelFn, img01, label, cfg, max_evals=None):
    import shap
    masker = shap.maskers.Image(cfg.shap_blur, img01.shape)
    explainer = shap.explainers.Partition(fn, masker)
    evals = max_evals or cfg.shap_max_evals
    try:
        sv = explainer(img01[None], max_evals=evals, batch_size=cfg.shap_batch,
                       outputs=np.array([int(label)]), silent=True)
        vals = np.asarray(sv.values)[0]
        vals = vals[..., 0] if vals.ndim == 4 else vals
    except Exception:  # older/newer shap versions: explain all outputs and pick the label
        sv = explainer(img01[None], max_evals=evals, batch_size=cfg.shap_batch, silent=True)
        vals = np.asarray(sv.values)[0][..., int(label)]
    return normalise_map(vals.sum(axis=-1))


@_timed
def reference_map(fn: ModelFn, img01, label, cfg, seed=0):
    """KernelSHAP over SLIC superpixels; absent superpixels are replaced by a blurred copy."""
    import shap
    from skimage.segmentation import slic
    segs = slic(img01, n_segments=cfg.ref_slic_segments, compactness=cfg.ref_slic_compactness,
                start_label=0, channel_axis=-1)
    n_seg = int(segs.max()) + 1
    base = blurred(img01, cfg.blur_sigma)

    def f(zs):
        zs = np.atleast_2d(zs)
        out = np.empty(len(zs))
        for i in range(0, len(zs), fn.batch):
            chunk = zs[i:i + fn.batch]
            imgs = np.stack([np.where((z[segs] > 0.5)[..., None], img01, base) for z in chunk])
            out[i:i + len(chunk)] = fn(imgs)[:, int(label)]
        return out

    np.random.seed(seed)
    explainer = shap.KernelExplainer(f, np.zeros((1, n_seg)))
    phi = np.asarray(explainer.shap_values(np.ones((1, n_seg)), nsamples=cfg.ref_nsamples, silent=True))
    phi = phi.reshape(-1)[:n_seg]
    return normalise_map(phi[segs])


@_timed
def gradcam_map(model, img01, label, device):
    layer = model.target_layer()
    store = {}

    def fwd_hook(_m, _i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd_hook)
    try:
        model.eval()
        x = torch.from_numpy(img01.astype(np.float32)).permute(2, 0, 1)[None].to(device)
        model.zero_grad(set_to_none=True)
        model(x)[0, int(label)].backward()
        a, g = store["a"], store["g"]
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    finally:
        h.remove()
    return normalise_map(cam[0, 0].detach().cpu().numpy())


@_timed
def ig_map(model, img01, label, cfg, chunk=16):
    device = cfg.device
    model.eval()
    x = torch.from_numpy(img01.astype(np.float32)).permute(2, 0, 1)[None].to(device)
    baseline = torch.zeros_like(x)
    alphas = (torch.arange(cfg.ig_steps, dtype=torch.float32) + 0.5) / cfg.ig_steps  # midpoint rule
    total = torch.zeros_like(x)
    for i in range(0, len(alphas), chunk):
        a = alphas[i:i + chunk].to(device).view(-1, 1, 1, 1)
        xs = (baseline + a * (x - baseline)).requires_grad_(True)
        out = model(xs)[:, int(label)].sum()
        total += torch.autograd.grad(out, xs)[0].sum(0, keepdim=True)
    attr = (x - baseline) * total / cfg.ig_steps
    return normalise_map(attr[0].sum(0).detach().cpu().numpy())
