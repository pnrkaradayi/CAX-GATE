"""Temperature scaling, Monte Carlo dropout and the five-dimensional context vector z(x)."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .config import CLASSES, NEOPLASIA_CLASSES
from .models import enable_mc_dropout

NEO_IDX = [CLASSES.index(c) for c in NEOPLASIA_CLASSES]
CONTEXT_NAMES = ("conf", "u_aleatoric", "u_epistemic", "v_texture", "r_clinical")


def to_tensor(images) -> torch.Tensor:
    """uint8 (N,H,W,3) or float [0,1] (N,H,W,3) numpy -> float tensor (N,3,H,W) in [0,1]."""
    x = np.asarray(images)
    is_uint8 = x.dtype == np.uint8
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float()
    return x / 255.0 if is_uint8 else x


@torch.no_grad()
def predict_logits(model, images, device, batch=64) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        out.append(model(to_tensor(images[i:i + batch]).to(device)).float().cpu())
    return torch.cat(out)


def fit_temperature(logits: torch.Tensor, labels) -> float:
    """Single temperature minimising the validation negative log-likelihood (Guo et al., 2017)."""
    labels = torch.as_tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def sobel_energy(images01: np.ndarray) -> np.ndarray:
    """Mean Sobel gradient magnitude of the luminance x_g = mean over RGB, per image."""
    out = []
    for img in images01:
        g = img.mean(axis=2)
        gx, gy = ndimage.sobel(g, axis=1), ndimage.sobel(g, axis=0)
        out.append(np.sqrt(gx ** 2 + gy ** 2).mean())
    return np.array(out, dtype=np.float32)


@torch.no_grad()
def context_vectors(model, images, tau: float, cfg, batch=32) -> dict:
    """Return probabilities, prediction and the five context components for each image.

    conf        max_c p(c|x)                             (temperature-scaled softmax)
    u_aleatoric normalised Shannon entropy H(p)/ln C
    u_epistemic (1/T) sum_t ||p_t - mean_t p_t||^2       (MC dropout, T passes)
    v_texture   mean Sobel gradient energy of the luminance
    r_clinical  gamma * sum of p over the neoplasia classes
    """
    device = cfg.device
    C = len(CLASSES)
    probs, epi, tex = [], [], []
    for i in range(0, len(images), batch):
        xb = to_tensor(images[i:i + batch])
        tex.append(sobel_energy(xb.permute(0, 2, 3, 1).numpy()))
        x = xb.to(device)
        model.eval()
        probs.append(F.softmax(model(x).float() / tau, dim=1).cpu())
        enable_mc_dropout(model)
        mc = torch.stack([F.softmax(model(x).float() / tau, dim=1) for _ in range(cfg.mc_passes)])
        epi.append(((mc - mc.mean(0, keepdim=True)) ** 2).sum(-1).mean(0).cpu())
        model.eval()
    p = torch.cat(probs).numpy()
    u_epi = torch.cat(epi).numpy()
    u_ale = -(p * np.log(p + 1e-12)).sum(1) / np.log(C)
    tex = np.concatenate(tex)
    risk = cfg.gamma_risk * p[:, NEO_IDX].sum(1)
    z = np.stack([p.max(1), u_ale, u_epi, tex, risk], axis=1).astype(np.float32)
    return {"probs": p.astype(np.float32), "pred": p.argmax(1), "z": z,
            "u_aleatoric": u_ale.astype(np.float32), "u_epistemic": u_epi.astype(np.float32),
            "composite": (u_ale + u_epi).astype(np.float32)}
