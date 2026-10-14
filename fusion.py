"""Spatial prior, convex blending, the gating network and the parameter-free baseline rules."""
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from .explainers import normalise_map

# Gate variants: which context components are used and whether the LoG modulation is active.
# Indices refer to (conf, u_aleatoric, u_epistemic, v_texture, r_clinical).
GATE_VARIANTS = {
    "full": {"inputs": [0, 1, 2, 3, 4], "use_log": True},
    "entropy_only": {"inputs": [1], "use_log": True},
    "posterior": {"inputs": [0, 1, 4], "use_log": True},
    "posterior_epistemic": {"inputs": [0, 1, 2, 4], "use_log": True},
    "full_no_log": {"inputs": [0, 1, 2, 3, 4], "use_log": False},
}


def log_mask(img01: np.ndarray, sigma: float) -> np.ndarray:
    """M_spatial: min-max normalised |LoG| of the luminance."""
    g = img01.mean(axis=2)
    return normalise_map(np.abs(ndimage.gaussian_laplace(g, sigma)))


def gamma_np(alpha: float, m_spatial: np.ndarray, lam: float) -> np.ndarray:
    return np.clip(alpha * (1.0 - lam * m_spatial), 0.0, 1.0)


def blend_np(lime_m, shap_m, gamma):
    return gamma * lime_m + (1.0 - gamma) * shap_m


def blend_torch(alpha, lime_m, shap_m, m_spatial, lam):
    """alpha (N,), maps (N,H,W) -> fused map (N,H,W); differentiable in alpha."""
    gamma = torch.clamp(alpha.view(-1, 1, 1) * (1.0 - lam * m_spatial), 0.0, 1.0)
    return gamma * lime_m + (1.0 - gamma) * shap_m


class Gate(nn.Module):
    """g_theta: z -> alpha = sigmoid(w2^T LayerNorm(GELU(W1 z + b1)) + b2), z standardised."""

    def __init__(self, in_dim, hidden, z_mean, z_std):
        super().__init__()
        self.register_buffer("z_mean", torch.as_tensor(z_mean, dtype=torch.float32))
        self.register_buffer("z_std", torch.as_tensor(z_std, dtype=torch.float32).clamp_min(1e-6))
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, z):
        h = self.ln(self.act(self.fc1((z - self.z_mean) / self.z_std)))
        return torch.sigmoid(self.fc2(h)).squeeze(-1)


def alpha_entropy_rule(u_ale: np.ndarray) -> np.ndarray:
    """Parameter-free rule (i): alpha = 1 - H(x)/log C."""
    return np.clip(1.0 - u_ale, 0.0, 1.0)


def alpha_tertile_rule(u_ale: np.ndarray, thresholds, values) -> np.ndarray:
    """Parameter-free rule (ii): fixed alpha per entropy tertile (low, medium, high)."""
    t1, t2 = thresholds
    return np.where(u_ale <= t1, values[0], np.where(u_ale <= t2, values[1], values[2])).astype(np.float32)
