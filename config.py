"""All paths and hyperparameters in one place.

Every value used in the experiments is defined here, so that the hyperparameter table of the
manuscript can be generated from this file (see report.py). Paths can be overridden with the
environment variables CAXGATE_DATA and CAXGATE_OUT.
"""
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# The 11 lower-GI classes used in the study (folder names of the Hyper-Kvasir release).
CLASSES = (
    "cecum",
    "retroflex-rectum",
    "polyps",
    "ulcerative-colitis-grade-1",
    "ulcerative-colitis-grade-2",
    "ulcerative-colitis-grade-3",
    "bbps-0-1",
    "bbps-2-3",
    "impacted-stool",
    "dyed-lifted-polyps",
    "dyed-resection-margins",
)

# Image counts of the official Hyper-Kvasir release, used as a sanity check of the download.
EXPECTED_COUNTS = {
    "cecum": 1009, "retroflex-rectum": 391, "polyps": 1028,
    "ulcerative-colitis-grade-1": 201, "ulcerative-colitis-grade-2": 443,
    "ulcerative-colitis-grade-3": 133, "bbps-0-1": 646, "bbps-2-3": 1148,
    "impacted-stool": 131, "dyed-lifted-polyps": 1002, "dyed-resection-margins": 989,
}

# Classes whose probability mass forms the model-derived risk score R(x).
NEOPLASIA_CLASSES = ("polyps", "dyed-lifted-polyps")
POLYP_CLASS = "polyps"


@dataclass
class Config:
    # ---------------------------------------------------------------- paths
    data_root: str = os.environ.get("CAXGATE_DATA", "/content/drive/MyDrive/caxgate/data")
    out_root: str = os.environ.get("CAXGATE_OUT", "/content/drive/MyDrive/caxgate/outputs")
    hyperkvasir_dirname: str = "hyper-kvasir"   # contains labeled-images/ and segmented-images/
    cvc_dirname: str = "CVC-ClinicDB"            # contains Original/ and Ground Truth/ (any case, tif or png)
    cvc_sequences_csv: str = "cvc_sequences.csv"  # optional: frame,sequence (enables video-level bootstrap)

    # ---------------------------------------------------------------- data and splits
    image_size: int = 224
    split_seed: int = 42                 # one fixed split, shared by all training seeds
    split_folds: int = 10                # StratifiedGroupKFold: 2 folds test, 1 val, 7 train
    phash_max_hamming: int = 8           # frames closer than this are grouped (near duplicates)
    also_image_level_split: bool = True  # also train the proposed backbone on an image-level split

    # ---------------------------------------------------------------- backbones
    backbones: tuple = ("proposed", "resnet50", "efficientnet_b0")
    proposed_arch: str = "convnext_tiny"  # "Proposed CNN" = ConvNeXt-Tiny + MC-dropout head
    pretrained: bool = True               # ImageNet-1k weights from torchvision
    head_dropout: float = 0.2
    seeds_classifier: tuple = (0, 1, 2, 3, 4)
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 40
    patience: int = 10                    # early stopping on validation loss
    num_workers: int = 2
    # augmentation (training split only)
    aug_rotation_deg: float = 10.0
    aug_scale: float = 0.10
    aug_clahe_clip: float = 2.0
    aug_clahe_prob: float = 0.5
    aug_noise_sigma: float = 0.02
    # classical baselines on HOG features
    hog_pixels_per_cell: int = 16
    hog_cells_per_block: int = 2
    rf_trees: int = 500

    # ---------------------------------------------------------------- uncertainty and context
    mc_passes: int = 20                   # T, Monte Carlo dropout forward passes
    gamma_risk: float = 1.0               # multiplier of the model-derived risk score
    ece_bins: int = 15

    # ---------------------------------------------------------------- explainers
    seeds_explain: tuple = (0, 1, 2)      # classifier seeds used for the explanation experiments
    lime_samples: int = 1000
    lime_kernel_width: float = 0.25
    lime_qs_kernel_size: int = 4          # quickshift (lime_image defaults)
    lime_qs_max_dist: int = 200
    lime_qs_ratio: float = 0.2
    shap_max_evals: int = 2000            # Partition SHAP inside CAX-Gate
    shap_blur: str = "blur(128,128)"
    shap_batch: int = 64
    ref_slic_segments: int = 100          # reference: KernelSHAP over SLIC superpixels
    ref_slic_compactness: float = 10.0
    ref_nsamples: int = 4096
    ig_steps: int = 50                    # Integrated Gradients, all-black baseline
    blur_sigma: float = 10.0              # Gaussian blur used as "removed" content (deletion, insertion, KernelSHAP)

    # ---------------------------------------------------------------- fusion and gate
    log_sigma: float = 2.0                # Laplacian-of-Gaussian scale for M_spatial
    lambda_struct: float = 0.5
    alpha_fast: float = 0.80
    gate_hidden: int = 16
    gate_lr: float = 1e-3
    gate_epochs: int = 40
    gate_batch: int = 8
    gate_patience: int = 8
    gate_val_fraction: float = 0.2
    beta_clinical: float = 1.0            # beta_1
    beta_sparsity: float = 0.1            # beta_2
    beta_cost: float = 0.1                # beta_3
    faith_fractions: tuple = (0.1, 0.2, 0.3, 0.4, 0.5)   # top-k fractions in L_faith
    faith_soft_temp: float = 0.05         # temperature of the sigmoid relaxation of top-k masking
    # fixed parameter-free rules (baselines for the learned gate)
    tertile_alphas: tuple = (0.75, 0.50, 0.25)

    # ---------------------------------------------------------------- evaluation
    mass_top_fraction: float = 0.30
    curve_step: float = 0.02              # deletion/insertion step (2% of the pixels)
    stability_frames: int = 100
    stability_copies: int = 10
    stability_noise_sigma: float = 0.05
    stability_contrast: float = 0.10
    latency_frames: int = 200
    latency_warmup: int = 5
    bootstrap_resamples: int = 2000
    alpha_level: float = 0.05

    # ---------------------------------------------------------------- run control
    device: str = "cuda"
    smoke: bool = False                   # tiny budgets for a quick end-to-end test
    extra: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- derived paths
    @property
    def hk_dir(self) -> Path:
        return Path(self.data_root) / self.hyperkvasir_dirname

    @property
    def cvc_dir(self) -> Path:
        return Path(self.data_root) / self.cvc_dirname

    @property
    def out(self) -> Path:
        p = Path(self.out_root)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def sub(self, *parts) -> Path:
        p = self.out.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def as_dict(self) -> dict:
        return asdict(self)


def smoke_overrides(cfg: Config) -> Config:
    """Tiny budgets so that every stage runs in minutes on a CPU."""
    cfg.smoke = True
    cfg.pretrained = False
    cfg.seeds_classifier = (0, 1)
    cfg.seeds_explain = (0,)
    cfg.max_epochs = 2
    cfg.patience = 2
    cfg.batch_size = 8
    cfg.num_workers = 0
    cfg.mc_passes = 3
    cfg.lime_samples = 40
    cfg.shap_max_evals = 60
    cfg.shap_batch = 20
    cfg.ref_slic_segments = 20
    cfg.ref_nsamples = 64
    cfg.ig_steps = 4
    cfg.gate_epochs = 2
    cfg.stability_frames = 3
    cfg.stability_copies = 3
    cfg.latency_frames = 3
    cfg.latency_warmup = 1
    cfg.bootstrap_resamples = 50
    cfg.rf_trees = 20
    cfg.split_folds = 5
    return cfg


def get_config(smoke: bool = False) -> Config:
    cfg = Config()
    if smoke or os.environ.get("CAXGATE_SMOKE") == "1":
        cfg = smoke_overrides(cfg)
    try:
        import torch
        if cfg.device == "cuda" and not torch.cuda.is_available():
            cfg.device = "cpu"
    except ImportError:
        cfg.device = "cpu"
    return cfg
