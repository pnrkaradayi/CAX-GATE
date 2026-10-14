"""Train and evaluate the classifiers (Table 3 of the manuscript).

    python -m caxgate.train_backbone             # all backbones, all seeds, resumable
    python -m caxgate.train_backbone --only proposed --seeds 0

For every deep backbone and seed: training with early stopping on the validation loss,
temperature scaling on the validation split, metrics on the internal test split, and polyp recall
on CVC-ClinicDB. HOG + PCA + RBF-SVM and HOG + random forest are the classical baselines.
"""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, roc_auc_score

from .config import CLASSES, POLYP_CLASS, get_config
from .data import ImageDataset, channel_stats, prepare
from .models import Classifier, arch_of, checkpoint_path, count_parameters, load_classifier, save_classifier
from .uncertainty import ece, fit_temperature, predict_logits
from .utils import calibration_path, load_json, save_json, seed_everything

POLYP_IDX = CLASSES.index(POLYP_CLASS)


def classification_metrics(probs, labels):
    pred = probs.argmax(1)
    out = {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "precision_macro": float(precision_score(labels, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, pred, average="macro", zero_division=0)),
    }
    try:
        out["auc_macro_ovr"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro",
                                                   labels=list(range(len(CLASSES)))))
    except ValueError:
        out["auc_macro_ovr"] = float("nan")
    return out


def result_path(cfg, name, seed, split):
    return cfg.sub("results", "classification", split) / f"{name}_seed{seed}.json"


def train_deep(cfg, data, splits, name, seed, split="grouped"):
    res_file = result_path(cfg, name, seed, split)
    if res_file.exists():
        print(f"[skip] {split}/{name} seed {seed}")
        return
    seed_everything(seed)
    device = cfg.device
    images, labels = data["hk_images"], data["hk_labels"]
    sp = splits[split]
    stats_file = cfg.out / f"channel_stats_{split}.json"
    if stats_file.exists():
        mean, std = load_json(stats_file)
    else:
        mean, std = channel_stats(images, sp["train"])
        save_json(stats_file, [mean, std])

    model = Classifier(arch_of(name, cfg), len(CLASSES), mean, std, cfg.pretrained, cfg.head_dropout).to(device)
    train_dl = torch.utils.data.DataLoader(ImageDataset(images, labels, sp["train"], cfg, augment=True),
                                           batch_size=cfg.batch_size, shuffle=True,
                                           num_workers=cfg.num_workers, drop_last=True)
    val_dl = torch.utils.data.DataLoader(ImageDataset(images, labels, sp["val"], cfg),
                                         batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    best, best_state, wait, history = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        model.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * len(y)
        sched.step()
        model.eval()
        va_loss, n = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                va_loss += F.cross_entropy(model(x).float(), y, reduction="sum").item()
                n += len(y)
        va_loss /= max(n, 1)
        history.append({"epoch": epoch + 1, "train_loss": tr_loss / max(len(train_dl.dataset), 1), "val_loss": va_loss})
        print(f"{split}/{name} seed {seed} epoch {epoch + 1}: val loss {va_loss:.4f}")
        if va_loss < best - 1e-4:
            best, best_state, wait = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    ck = checkpoint_path(cfg, name, seed, split)
    save_classifier(ck, model, mean, std, {"history": history})
    evaluate_deep(cfg, data, splits, name, seed, split, model, history, time.time() - t0)


def evaluate_deep(cfg, data, splits, name, seed, split, model=None, history=None, train_seconds=None):
    device = cfg.device
    if model is None:
        model = load_classifier(checkpoint_path(cfg, name, seed, split), len(CLASSES), device)
    images, labels = data["hk_images"], data["hk_labels"]
    sp = splits[split]
    val_idx, test_idx = np.array(sp["val"]), np.array(sp["test"])

    val_logits = predict_logits(model, images[val_idx], device)
    tau = fit_temperature(val_logits, labels[val_idx])
    val_p_raw = F.softmax(val_logits, 1).numpy()
    val_p = F.softmax(val_logits / tau, 1).numpy()
    save_json(calibration_path(cfg, name, seed, split), {
        "tau": tau, "ece_before": ece(val_p_raw, labels[val_idx], cfg.ece_bins),
        "ece_after": ece(val_p, labels[val_idx], cfg.ece_bins)})

    test_p = F.softmax(predict_logits(model, images[test_idx], device) / tau, 1).numpy()
    cvc_p = F.softmax(predict_logits(model, data["cvc_images"], device) / tau, 1).numpy()
    np.savez_compressed(cfg.sub("results", "classification", split) / f"{name}_seed{seed}_preds.npz",
                        test_idx=test_idx, test_probs=test_p, cvc_probs=cvc_p)
    res = {"model": name, "arch": model.arch, "seed": seed, "split": split, "tau": tau,
           "n_parameters": count_parameters(model), "train_seconds": train_seconds,
           "epochs": len(history) if history else None,
           "test": classification_metrics(test_p, labels[test_idx]), "n_test": int(len(test_idx)),
           "cvc_polyp_recall": float((cvc_p.argmax(1) == POLYP_IDX).mean()), "n_cvc": int(len(cvc_p))}
    save_json(result_path(cfg, name, seed, split), res)
    print(f"{split}/{name} seed {seed}: test acc {res['test']['accuracy']:.4f}, CVC polyp recall {res['cvc_polyp_recall']:.4f}")


def hog_features(cfg, images):
    from skimage.feature import hog
    return np.stack([hog(np.asarray(im).mean(axis=2), orientations=9,
                         pixels_per_cell=(cfg.hog_pixels_per_cell,) * 2,
                         cells_per_block=(cfg.hog_cells_per_block,) * 2) for im in images]).astype(np.float32)


def train_classical(cfg, data, splits, seed):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    cache = cfg.sub("cache")
    if not (cache / "hog_hk.npy").exists():
        np.save(cache / "hog_hk.npy", hog_features(cfg, data["hk_images"]))
        np.save(cache / "hog_cvc.npy", hog_features(cfg, data["cvc_images"]))
    X, Xc = np.load(cache / "hog_hk.npy"), np.load(cache / "hog_cvc.npy")
    y = data["hk_labels"]
    sp = splits["grouped"]
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    n_comp = min(256, len(tr) - 1, X.shape[1])
    models = {
        "svm_hog": make_pipeline(StandardScaler(), PCA(n_comp, random_state=seed),
                                 SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=seed)),
        "rf_hog": RandomForestClassifier(n_estimators=cfg.rf_trees, random_state=seed, n_jobs=-1),
    }
    for name, clf in models.items():
        rf = result_path(cfg, name, seed, "grouped")
        if rf.exists():
            continue
        clf.fit(X[tr], y[tr])
        p_full = np.zeros((len(te), len(CLASSES)))
        p_full[:, clf.classes_] = clf.predict_proba(X[te])
        pc = np.zeros((len(Xc), len(CLASSES)))
        pc[:, clf.classes_] = clf.predict_proba(Xc)
        np.savez_compressed(cfg.sub("results", "classification", "grouped") / f"{name}_seed{seed}_preds.npz",
                            test_idx=te, test_probs=p_full, cvc_probs=pc)
        save_json(rf, {"model": name, "seed": seed, "split": "grouped", "test": classification_metrics(p_full, y[te]),
                       "n_test": int(len(te)), "cvc_polyp_recall": float((pc.argmax(1) == POLYP_IDX).mean()),
                       "n_cvc": int(len(pc))})
        print(f"{name} seed {seed} done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="subset of backbones, svm_hog, rf_hog")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    cfg = get_config(args.smoke)
    data, splits = prepare(cfg, verbose=False)
    seeds = args.seeds if args.seeds is not None else list(cfg.seeds_classifier)
    names = args.only or list(cfg.backbones) + ["classical"]
    for seed in seeds:
        for name in names:
            if name in ("classical", "svm_hog", "rf_hog"):
                train_classical(cfg, data, splits, seed)
            else:
                train_deep(cfg, data, splits, name, seed, "grouped")
        if cfg.also_image_level_split and "proposed" in names:
            train_deep(cfg, data, splits, "proposed", seed, "image_level")


if __name__ == "__main__":
    main()
