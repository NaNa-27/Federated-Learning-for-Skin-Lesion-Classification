#!/usr/bin/env python
"""Train one arm of the HAM10000 federated-learning study.

Arms (all share backbone, 224x224 input, augmentation, loss, optimizer and budget):
  centralized  pooled training split of all 4 clients
  local        one client only (--client)
  fedavg       FedAvg, sample-weighted averaging, all clients every round
  fedprox      FedAvg + proximal term (--mu)

Budget rule (proposal, Sec. V-C): R rounds x E local epochs = centralized epochs,
so every arm makes the same number of passes over each training image.

Model selection uses the pooled validation set only. The test set is predicted
once, with the best checkpoint, at the end of the run.

Usage:
  python fl_train.py --build-cache --partition P --image-root R --cache C
  python fl_train.py --arm fedavg --partition P --cache C --out-dir runs
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
CLIENTS = ["vidir_molemax", "vidir_modern", "rosendahl", "vienna_dias"]
FL_COLUMNS = ["image_id", "lesion_id", "dx", "split", "client"]
FL_MANIFEST_SHA256 = "0ab49e23b128c7d9f2d309bee762ade1d6ab69f04a31ca94096f29484c9a033d"
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- data
def fingerprint(df, cols):
    """Same definition as 02_federated_client_partition.ipynb."""
    csv = df[cols].sort_values("image_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()


def load_partition(path, check=True):
    part = pd.read_csv(path)
    assert list(part.columns) == FL_COLUMNS, part.columns.tolist()
    if check:
        sha = fingerprint(part, FL_COLUMNS)
        if sha != FL_MANIFEST_SHA256:
            sys.exit(f"FL partition fingerprint mismatch: {sha}. Do not train on this file.")
    return part.sort_values("image_id").reset_index(drop=True)


def _load_one(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1).copy()


def build_cache(partition_path, image_root, cache_path):
    """Resize every image once to 224x224 (proposal: 'resized to 224 x 224')."""
    from multiprocessing import Pool

    part = load_partition(partition_path)
    paths = {}
    for p in Path(image_root).rglob("ISIC_*.jpg"):
        paths.setdefault(p.stem, p)  # the Kaggle copy has duplicate folders; any copy is fine
    missing = [i for i in part["image_id"] if i not in paths]
    assert not missing, f"{len(missing)} images not found under {image_root}, e.g. {missing[:3]}"

    files = [str(paths[i]) for i in part["image_id"]]
    X = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8,
                                  shape=(len(files), 3, IMG_SIZE, IMG_SIZE))
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        for i, arr in enumerate(pool.imap(_load_one, files, chunksize=32)):
            X[i] = arr
            if (i + 1) % 2000 == 0:
                print(f"  cached {i + 1:,}/{len(files):,} ({time.time() - t0:.0f}s)", flush=True)
    X.flush()
    Path(str(cache_path) + ".ids.txt").write_text("\n".join(part["image_id"]) + "\n")
    print(f"Cache written: {cache_path} {X.shape} in {time.time() - t0:.0f}s")


def load_cache(cache_path, part):
    ids = Path(str(cache_path) + ".ids.txt").read_text().split()
    assert ids == part["image_id"].tolist(), "Cache order does not match the partition; rebuild it."
    return np.load(cache_path, mmap_mode="r")


# --------------------------------------------------------------------------- model
def build_model(name, pretrained):
    import torchvision
    w = "IMAGENET1K_V1" if pretrained else None
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    elif name == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    else:
        raise ValueError(name)
    return m


class Normalizer:
    def __init__(self, device):
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, x_uint8):
        x = x_uint8.float().div_(255)
        return ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)


def gpu_augment(x_uint8, gen):
    """Dihedral group (flip + 90-degree rotations) and mild brightness/contrast.

    Dermoscopic lesions have no canonical orientation, so all 8 orientations are valid.
    Runs on the GPU so the data pipeline never waits on the 4 Kaggle CPUs.
    """
    x = x_uint8.float().div_(255)
    b = x.shape[0]
    dev = x.device
    flip = torch.rand(b, device=dev, generator=gen) < 0.5
    x = torch.where(flip.view(b, 1, 1, 1), x.flip(3), x)
    k = torch.randint(0, 4, (b,), device=dev, generator=gen)
    out = x.clone()
    for r in (1, 2, 3):
        m = k == r
        if m.any():
            out[m] = torch.rot90(x[m], r, dims=(2, 3))
    bright = (torch.rand(b, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.2
    contrast = 1 + (torch.rand(b, 1, 1, 1, device=dev, generator=gen) - 0.5) * 0.2
    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    out = ((out - mean) * contrast + mean + bright).clamp_(0, 1)
    return out.mul_(255)  # back to [0, 255] float; Normalizer divides again


# --------------------------------------------------------------------------- metrics
def compute_metrics(y, prob):
    pred = prob.argmax(1)
    k = len(CLASSES)
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    predicted = cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(k), where=(precision + recall) > 0)
    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "accuracy": float(tp.sum() / cm.sum()),
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm.tolist(),
    }


# --------------------------------------------------------------------------- training
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(base, step, total):
    return base * 0.5 * (1 + math.cos(math.pi * step / total))


def make_optimizer(model, lr, args):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                           weight_decay=args.weight_decay, nesterov=True)


def train_one_epoch(model, opt, scaler, X, y, idx, ctx, global_params=None, mu=0.0):
    """One pass over `idx` (a 1-D LongTensor of rows on the device)."""
    model.train()
    perm = idx[torch.randperm(len(idx), device=idx.device, generator=ctx["gen"])]
    bs = ctx["batch_size"]
    total, n = 0.0, 0
    for s in range(0, len(perm), bs):
        bidx = perm[s:s + bs]
        if len(bidx) < 2:  # BatchNorm needs >1 sample
            continue
        xb = ctx["norm"](gpu_augment(X[bidx], ctx["gen"]))
        yb = y[bidx]
        with torch.autocast(device_type=ctx["device_type"], dtype=ctx["amp_dtype"],
                            enabled=ctx["amp"]):
            logits = model(xb)
        loss = F.cross_entropy(logits.float(), yb, weight=ctx["class_weights"])
        if mu > 0:
            prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
            loss = loss + 0.5 * mu * prox
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * len(bidx)
        n += len(bidx)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, X, idx, ctx, bs=256):
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        xb = ctx["norm"](X[idx[s:s + bs]])
        with torch.autocast(device_type=ctx["device_type"], dtype=ctx["amp_dtype"],
                            enabled=ctx["amp"]):
            out.append(model(xb).float().softmax(1))
    return torch.cat(out).cpu().numpy()


def fedavg_aggregate(states, weights):
    agg = {}
    for key, ref in states[0].items():
        if ref.is_floating_point():
            agg[key] = sum(w * s[key].float() for s, w in zip(states, weights)).to(ref.dtype)
        else:  # e.g. BatchNorm num_batches_tracked
            agg[key] = ref.clone()
    return agg


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def run_name_for(args):
    base = f"{args.arm}_{args.backbone}_s{args.seed}"
    if args.arm == "local":
        base = f"local-{args.client}_{args.backbone}_s{args.seed}"
    if args.arm == "fedprox":
        base = f"fedprox-mu{args.mu:g}_{args.backbone}_s{args.seed}"
    return args.run_name or base


def env_info(device):
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
    return info


def main(args):
    if args.build_cache:
        build_cache(args.partition, args.image_root, args.cache)
        return

    run_dir = Path(args.out_dir) / run_name_for(args)
    if (run_dir / "metrics.json").exists():
        print(f"[skip] {run_dir.name} already finished")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    part = load_partition(args.partition)
    X_np = load_cache(args.cache, part)
    if args.debug_n:  # smoke test: a few images per client x split
        part = part.groupby(["client", "split"], group_keys=False).head(args.debug_n)
    rows = part.index.to_numpy()
    X = torch.from_numpy(np.ascontiguousarray(X_np[rows])).to(device)  # uint8 on the GPU (~1.5 GB)
    part = part.reset_index(drop=True)
    y = torch.tensor(part["dx"].map(CLASSES.index).to_numpy(), device=device)

    def rows_of(mask):
        return torch.tensor(np.flatnonzero(mask.to_numpy()), device=device)

    train_mask = part["split"] == "train"
    val_idx, test_idx = rows_of(part["split"] == "val"), rows_of(part["split"] == "test")
    client_idx = {c: rows_of(train_mask & (part["client"] == c)) for c in CLIENTS}
    pooled_idx = rows_of(train_mask)

    # Class weights: inverse square-root frequency of the pooled training split,
    # normalized to mean 1, identical for every arm.
    counts = np.bincount(part.loc[train_mask, "dx"].map(CLASSES.index), minlength=len(CLASSES))
    cw = 1.0 / np.sqrt(np.maximum(counts, 1))
    cw = cw / cw.mean()

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    amp = device.type == "cuda"
    ctx = {"gen": gen, "batch_size": args.batch_size, "norm": Normalizer(device),
           "device_type": device.type, "amp": amp, "amp_dtype": torch.float16,
           "class_weights": torch.tensor(cw, dtype=torch.float32, device=device)}
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    model = build_model(args.backbone, pretrained=not args.no_pretrained).to(device)
    model = model.to(memory_format=torch.channels_last)

    federated = args.arm in ("fedavg", "fedprox")
    if federated:
        assert args.epochs % args.local_epochs == 0, "epochs must be a multiple of local_epochs"
        n_steps = args.epochs // args.local_epochs  # rounds R
        mu = args.mu if args.arm == "fedprox" else 0.0
        sizes = np.array([len(client_idx[c]) for c in CLIENTS], dtype=float)
        agg_w = (sizes / sizes.sum()).tolist()
    else:
        n_steps = args.epochs
        train_idx = client_idx[args.client] if args.arm == "local" else pooled_idx
        opt = make_optimizer(model, args.lr, args)

    # ---- resume -------------------------------------------------------------
    start, best_f1, best_step, history = 0, -1.0, -1, []
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if not federated:
            opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        gen.set_state(ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        start, best_f1, best_step, history = ck["step"], ck["best_f1"], ck["best_step"], ck["history"]
        print(f"[resume] {run_dir.name} from step {start}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    y_val = y[val_idx].cpu().numpy()

    for step in range(start, n_steps):
        lr = cosine_lr(args.lr, step, n_steps)
        ts = time.time()
        if federated:
            global_state = clone_state(model)
            global_params = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
            states, losses = [], []
            for c in CLIENTS:
                model.load_state_dict(global_state)
                copt = make_optimizer(model, lr, args)  # fresh client optimizer each round
                for _ in range(args.local_epochs):
                    losses.append(train_one_epoch(model, copt, scaler, X, y, client_idx[c], ctx,
                                                  global_params, mu))
                states.append(clone_state(model))
            model.load_state_dict(fedavg_aggregate(states, agg_w))
            del states
            train_loss = float(np.mean(losses))
        else:
            for g in opt.param_groups:
                g["lr"] = lr
            train_loss = train_one_epoch(model, opt, scaler, X, y, train_idx, ctx)

        val_prob = predict(model, X, val_idx, ctx)
        vm = compute_metrics(y_val, val_prob)
        improved = vm["macro_f1"] > best_f1
        if improved:
            best_f1, best_step = vm["macro_f1"], step + 1
            torch.save(model.state_dict(), best_path)
        history.append({"step": step + 1, "lr": lr, "train_loss": train_loss,
                        "val_macro_f1": vm["macro_f1"], "val_bal_acc": vm["balanced_accuracy"],
                        "val_acc": vm["accuracy"], "seconds": time.time() - ts})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"{run_dir.name} {'round' if federated else 'epoch'} {step + 1}/{n_steps} "
              f"loss {train_loss:.4f} val-MF1 {vm['macro_f1']:.4f}{' *' if improved else ''} "
              f"({time.time() - ts:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "opt": None if federated else opt.state_dict(),
                    "scaler": scaler.state_dict(), "gen": gen.get_state(),
                    "torch_rng": torch.get_rng_state(), "step": step + 1, "best_f1": best_f1,
                    "best_step": best_step, "history": history}, last_path)

    # ---- final evaluation with the best checkpoint (test used once) ----------
    model.load_state_dict(torch.load(best_path, map_location=device))
    val_prob = predict(model, X, val_idx, ctx)
    test_prob = predict(model, X, test_idx, ctx)
    y_test = y[test_idx].cpu().numpy()
    np.savez_compressed(
        run_dir / "preds.npz",
        val_ids=part.loc[val_idx.cpu().numpy(), "image_id"].to_numpy(), val_y=y_val, val_prob=val_prob,
        test_ids=part.loc[test_idx.cpu().numpy(), "image_id"].to_numpy(), test_y=y_test,
        test_prob=test_prob)

    metrics = {
        "run": run_dir.name, "arm": args.arm, "client": args.client if args.arm == "local" else None,
        "backbone": args.backbone, "seed": args.seed,
        "mu": args.mu if args.arm == "fedprox" else (0.0 if federated else None),
        "epochs_budget": args.epochs,
        "rounds": n_steps if federated else None,
        "local_epochs": args.local_epochs if federated else None,
        "lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "optimizer": "SGD nesterov, cosine schedule",
        "loss": "cross-entropy, inverse-sqrt class weights (pooled train)",
        "class_weights": dict(zip(CLASSES, map(float, cw))),
        "n_train_images": int(sum(len(v) for v in client_idx.values())) if federated or args.arm == "centralized"
        else int(len(client_idx[args.client])),
        "best_step": best_step, "selection": "best pooled-val macro-F1",
        "val": compute_metrics(y_val, val_prob), "test": compute_metrics(y_test, test_prob),
        "train_seconds": time.time() - t0,
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None,
        "debug_n": args.debug_n, "env": env_info(device),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    last_path.unlink(missing_ok=True)
    print(f"[done] {run_dir.name}: best {('round' if federated else 'epoch')} {best_step}, "
          f"val MF1 {metrics['val']['macro_f1']:.4f}, test MF1 {metrics['test']['macro_f1']:.4f}, "
          f"{metrics['train_seconds'] / 60:.1f} min")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build-cache", action="store_true")
    p.add_argument("--image-root", default="/kaggle/input")
    p.add_argument("--partition", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--arm", choices=["centralized", "local", "fedavg", "fedprox"], default="centralized")
    p.add_argument("--client", choices=CLIENTS, default=None)
    p.add_argument("--backbone", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20, help="budget: centralized epochs = R x E")
    p.add_argument("--local-epochs", type=int, default=1, help="E (federated arms)")
    p.add_argument("--mu", type=float, default=0.0, help="FedProx proximal weight")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--debug-n", type=int, default=0, help="smoke test: images per client x split")
    a = p.parse_args(argv)
    if a.arm == "local" and a.client is None:
        p.error("--arm local needs --client")
    if a.arm == "fedprox" and a.mu <= 0:
        p.error("--arm fedprox needs --mu > 0")
    return a


if __name__ == "__main__":
    main(parse_args())
