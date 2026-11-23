"""Train one model and save best/last checkpoints + per-epoch history.

Example:
    python -m src.train --model resnet18 --epochs 6 --samples-per-epoch 16000

Outputs (in ``models/``):
    {run}_best.pt       best epoch by validation macro-F1 (used for evaluation/app)
    {run}_last.pt       latest epoch incl. optimiser state (used to --resume)
    {run}_history.json  per-epoch train/val loss + accuracy, timings, status
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from src import config
from src.dataset import get_device, make_loaders, prepare_batch
from src.models import build_model, count_parameters, save_checkpoint


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    """One pass over ``loader``. Trains if an optimiser is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            inputs = prepare_batch(images, device, augment=training)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            preds.append(logits.argmax(1).cpu())
            targets.append(labels.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {"loss": total_loss / len(targets), "acc": float((preds == targets).mean()),
            "macro_f1": float(f1_score(targets, preds, average="macro")),
            "n": int(len(targets))}


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = get_device()
    run = args.run_name or f"{args.model}_{args.mode}"
    config.MODELS_DIR.mkdir(exist_ok=True)
    best_path = config.MODELS_DIR / f"{run}_best.pt"
    last_path = config.MODELS_DIR / f"{run}_last.pt"
    hist_path = config.MODELS_DIR / f"{run}_history.json"

    train_loader, val_loader, _ = make_loaders(
        batch_size=args.batch_size, samples_per_epoch=args.samples_per_epoch,
        val_limit=args.val_limit, num_workers=args.workers)

    model_kwargs = {}
    if args.model == "custom_cnn":
        model_kwargs = {"dropout": args.dropout, "batch_norm": not args.no_bn,
                        "activation": args.activation}
    model = build_model(args.model, mode=args.mode,
                        **({"dropout": args.dropout} | model_kwargs)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # Adam's weight_decay adds an L2 penalty (lambda * w) to every gradient.
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.3, patience=1)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    total_params, trainable_params = count_parameters(model)

    history = {"run": run, "model": args.model, "mode": args.mode, "args": vars(args),
               "device": str(device), "total_params": total_params,
               "trainable_params": trainable_params, "epochs": [],
               "best_epoch": None, "best_val_macro_f1": -1.0, "status": "running"}
    start_epoch, bad_epochs = 1, 0
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = json.loads(hist_path.read_text())
        history["status"] = "running"
        start_epoch, bad_epochs = state["epoch"] + 1, state["bad_epochs"]
        print(f"[resume] {run} from epoch {start_epoch}")

    print(f"[train] {run} on {device} | params {total_params:,} "
          f"(trainable {trainable_params:,}) | "
          f"train/epoch {args.samples_per_epoch or len(train_loader.dataset):,} "
          f"| val {len(val_loader.dataset):,}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, args.max_batches)
        t1 = time.time()
        va = run_epoch(model, val_loader, criterion, device, None, args.max_batches)
        t2 = time.time()
        scheduler.step(va["macro_f1"])
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"],
               "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"],
               "lr": optimizer.param_groups[0]["lr"], "train_seconds": t1 - t0,
               "val_seconds": t2 - t1, "train_images": tr["n"],
               "train_images_per_sec": tr["n"] / (t1 - t0)}
        history["epochs"].append(row)
        improved = va["macro_f1"] > history["best_val_macro_f1"]
        if improved:
            history.update(best_epoch=epoch, best_val_macro_f1=va["macro_f1"],
                           best_val_acc=va["acc"])
            save_checkpoint(best_path, model, args.model, args.mode,
                            {"model_kwargs": model_kwargs, "epoch": epoch,
                             "val_acc": va["acc"], "val_macro_f1": va["macro_f1"],
                             "run": run})
            bad_epochs = 0
        else:
            bad_epochs += 1
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "bad_epochs": bad_epochs}, last_path)
        print(f"[{run}] ep {epoch}/{args.epochs} train loss {tr['loss']:.4f} "
              f"acc {tr['acc']:.4f} | val loss {va['loss']:.4f} acc {va['acc']:.4f} "
              f"F1 {va['macro_f1']:.4f} | {t1 - t0:.0f}s+{t2 - t1:.0f}s "
              f"({row['train_images_per_sec']:.0f} img/s){' *best*' if improved else ''}",
              flush=True)
        if args.max_batches:
            history["status"] = "smoke_test"
        hist_path.write_text(json.dumps(history, indent=2))
        if bad_epochs >= args.patience:
            history["status"] = "early_stopped"
            print(f"[{run}] early stopping after {args.patience} epochs without improvement")
            break
    if history["status"] == "running":
        history["status"] = "complete"
    history["total_train_seconds"] = sum(e["train_seconds"] + e["val_seconds"]
                                         for e in history["epochs"])
    hist_path.write_text(json.dumps(history, indent=2))
    return history


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=config.MODEL_NAMES, required=True)
    p.add_argument("--mode", choices=["finetune", "feature_extract"], default="finetune")
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--samples-per-epoch", type=int, default=None,
                   help="class-balanced scans drawn per epoch (default: full train split)")
    p.add_argument("--val-limit", type=int, default=None,
                   help="fixed stratified validation subset size (default: all)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 strength")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-bn", action="store_true", help="custom CNN without BatchNorm")
    p.add_argument("--activation", default="relu", choices=["relu", "leaky_relu", "elu"])
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=3, help="early-stopping patience")
    p.add_argument("--workers", type=int, default=0,
                   help="0 is fastest here: the memmap cache is already in RAM-speed")
    p.add_argument("--max-batches", type=int, default=None, help="smoke test only")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
