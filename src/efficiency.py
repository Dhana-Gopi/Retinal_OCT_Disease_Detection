"""Computational efficiency of every trained checkpoint.

Per run: parameter count, checkpoint size, FLOPs, single-image latency on CPU
and on the Apple GPU (MPS), batch throughput, memory, and training time.

    python -m src.efficiency                 # measure (cached runs are reused)
    python -m src.efficiency --only-if-idle  # do nothing while a training job runs
    python -m src.efficiency --force         # re-measure everything

Timing is only fair when no training job shares the machine.  If
``src.train`` is running, results are still written but flagged with
``"measured_while_training": true``; such entries are re-measured
automatically on the next run when the machine is idle.

Outputs: outputs/metrics/eval_efficiency.{json,csv,md}
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config

FINAL_STATUSES = {"complete", "early_stopped"}


# --------------------------------------------------------------------------
# Run discovery (shared with src.evaluate)
# --------------------------------------------------------------------------
def run_group(run: str, status: str | None = None) -> str:
    """final | ablation | sweep | smoke."""
    if run.startswith("smoke") or status == "smoke_test":
        return "smoke"
    if run.startswith("cnn_bs"):
        return "sweep"
    if run.startswith("cnn_no_"):
        return "ablation"
    return "final"


def discover_runs(include_smoke: bool = False) -> list[dict]:
    """Every ``models/*_best.pt`` whose history says the run has finished.

    Returns dicts with run, checkpoint, history (parsed), status, group.
    Runs still ``running`` (or without a history) are skipped.
    """
    runs = []
    for ckpt in sorted(config.MODELS_DIR.glob("*_best.pt")):
        run = ckpt.name[: -len("_best.pt")]
        hist_path = config.MODELS_DIR / f"{run}_history.json"
        if not hist_path.exists():
            print(f"[skip] {run}: no history file")
            continue
        try:
            history = json.loads(hist_path.read_text())
        except json.JSONDecodeError:
            print(f"[skip] {run}: history unreadable (being written?)")
            continue
        status = history.get("status")
        group = run_group(run, status)
        if group == "smoke":
            if not include_smoke:
                continue
        elif status not in FINAL_STATUSES:
            print(f"[skip] {run}: status '{status}'")
            continue
        runs.append({"run": run, "checkpoint": ckpt, "history": history,
                     "status": status, "group": group})
    return runs


def training_running() -> bool:
    """True if a ``src.train`` process (other than this one) is alive."""
    try:
        out = subprocess.run(["pgrep", "-f", "src.train"], capture_output=True,
                             text=True, check=False).stdout
    except FileNotFoundError:
        return False
    pids = {int(p) for p in out.split() if p.strip().isdigit()}
    pids.discard(os.getpid())
    return bool(pids)


def training_summary(history: dict) -> dict:
    epochs = history.get("epochs", [])
    train_s = sum(e.get("train_seconds", 0.0) for e in epochs)
    total_s = history.get("total_train_seconds",
                          sum(e.get("train_seconds", 0) + e.get("val_seconds", 0)
                              for e in epochs))
    imgs = sum(e.get("train_images", 0) for e in epochs)
    return {"epochs_run": len(epochs), "best_epoch": history.get("best_epoch"),
            "train_time_min": round(total_s / 60, 2),
            "train_only_time_min": round(train_s / 60, 2),
            "train_images_seen": imgs,
            "train_images_per_sec": round(imgs / train_s, 1) if train_s else None,
            "training_device": history.get("device")}


def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    """Minimal Markdown table (avoids the optional ``tabulate`` dependency)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return "" if v is None else str(v).replace("|", "/")
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**20


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def count_flops(model: torch.nn.Module) -> float | None:
    """GFLOPs for one 224x224 image (1 multiply-add = 2 FLOPs), CPU."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        x = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(x)
        return counter.get_total_flops() / 1e9
    except Exception as err:  # noqa: BLE001 - FLOPs are optional
        print(f"[efficiency] FLOP count failed: {err}")
        return None


def time_inference(model: torch.nn.Module, device: str, batch: int, warmup: int,
                   repeats: int) -> list[float]:
    """Seconds per forward pass (synchronised) for a ``batch`` x 3 x 224 x 224 input."""
    x = torch.randn(batch, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    times = []
    with torch.inference_mode():
        for i in range(warmup + repeats):
            _sync(device)
            t0 = time.perf_counter()
            model(x)
            _sync(device)
            if i >= warmup:
                times.append(time.perf_counter() - t0)
    return times


def measure_run(entry: dict, devices: list[str], repeats: int = 30,
                batch: int = 32) -> dict:
    from src.models import count_parameters, load_checkpoint

    ckpt_path = Path(entry["checkpoint"])
    busy_before = training_running()
    gc.collect()
    rss0 = _rss_mb()
    model, ckpt = load_checkpoint(ckpt_path, "cpu")
    total, trainable = count_parameters(model)
    trainable_hist = entry["history"].get("trainable_params", trainable)
    result = {
        "run": entry["run"], "model": ckpt["model_name"], "mode": ckpt.get("mode"),
        "group": entry["group"], "checkpoint": str(ckpt_path.relative_to(config.PROJECT_ROOT)),
        "checkpoint_mtime": ckpt_path.stat().st_mtime,
        "params": total, "trainable_params_during_training": trainable_hist,
        "params_millions": round(total / 1e6, 3),
        "checkpoint_mb": round(ckpt_path.stat().st_size / 2**20, 2),
        "weights_fp32_mb": round(total * 4 / 2**20, 2),
        "gflops_per_image": None,
        **training_summary(entry["history"]),
    }
    flops = count_flops(model)
    result["gflops_per_image"] = round(flops, 3) if flops is not None else None

    for device in devices:
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        if device == "cuda" and not torch.cuda.is_available():
            continue
        if device == "mps":
            torch.mps.empty_cache()
            mps0 = torch.mps.current_allocated_memory()
        model = model.to(device)
        rss_before = _rss_mb()
        single = time_inference(model, device, 1, warmup=5, repeats=repeats)
        batched = time_inference(model, device, batch, warmup=2,
                                 repeats=max(3, repeats // 6))
        rss_after = _rss_mb()
        result[f"{device}_latency_ms"] = round(statistics.median(single) * 1e3, 2)
        result[f"{device}_latency_p90_ms"] = round(float(np.percentile(single, 90)) * 1e3, 2)
        result[f"{device}_batch{batch}_ms_per_image"] = round(
            statistics.median(batched) * 1e3 / batch, 2)
        result[f"{device}_throughput_img_s"] = round(batch / statistics.median(batched), 1)
        result[f"{device}_rss_increase_mb"] = round(rss_after - rss_before, 1)
        if device == "mps":
            # weights + cached activations still held by the MPS allocator
            result["mps_allocated_mb"] = round(
                (torch.mps.current_allocated_memory() - mps0) / 2**20, 1)
            result["mps_driver_allocated_mb"] = round(
                torch.mps.driver_allocated_memory() / 2**20, 1)
        model = model.to("cpu")
    result["process_rss_increase_mb"] = round(_rss_mb() - rss0, 1)
    result["cpu_threads"] = torch.get_num_threads()
    result["measured_while_training"] = busy_before or training_running()
    result["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    lat = result.get("cpu_latency_ms")
    if lat is not None:
        result["cpu_feasibility"] = (
            "real-time on a laptop CPU (<50 ms/scan)" if lat < 50 else
            "interactive on CPU (<250 ms/scan)" if lat < 250 else
            "slow on CPU (>=250 ms/scan); GPU recommended")
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


def machine_info() -> dict:
    import psutil

    info = {"platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "torch": torch.__version__, "mps_available": torch.backends.mps.is_available()}
    try:
        info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                      capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        pass
    return info


COLUMNS = ["run", "model", "mode", "group", "params_millions", "checkpoint_mb", "gflops_per_image",
           "cpu_latency_ms", "cpu_throughput_img_s", "mps_latency_ms", "mps_throughput_img_s",
           "process_rss_increase_mb", "mps_allocated_mb", "train_time_min", "epochs_run",
           "train_images_per_sec", "cpu_feasibility", "measured_while_training"]


def run_efficiency(runs: list[dict], out_dir: Path = config.METRICS_DIR, force: bool = False,
                   only_if_idle: bool = False, devices: list[str] | None = None,
                   repeats: int = 30) -> dict:
    """Measure (or reuse cached) efficiency for ``runs``; returns {run: result}."""
    devices = devices or ["cpu", "mps"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_efficiency.json"
    cache = {}
    if json_path.exists():
        try:
            cache = json.loads(json_path.read_text()).get("runs", {})
        except json.JSONDecodeError:
            cache = {}
    busy = training_running()
    if only_if_idle and busy:
        print("[efficiency] training is running -> --only-if-idle: nothing measured")
        return {r["run"]: cache[r["run"]] for r in runs if r["run"] in cache}

    results = {}
    for entry in runs:
        old = cache.get(entry["run"])
        fresh = (old is not None and not force
                 and abs(old.get("checkpoint_mtime", 0) - entry["checkpoint"].stat().st_mtime) < 1
                 and not (old.get("measured_while_training") and not busy)
                 and all(f"{d}_latency_ms" in old for d in devices
                         if d != "mps" or torch.backends.mps.is_available()))
        if fresh:
            results[entry["run"]] = old
            continue
        print(f"[efficiency] measuring {entry['run']} on {devices}"
              f"{' (training running: flagged)' if busy else ''}", flush=True)
        results[entry["run"]] = measure_run(entry, devices, repeats=repeats)
    # keep cached entries of runs not requested this time (e.g. partial reruns)
    merged = {**{k: v for k, v in cache.items() if k not in results}, **results}
    json_path.write_text(json.dumps({"machine": machine_info(), "runs": merged}, indent=2))
    df = pd.DataFrame(list(merged.values()))
    if not df.empty:
        df = df[[c for c in COLUMNS if c in df.columns]].sort_values(["group", "run"])
        df.to_csv(out_dir / "eval_efficiency.csv", index=False)
        note = ("\n\n> **Warning:** rows with `measured_while_training = True` were timed while a "
                "training job shared the machine; rerun `python -m src.efficiency --only-if-idle` "
                "when idle.\n" if df["measured_while_training"].any() else "\n")
        (out_dir / "eval_efficiency.md").write_text(
            "# Computational efficiency\n\nLatency = median of single-image forward passes "
            "(after warm-up); throughput = batch of 32. FLOPs: 1 multiply-add = 2 FLOPs.\n\n"
            + df_to_markdown(df) + note)
    print(f"[efficiency] wrote {json_path}")
    return results


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", help="only these run names")
    p.add_argument("--only-if-idle", action="store_true",
                   help="skip measuring when a src.train process is running")
    p.add_argument("--force", action="store_true", help="ignore cached measurements")
    p.add_argument("--include-smoke", action="store_true",
                   help="testing only: include smoke_* runs (written to a separate folder)")
    p.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    out_dir = args.out_dir or (config.METRICS_DIR / "eval_smoke_test" if args.include_smoke
                               else config.METRICS_DIR)
    runs = discover_runs(include_smoke=args.include_smoke)
    if args.runs:
        runs = [r for r in runs if r["run"] in set(args.runs)]
    if not runs:
        print("[efficiency] no finished runs found")
        return
    run_efficiency(runs, out_dir, force=args.force, only_if_idle=args.only_if_idle,
                   devices=args.devices, repeats=args.repeats)


if __name__ == "__main__":
    main()
