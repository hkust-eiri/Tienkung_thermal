#!/usr/bin/env python3
"""在 test（或 val）集上对已训练的 UltraThermalLSTM 做完整评估。

输出：
  - 等权 MAE@15s（Gate 指标）
  - 每关节 MAE@15s
  - 每 horizon 全局 MAE
  - 最大绝对误差
  - 可选：将结果写入 JSON

用法::

    python scripts/evaluate.py --checkpoint checkpoints/best_ultra_thermal.pt

    # 在 val 集上评估
    python scripts/evaluate.py --checkpoint checkpoints/best_ultra_thermal.pt --split val

    # 保存结果到 JSON
    python scripts/evaluate.py --checkpoint checkpoints/best_ultra_thermal.pt --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _collect_h5(h5_dir: str, manifest_path: str | None) -> tuple[list[str], list[str], list[str]]:
    """与 train.py 相同的划分逻辑，保证 test 集一致。"""
    import csv
    import random

    h5_dir = Path(h5_dir)
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path) as f:
            rows = list(csv.DictReader(f))
        train, val, test = [], [], []
        unassigned = []
        for r in rows:
            p = r.get("hdf5_path", "")
            if not p or not Path(p).exists():
                continue
            split = (r.get("split") or "").strip().lower()
            if split == "train":
                train.append(p)
            elif split == "val":
                val.append(p)
            elif split == "test":
                test.append(p)
            else:
                unassigned.append(p)
        if unassigned:
            random.seed(42)
            random.shuffle(unassigned)
            n = len(unassigned)
            n_val = max(1, n // 10)
            n_test = max(1, n // 10)
            val.extend(unassigned[:n_val])
            test.extend(unassigned[n_val : n_val + n_test])
            train.extend(unassigned[n_val + n_test :])
        return train, val, test

    all_h5 = sorted(h5_dir.glob("*.h5"))
    all_paths = [str(p) for p in all_h5]
    if not all_paths:
        raise FileNotFoundError(f"no .h5 files in {h5_dir}")
    random.seed(42)
    random.shuffle(all_paths)
    n = len(all_paths)
    n_val = max(1, n // 10)
    n_test = max(1, n // 10)
    return (
        all_paths[n_val + n_test :],
        all_paths[:n_val],
        all_paths[n_val : n_val + n_test],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UltraThermalLSTM on test/val set")
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pt 路径")
    parser.add_argument("--config", default="configs/ultra_thermal_lstm.yaml")
    parser.add_argument("--split", default="test", choices=["test", "val"], help="评估哪个划分")
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=None, help="结果 JSON 输出路径")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("evaluate")

    cfg = _load_config(args.config)
    model_cfg = cfg["model"]
    seq_cfg = cfg["sequence"]
    feat_cfg = cfg["features"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    data_cfg = cfg["data"]

    use_derived = not args.raw_only and feat_cfg.get("use_derived", True)
    use_adj = feat_cfg.get("optional_adjacent_temp", False)
    use_imu = feat_cfg.get("optional_imu", False)

    d = 5
    if use_derived:
        d += 4
    if use_adj:
        d += 2
    if use_imu:
        d += 9

    import torch
    from torch.utils.data import DataLoader

    from tienkung_thermal.data.dataset import UltraThermalDataset
    from tienkung_thermal.data.norm import load_norm_stats, stats_to_tensors
    from tienkung_thermal.models.thermal_lstm import UltraThermalLSTM
    from tienkung_thermal.training.trainer import ThermalLoss

    # ── 归一化 ────────────────────────────────────────────
    norm_cfg = cfg.get("normalization", {})
    stats_path = norm_cfg.get("stats_path")
    log1p_fields = tuple(norm_cfg.get("log1p_fields", ["ddq_abs", "tau_sq"]))

    norm_stats_tensors = None
    if norm_cfg.get("method") == "z_score":
        if not stats_path:
            stats_path = str(Path(data_cfg.get("h5_dir", "data/processed/leg_status_500hz")) / "norm_stats.json")
        if Path(stats_path).exists():
            norm_stats_tensors = stats_to_tensors(load_norm_stats(stats_path))
        else:
            log.warning("norm_stats not found at %s — running without normalization", stats_path)

    # ── 数据集 ────────────────────────────────────────────
    train_h5, val_h5, test_h5 = _collect_h5(
        data_cfg.get("h5_dir", "data/processed/leg_status_500hz"),
        data_cfg.get("manifest_path"),
    )
    split_h5 = test_h5 if args.split == "test" else val_h5
    log.info("split=%s  sessions=%d", args.split, len(split_h5))

    horizon_steps = cfg.get("horizon_steps", [250, 500, 1000, 1500, 2500, 3500, 5000, 6000, 7500])
    seq_len = seq_cfg.get("seq_len", 2500)
    stride = seq_cfg.get("stride", 50)

    ds = UltraThermalDataset(
        split_h5,
        seq_len=seq_len,
        horizon_steps=horizon_steps,
        use_derived=use_derived,
        use_adjacent_temp=use_adj,
        use_imu=use_imu,
        stride=stride,
        norm_stats=norm_stats_tensors,
        log1p_fields=log1p_fields,
    )
    log.info("samples=%d", len(ds))
    if len(ds) == 0:
        log.error("dataset is empty")
        sys.exit(1)

    batch_size = args.batch_size or train_cfg.get("batch_size", 128)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ── 模型 ──────────────────────────────────────────────
    device = torch.device(args.device or train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = UltraThermalLSTM(
        input_dim=d,
        proj_dim=model_cfg.get("proj_dim", 32),
        hidden_dim=model_cfg.get("hidden_dim", 96),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.10),
        mid_dim=model_cfg.get("mid_dim", 64),
        horizon=model_cfg.get("horizon", 9),
        n_joints=model_cfg.get("n_joints", 12),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("loaded checkpoint epoch=%s  val_mae_15s=%.4f°C", ckpt.get("epoch"), ckpt.get("val_mae_15s", float("nan")))

    criterion = ThermalLoss(
        huber_weight=loss_cfg.get("huber_weight", 0.5),
        mae_weight=loss_cfg.get("mae_weight", 0.5),
        huber_delta=loss_cfg.get("huber_delta", 1.0),
    ).to(device)

    # ── 评估 ──────────────────────────────────────────────
    sample_rate = seq_cfg.get("sample_rate_hz", 500)
    horizon_seconds = [h / sample_rate for h in horizon_steps]

    model.eval()
    n_horizons = len(horizon_steps)
    n_joints = 12

    ae_sum_per_joint_horizon = torch.zeros(n_joints, n_horizons, device=device)
    count_per_joint = torch.zeros(n_joints, device=device)
    ae_sum_per_horizon = torch.zeros(n_horizons, device=device)
    max_ae = 0.0
    total_loss = 0.0
    n_batches = 0
    n_samples = 0

    with torch.no_grad():
        for x, ji, y in loader:
            x, ji, y = x.to(device), ji.to(device), y.to(device)
            pred = model(x, ji)
            total_loss += criterion(pred, y, ji).item()
            n_batches += 1

            ae = (pred - y).abs()  # (B, H)
            ae_sum_per_horizon += ae.sum(dim=0)
            batch_max = ae.max().item()
            if batch_max > max_ae:
                max_ae = batch_max

            for j in range(n_joints):
                mask = ji == j
                if mask.any():
                    ae_sum_per_joint_horizon[j] += ae[mask].sum(dim=0)
                    count_per_joint[j] += mask.sum()

            n_samples += x.size(0)

    safe_count = count_per_joint.clamp(min=1)
    mae_per_joint_horizon = ae_sum_per_joint_horizon / safe_count.unsqueeze(1)  # (12, H)
    mae_per_horizon = ae_sum_per_horizon / max(n_samples, 1)  # (H,)
    active = count_per_joint > 0
    mae_15s_eq = mae_per_joint_horizon[active, -1].mean().item() if active.any() else float("inf")
    avg_loss = total_loss / max(n_batches, 1)

    # ── 输出报告 ──────────────────────────────────────────
    gate_threshold = cfg.get("acceptance", {}).get("gate_threshold_celsius", 1.5)
    gate_pass = mae_15s_eq < gate_threshold

    print(f"\n{'='*60}")
    print(f"  评估报告 — split={args.split}  samples={n_samples}")
    print(f"{'='*60}")
    print(f"  Gate 指标 (MAE@15s equal-weight): {mae_15s_eq:.4f}°C  {'PASS' if gate_pass else 'FAIL'} (阈值 {gate_threshold}°C)")
    print(f"  平均损失: {avg_loss:.4f}")
    print(f"  最大绝对误差: {max_ae:.2f}°C")
    print()

    print("  各 Horizon 全局 MAE:")
    for i, (hs, sec) in enumerate(zip(horizon_steps, horizon_seconds)):
        print(f"    {sec:6.1f}s (step {hs:5d}):  {mae_per_horizon[i].item():.4f}°C")
    print()

    print("  各关节 MAE@15s:")
    for j in range(n_joints):
        cnt = int(count_per_joint[j].item())
        val = mae_per_joint_horizon[j, -1].item() if cnt > 0 else float("nan")
        print(f"    joint {j:2d}:  {val:.4f}°C  (n={cnt})")
    print()

    print("  各关节 × 各 Horizon MAE 矩阵 (°C):")
    header = "  joint \\ sec " + "".join(f"{s:>7.1f}" for s in horizon_seconds)
    print(header)
    for j in range(n_joints):
        row = f"    j{j:<2d}       "
        for i in range(n_horizons):
            v = mae_per_joint_horizon[j, i].item() if count_per_joint[j] > 0 else float("nan")
            row += f"{v:>7.3f}"
        print(row)
    print(f"{'='*60}\n")

    # ── 保存 JSON ─────────────────────────────────────────
    if args.output:
        result = {
            "split": args.split,
            "checkpoint": args.checkpoint,
            "n_samples": n_samples,
            "gate_mae_15s_equal_weight": mae_15s_eq,
            "gate_pass": gate_pass,
            "gate_threshold": gate_threshold,
            "avg_loss": avg_loss,
            "max_ae": max_ae,
            "horizon_steps": horizon_steps,
            "horizon_seconds": horizon_seconds,
            "mae_per_horizon": mae_per_horizon.cpu().tolist(),
            "mae_per_joint_15s": mae_per_joint_horizon[:, -1].cpu().tolist(),
            "mae_per_joint_horizon": mae_per_joint_horizon.cpu().tolist(),
            "count_per_joint": count_per_joint.cpu().tolist(),
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log.info("results saved → %s", out_path)


if __name__ == "__main__":
    main()
