#!/usr/bin/env python3
"""对单个 HDF5 session 做推理，输出指定关节的多视距温度预测。

支持两种模式：
  1. 单窗口推理：指定 --start-frame，输出一条预测
  2. 滑窗推理：沿整个 session 滑动，输出 CSV 时间序列

用法::

    # 单窗口：从第 10000 帧开始，预测 joint 3
    python scripts/inference.py \\
        --checkpoint checkpoints/best_ultra_thermal.pt \\
        --h5 data/processed/leg_status_500hz/rosbag2_2026_04_07-10_50_21.h5 \\
        --joint 3 --start-frame 10000

    # 滑窗：输出整条 session 的预测 CSV
    python scripts/inference.py \\
        --checkpoint checkpoints/best_ultra_thermal.pt \\
        --h5 data/processed/leg_status_500hz/rosbag2_2026_04_07-10_50_21.h5 \\
        --joint 3 --sliding --output pred_j3.csv
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with UltraThermalLSTM")
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pt 路径")
    parser.add_argument("--config", default="configs/ultra_thermal_lstm.yaml")
    parser.add_argument("--h5", required=True, help="单个 HDF5 session 路径")
    parser.add_argument("--joint", type=int, required=True, help="关节编号 0-11")
    parser.add_argument("--start-frame", type=int, default=None, help="单窗口模式：起始帧号")
    parser.add_argument("--sliding", action="store_true", help="滑窗模式：沿整个 session 输出预测")
    parser.add_argument("--stride", type=int, default=None, help="滑窗步长（帧），默认用 YAML 配置")
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="CSV 输出路径（滑窗模式）")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if not args.sliding and args.start_frame is None:
        parser.error("单窗口模式需要 --start-frame；或使用 --sliding 进行滑窗推理")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("inference")

    cfg = _load_config(args.config)
    model_cfg = cfg["model"]
    seq_cfg = cfg["sequence"]
    feat_cfg = cfg["features"]
    train_cfg = cfg["training"]
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

    import numpy as np
    import torch

    from tienkung_thermal.data.dataset import UltraThermalDataset
    from tienkung_thermal.data.norm import load_norm_stats, stats_to_tensors
    from tienkung_thermal.models.thermal_lstm import UltraThermalLSTM

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

    # ── 模型 ──────────────────────────────────────────────
    device = torch.device(args.device or train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = UltraThermalLSTM(
        input_dim=d,
        proj_dim=model_cfg.get("proj_dim", 32),
        hidden_dim=model_cfg.get("hidden_dim", 96),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=0.0,
        mid_dim=model_cfg.get("mid_dim", 64),
        horizon=model_cfg.get("horizon", 9),
        n_joints=model_cfg.get("n_joints", 12),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("loaded checkpoint epoch=%s", ckpt.get("epoch"))

    # ── 数据集 ────────────────────────────────────────────
    horizon_steps = cfg.get("horizon_steps", [250, 500, 1000, 1500, 2500, 3500, 5000, 6000, 7500])
    seq_len = seq_cfg.get("seq_len", 2500)
    stride = args.stride or seq_cfg.get("stride", 50)
    sample_rate = seq_cfg.get("sample_rate_hz", 500)
    horizon_seconds = [h / sample_rate for h in horizon_steps]

    ds = UltraThermalDataset(
        [args.h5],
        seq_len=seq_len,
        horizon_steps=horizon_steps,
        use_derived=use_derived,
        use_adjacent_temp=use_adj,
        use_imu=use_imu,
        stride=stride,
        norm_stats=norm_stats_tensors,
        log1p_fields=log1p_fields,
    )

    if len(ds) == 0:
        log.error("dataset is empty — session too short for seq_len=%d + max_horizon=%d", seq_len, max(horizon_steps))
        sys.exit(1)

    joint = args.joint
    if joint < 0 or joint > 11:
        log.error("joint must be 0-11, got %d", joint)
        sys.exit(1)

    # ── 单窗口推理 ────────────────────────────────────────
    if not args.sliding:
        cache = ds._caches[0]
        max_start = cache.n_frames - seq_len - ds.max_horizon
        if args.start_frame < 0 or args.start_frame > max_start:
            log.error("start-frame must be in [0, %d], got %d", max_start, args.start_frame)
            sys.exit(1)

        sl = slice(args.start_frame, args.start_frame + seq_len)
        feature_cols = []
        for field in ds.RAW_FIELDS:
            feature_cols.append(cache.joints[field][sl, joint])
        if ds.use_derived:
            for field in ds.DERIVED_FIELDS:
                feature_cols.append(cache.joints[field][sl, joint])
        x = np.stack(feature_cols, axis=-1)
        x_t = torch.from_numpy(x)
        if ds._log1p_indices:
            for idx in ds._log1p_indices:
                x_t[:, idx] = torch.log1p(x_t[:, idx].abs())
        if ds.norm_stats is not None:
            x_t = (x_t - ds.norm_stats["mean"]) / (ds.norm_stats["std"] + 1e-8)

        x_t = x_t.unsqueeze(0).to(device)
        ji = torch.tensor([joint], dtype=torch.long, device=device)

        with torch.no_grad():
            pred = model(x_t, ji)  # (1, H)

        target_idx = args.start_frame + seq_len
        actual = [cache.joints["temperature"][target_idx + h - 1, joint] for h in horizon_steps]

        print(f"\n  推理结果 — joint={joint}  start_frame={args.start_frame}")
        print(f"  {'Horizon':>10s}  {'预测(°C)':>10s}  {'实际(°C)':>10s}  {'误差(°C)':>10s}")
        for i, sec in enumerate(horizon_seconds):
            p = pred[0, i].item()
            a = float(actual[i])
            print(f"  {sec:>9.1f}s  {p:>10.3f}  {a:>10.3f}  {abs(p - a):>10.3f}")
        print()
        return

    # ── 滑窗推理 ──────────────────────────────────────────
    cache = ds._caches[0]
    max_start = cache.n_frames - seq_len - ds.max_horizon
    n_windows = (max_start + stride - 1) // stride
    log.info("sliding inference: joint=%d  windows=%d  stride=%d", joint, n_windows, stride)

    rows = []
    batch_size = 64
    starts = list(range(0, max_start, stride))

    for batch_begin in range(0, len(starts), batch_size):
        batch_starts = starts[batch_begin : batch_begin + batch_size]
        xs = []
        for st in batch_starts:
            sl = slice(st, st + seq_len)
            feature_cols = []
            for field in ds.RAW_FIELDS:
                feature_cols.append(cache.joints[field][sl, joint])
            if ds.use_derived:
                for field in ds.DERIVED_FIELDS:
                    feature_cols.append(cache.joints[field][sl, joint])
            x = np.stack(feature_cols, axis=-1)
            x_t = torch.from_numpy(x)
            if ds._log1p_indices:
                for idx in ds._log1p_indices:
                    x_t[:, idx] = torch.log1p(x_t[:, idx].abs())
            if ds.norm_stats is not None:
                x_t = (x_t - ds.norm_stats["mean"]) / (ds.norm_stats["std"] + 1e-8)
            xs.append(x_t)

        x_batch = torch.stack(xs, dim=0).to(device)
        ji_batch = torch.full((len(batch_starts),), joint, dtype=torch.long, device=device)

        with torch.no_grad():
            pred_batch = model(x_batch, ji_batch)  # (B, H)

        for k, st in enumerate(batch_starts):
            target_idx = st + seq_len
            time_s = target_idx / sample_rate
            preds = pred_batch[k].cpu().tolist()
            actual_temp = float(cache.joints["temperature"][min(target_idx, cache.n_frames - 1), joint])
            rows.append([time_s, actual_temp] + preds)

    # 输出
    header = ["time_s", "actual_temp_now"] + [f"pred_{s:.1f}s" for s in horizon_seconds]

    if args.output:
        import csv
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        log.info("predictions saved → %s  (%d rows)", out_path, len(rows))
    else:
        print(",".join(header))
        for row in rows[:20]:
            print(",".join(f"{v:.4f}" for v in row))
        if len(rows) > 20:
            print(f"... ({len(rows)} rows total, use --output to save all)")


if __name__ == "__main__":
    main()
