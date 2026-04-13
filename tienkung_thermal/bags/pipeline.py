"""从 rosbag2 读取 /leg/status，清洗、500 Hz 重采样并写入 HDF5。"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tienkung_thermal.bags.mapping import CAN_TO_T_LEG
from tienkung_thermal.bags.ct_scale_config import resolve_ct_scale_t_leg
from tienkung_thermal.bags.rosbags_types import make_humble_typestore

# 与 configs/leg_index_mapping.yaml motor_names 一致
MOTOR_NAMES_ULTRA: tuple[str, ...] = (
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "hip_pitch_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
)

DT_GRID_S = 0.002  # 500 Hz
SAMPLE_RATE_HZ = 500.0


@dataclass
class ExportStats:
    n_messages_total: int = 0
    n_skipped_bad_status_len: int = 0
    n_skipped_unknown_can: int = 0
    n_skipped_error_nonzero: int = 0
    n_skipped_incomplete_12: int = 0
    n_valid_raw: int = 0
    n_grid_frames: int = 0
    ct_scale_profile_id: str = ""


def _stamp_to_sec(msg: Any, fallback_ts_ns: int | None = None) -> float | None:
    try:
        h = getattr(msg, "header", None)
        if h is None:
            raise AttributeError
        st = getattr(h, "stamp", None)
        if st is None:
            raise AttributeError
        return float(st.sec) + float(st.nanosec) * 1e-9
    except Exception:
        if fallback_ts_ns is not None:
            return float(fallback_ts_ns) * 1e-9
        return None


def parse_motor_status_msg_to_row(
    msg: Any,
    ct_scale_per_t_leg: np.ndarray,
    stats: ExportStats,
    fallback_ts_ns: int | None = None,
) -> tuple[float, dict[str, np.ndarray]] | None:
    """单条 MotorStatusMsg → (t_sec, arrays) 或 None。"""
    status = getattr(msg, "status", None)
    if not status:
        stats.n_skipped_bad_status_len += 1
        return None
    if len(status) != 12:
        stats.n_skipped_bad_status_len += 1
        return None

    q = np.full(12, np.nan, dtype=np.float64)
    dq = np.full(12, np.nan, dtype=np.float64)
    cur = np.full(12, np.nan, dtype=np.float64)
    temp = np.full(12, np.nan, dtype=np.float64)
    volt = np.full(12, np.nan, dtype=np.float64)
    err = np.zeros(12, dtype=np.int64)

    seen: set[int] = set()
    for m in status:
        try:
            can_id = int(getattr(m, "name", -1))
        except (TypeError, ValueError):
            stats.n_skipped_unknown_can += 1
            return None
        if can_id not in CAN_TO_T_LEG:
            stats.n_skipped_unknown_can += 1
            return None
        i = CAN_TO_T_LEG[can_id]
        if i in seen:
            stats.n_skipped_incomplete_12 += 1
            return None
        seen.add(i)
        try:
            e = int(getattr(m, "error", 0))
        except (TypeError, ValueError):
            e = 0
        err[i] = e
        q[i] = float(getattr(m, "pos", 0.0))
        dq[i] = float(getattr(m, "speed", 0.0))
        cur[i] = float(getattr(m, "current", 0.0))
        temp[i] = float(getattr(m, "temperature", 0.0))
        volt[i] = float(getattr(m, "voltage", 0.0))

    if len(seen) != 12:
        stats.n_skipped_incomplete_12 += 1
        return None
    if np.any(err != 0):
        stats.n_skipped_error_nonzero += 1
        return None

    t_sec = _stamp_to_sec(msg, fallback_ts_ns)
    if t_sec is None:
        stats.n_skipped_bad_status_len += 1
        return None

    tau = cur * ct_scale_per_t_leg
    row = {
        "q": q,
        "dq": dq,
        "current": cur,
        "temperature": temp,
        "voltage": volt,
        "tau_est": tau,
    }
    stats.n_valid_raw += 1
    return t_sec, row


def _dedupe_time_sort(
    t: np.ndarray, arrs: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """按时间排序；同一时间戳保留最后一条。"""
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    for k in arrs:
        arrs[k] = arrs[k][order]
    if len(t) <= 1:
        return t, arrs
    keep = np.ones(len(t), dtype=bool)
    for i in range(len(t) - 1):
        if t[i] == t[i + 1]:
            keep[i] = False
    t = t[keep]
    for k in arrs:
        arrs[k] = arrs[k][keep]
    return t, arrs


def resample_arrays_to_grid(
    t: np.ndarray,
    arrs: dict[str, np.ndarray],
    dt: float = DT_GRID_S,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """线性插值到 [t0, t1] 上均匀网格，步长 dt（秒）。"""
    if len(t) < 2:
        raise ValueError("有效样本不足 2，无法重采样")

    t0 = float(t[0])
    t1 = float(t[-1])
    grid = np.arange(t0, t1 + 0.5 * dt, dt, dtype=np.float64)
    if len(grid) < 2:
        raise ValueError("网格长度过短")

    out: dict[str, np.ndarray] = {}
    for k, v in arrs.items():
        out_k = np.empty((len(grid), 12), dtype=np.float64)
        for j in range(12):
            out_k[:, j] = np.interp(grid, t, v[:, j])
        out[k] = out_k

    dq = out["dq"]
    dq_abs = np.abs(dq)
    tau_est = out["tau_est"]
    tau_sq = tau_est * tau_est
    ddq = np.diff(dq, axis=0) / dt
    ddq_abs = np.abs(ddq)
    ddq_abs = np.vstack([ddq_abs[0:1, :], ddq_abs])

    out["dq_abs"] = dq_abs
    out["tau_sq"] = tau_sq
    out["ddq_abs"] = ddq_abs
    return grid, out


def export_bag_to_hdf5(
    bag_dir: Path,
    out_h5: Path,
    msg_package_roots: Sequence[Path],
    ct_scale_config: Path,
    log: Any = None,
) -> ExportStats:
    """导出单个 rosbag2 目录到 HDF5。"""
    log = log or sys.stderr
    bag_name = bag_dir.name
    ct_scale_per_t_leg, profile_id, prof_meta = resolve_ct_scale_t_leg(bag_name, ct_scale_config)

    stats = ExportStats()
    t_list: list[float] = []
    acc: dict[str, list[np.ndarray]] = {
        "q": [],
        "dq": [],
        "current": [],
        "temperature": [],
        "voltage": [],
        "tau_est": [],
    }

    from rosbags.highlevel import AnyReader

    typestore = make_humble_typestore(list(msg_package_roots))

    topic = "/leg/status"
    with AnyReader([bag_dir.resolve()], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise RuntimeError(f"{bag_dir} 中无 topic {topic}")
        for conn, ts, raw in reader.messages(connections=conns):
            stats.n_messages_total += 1
            try:
                msg = reader.deserialize(raw, conn.msgtype)
            except Exception as err:  # noqa: BLE001
                print(f"警告: 反序列化失败 @ts={ts}: {err}", file=log)
                continue
            parsed = parse_motor_status_msg_to_row(
                msg, ct_scale_per_t_leg, stats, fallback_ts_ns=int(ts)
            )
            if parsed is None:
                continue
            t_sec, row = parsed
            t_list.append(t_sec)
            for k in acc:
                acc[k].append(row[k])

    if stats.n_valid_raw < 2:
        raise RuntimeError(
            f"有效帧不足: valid={stats.n_valid_raw}, total_msgs={stats.n_messages_total}"
        )

    t_np = np.asarray(t_list, dtype=np.float64)
    arrs_np = {k: np.stack(v, axis=0) for k, v in acc.items()}
    t_np, arrs_np = _dedupe_time_sort(t_np, arrs_np)

    grid, resampled = resample_arrays_to_grid(t_np, arrs_np, DT_GRID_S)

    stats.n_grid_frames = int(len(grid))
    stats.ct_scale_profile_id = profile_id

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    _write_hdf5(
        out_h5,
        grid,
        resampled,
        bag_dir=bag_dir,
        profile_id=profile_id,
        prof_meta=prof_meta,
        ct_scale_per_t_leg=ct_scale_per_t_leg,
        stats=stats,
        ct_scale_config_path=ct_scale_config,
    )
    return stats


def _write_hdf5(
    out_h5: Path,
    grid: np.ndarray,
    resampled: dict[str, np.ndarray],
    bag_dir: Path,
    profile_id: str,
    prof_meta: dict[str, Any],
    ct_scale_per_t_leg: np.ndarray,
    stats: ExportStats,
    ct_scale_config_path: Path,
) -> None:
    import h5py

    export_ts = datetime.now(timezone.utc).isoformat()
    with h5py.File(out_h5, "w") as f:
        f.attrs["export_timestamp_utc"] = export_ts
        f.attrs["source_rosbag"] = str(bag_dir.resolve())
        f.attrs["sample_rate_hz"] = SAMPLE_RATE_HZ
        f.attrs["dt_grid_s"] = DT_GRID_S
        f.attrs["t_leg_order"] = ", ".join(MOTOR_NAMES_ULTRA)
        f.attrs["ct_scale_profile"] = profile_id
        f.attrs["ct_scale_config_path"] = str(ct_scale_config_path.resolve())
        f.attrs["ct_scale_per_t_leg_json"] = str([float(x) for x in ct_scale_per_t_leg])
        if prof_meta.get("profile_description"):
            f.attrs["ct_scale_profile_description"] = prof_meta["profile_description"]

        g_ts = f.create_dataset("timestamps", data=grid, compression="gzip", shuffle=True)
        g_ts.attrs["unit"] = "s"
        g_ts.attrs["description"] = "header.stamp 线性插值后的统一时间轴（相对会话内首帧可再离线处理）"

        gj = f.create_group("joints")
        for key in (
            "q",
            "dq",
            "current",
            "temperature",
            "voltage",
            "tau_est",
            "tau_sq",
            "dq_abs",
            "ddq_abs",
        ):
            gj.create_dataset(key, data=resampled[key], compression="gzip", shuffle=True)

        meta = f.create_group("metadata")
        meta.attrs["n_raw_messages_leg_status"] = stats.n_messages_total
        meta.attrs["n_valid_raw_frames"] = stats.n_valid_raw
        meta.attrs["n_skipped_bad_status_len"] = stats.n_skipped_bad_status_len
        meta.attrs["n_skipped_unknown_can"] = stats.n_skipped_unknown_can
        meta.attrs["n_skipped_error_nonzero"] = stats.n_skipped_error_nonzero
        meta.attrs["n_skipped_incomplete_12"] = stats.n_skipped_incomplete_12
        meta.attrs["n_grid_frames"] = len(grid)


def sanitize_session_id(name: str) -> str:
    """目录名 → 安全文件名。"""
    s = re.sub(r"[^\w.\-]+", "_", name)
    return s.strip("_") or "session"
