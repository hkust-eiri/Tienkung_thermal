#!/usr/bin/env python3
"""将 rosbag2 中 /leg/status 导出为 500 Hz HDF5（纯腿基线）。

依赖：pip install -e ".[rosbag]"（rosbags、h5py）

示例::

  cd /path/to/Tienkung_thermal
  pip install -e ".[rosbag]"
  python scripts/bags/export_leg_status_dataset.py \\
    --bags-root data/bags \\
    --out-dir data/processed/leg_status_500hz \\
    --msg-package /path/to/bodyctrl_msgs \\
    --ct-scale-config configs/ct_scale_profiles.yaml \\
    --manifest data/processed/leg_status_500hz/manifest.csv

单个 bag::

  python scripts/bags/export_leg_status_dataset.py data/bags/rosbag2_xxx \\
    --out-dir data/processed/leg_status_500hz
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_msg_packages(repo: Path) -> list[Path]:
    cands = [
        repo / ".." / "Tienkung" / "ros2ws" / "install" / "bodyctrl_msgs" / "share" / "bodyctrl_msgs",
        Path("/home/js/robot/Tienkung/ros2ws/install/bodyctrl_msgs/share/bodyctrl_msgs"),
    ]
    out: list[Path] = []
    for p in cands:
        pr = p.resolve()
        if (pr / "package.xml").is_file():
            out.append(pr)
    return out


def _find_bag_dirs(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, pattern)]
    return sorted(dirs)


def _manifest_row_from_existing_h5(h5_path: Path) -> dict[str, object] | None:
    """从已导出的 HDF5 读出 manifest 所需字段（用于 --skip-existing 仍追加 manifest）。"""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            mg = f.get("metadata")
            n_grid = int(len(f["timestamps"])) if "timestamps" in f else 0
            if mg is not None:
                n_raw = int(mg.attrs.get("n_raw_messages_leg_status", -1))
                n_valid = int(mg.attrs.get("n_valid_raw_frames", -1))
                n_err = int(mg.attrs.get("n_skipped_error_nonzero", -1))
            else:
                n_raw = n_valid = n_err = -1
            profile = str(f.attrs.get("ct_scale_profile", ""))
        return {
            "ct_scale_profile": profile,
            "n_raw_messages": n_raw,
            "n_valid_raw_frames": n_valid,
            "n_grid_frames_500hz": n_grid,
            "n_skipped_error_nonzero": n_err,
        }
    except OSError:
        return None


def main() -> None:
    repo = _repo_root()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "bags_path",
        type=Path,
        help="rosbag2 目录，或包含多个 rosbag2_* 子目录的根目录",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "data/processed/leg_status_500hz",
        help="输出 HDF5 与 manifest 的目录",
    )
    p.add_argument(
        "--glob",
        default="rosbag2_*",
        help="当 bags_path 为根目录时，匹配子目录名（glob）",
    )
    p.add_argument(
        "--msg-package",
        type=Path,
        action="append",
        default=[],
        metavar="PKG_ROOT",
        help="bodyctrl_msgs 包根（含 package.xml）。可重复。默认尝试 Tienkung/ros2ws/install/...",
    )
    p.add_argument(
        "--ct-scale-config",
        type=Path,
        default=repo / "configs/ct_scale_profiles.yaml",
        help="ct_scale 多版本配置",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="追加写入 manifest CSV（默认: <out-dir>/manifest.csv）",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标 HDF5 已存在则跳过",
    )
    p.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="开始前删除已有 manifest（便于全量重跑）",
    )
    args = p.parse_args()

    msg_roots = args.msg_package or _default_msg_packages(repo)
    if not msg_roots:
        raise SystemExit(
            "未找到 bodyctrl_msgs。请用 --msg-package 指定含 package.xml 的包根目录 "
            "（例如 Tienkung/ros2ws/install/bodyctrl_msgs/share/bodyctrl_msgs）。"
        )

    bags_path = args.bags_path.expanduser().resolve()
    if not bags_path.exists():
        raise SystemExit(f"路径不存在: {bags_path}")

    if (bags_path / "metadata.yaml").is_file():
        bag_dirs = [bags_path]
    else:
        bag_dirs = _find_bag_dirs(bags_path, args.glob)
        if not bag_dirs:
            raise SystemExit(f"在 {bags_path} 下未找到匹配 {args.glob!r} 的子目录")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = out_dir / "manifest.csv"
    else:
        manifest_path = manifest_path.expanduser().resolve()

    if args.overwrite_manifest and manifest_path.is_file():
        manifest_path.unlink()
        print(f"已删除旧 manifest: {manifest_path}", file=sys.stderr)

    from tienkung_thermal.bags.bag_dir import rosbag2_dir_status
    from tienkung_thermal.bags.pipeline import export_bag_to_hdf5, sanitize_session_id

    ct_cfg = args.ct_scale_config.expanduser().resolve()
    if not ct_cfg.is_file():
        raise SystemExit(f"找不到 ct_scale 配置: {ct_cfg}")

    manifest_new = not manifest_path.is_file()
    export_ts = datetime.now(timezone.utc).isoformat()

    for bag_dir in bag_dirs:
        ok, reason = rosbag2_dir_status(bag_dir)
        if not ok:
            print(f"跳过（{reason}）: {bag_dir}", file=sys.stderr)
            continue

        sid = sanitize_session_id(bag_dir.name)
        out_h5 = out_dir / f"{sid}.h5"

        fieldnames = [
            "export_timestamp_utc",
            "session_id",
            "source_bag",
            "hdf5_path",
            "ct_scale_config",
            "ct_scale_profile",
            "n_raw_messages",
            "n_valid_raw_frames",
            "n_grid_frames_500hz",
            "n_skipped_error_nonzero",
            "split",
        ]

        if args.skip_existing and out_h5.is_file():
            print(f"已存在，跳过导出: {out_h5}", file=sys.stderr)
            cached = _manifest_row_from_existing_h5(out_h5)
            if cached is None:
                print(f"  警告: 无法从 HDF5 读取统计，manifest 不写该行", file=sys.stderr)
                continue
            row = {
                "export_timestamp_utc": export_ts,
                "session_id": sid,
                "source_bag": str(bag_dir.resolve()),
                "hdf5_path": str(out_h5.resolve()),
                "ct_scale_config": str(ct_cfg),
                "ct_scale_profile": cached["ct_scale_profile"],
                "n_raw_messages": cached["n_raw_messages"],
                "n_valid_raw_frames": cached["n_valid_raw_frames"],
                "n_grid_frames_500hz": cached["n_grid_frames_500hz"],
                "n_skipped_error_nonzero": cached["n_skipped_error_nonzero"],
                "split": "",
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("a", newline="", encoding="utf-8") as fp:
                w = csv.DictWriter(fp, fieldnames=fieldnames)
                if manifest_new:
                    w.writeheader()
                    manifest_new = False
                w.writerow(row)
            continue

        print(f"导出 {bag_dir.name} -> {out_h5.name} ...", file=sys.stderr)
        try:
            stats = export_bag_to_hdf5(
                bag_dir,
                out_h5,
                msg_roots,
                ct_cfg,
                log=sys.stderr,
            )
        except Exception as err:  # noqa: BLE001
            print(f"失败 {bag_dir}: {err}", file=sys.stderr)
            continue

        print(
            f"  有效帧 {stats.n_valid_raw} / 消息 {stats.n_messages_total}, "
            f"跳过 error≠0: {stats.n_skipped_error_nonzero}",
            file=sys.stderr,
        )

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "export_timestamp_utc": export_ts,
            "session_id": sid,
            "source_bag": str(bag_dir.resolve()),
            "hdf5_path": str(out_h5.resolve()),
            "ct_scale_config": str(ct_cfg),
            "ct_scale_profile": stats.ct_scale_profile_id,
            "n_raw_messages": stats.n_messages_total,
            "n_valid_raw_frames": stats.n_valid_raw,
            "n_grid_frames_500hz": stats.n_grid_frames,
            "n_skipped_error_nonzero": stats.n_skipped_error_nonzero,
            "split": "",
        }
        with manifest_path.open("a", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            if manifest_new:
                w.writeheader()
                manifest_new = False
            w.writerow(row)

    print(f"完成。manifest: {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
