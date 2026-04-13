"""rosbag2 目录可用性检查（与 rosbags 打开条件对齐）。"""

from __future__ import annotations

from pathlib import Path


def rosbag2_dir_status(bag_dir: Path) -> tuple[bool, str]:
    """若可交给 ``AnyReader([bag_dir])`` 打开则返回 (True, "")；否则 (False, 原因)。"""
    bag_dir = bag_dir.resolve()
    meta = bag_dir / "metadata.yaml"
    if not meta.is_file():
        return False, "缺少 metadata.yaml"
    try:
        text = meta.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as e:
        return False, f"无法读取 metadata.yaml: {e}"
    if not text:
        return False, "metadata.yaml 为空（rosbags 无法解析；请补全或重新录制）"
    if "rosbag2_bagfile_information" not in text:
        return False, "metadata.yaml 缺少 rosbag2_bagfile_information 字段"
    dbs = sorted(bag_dir.glob("*.db3"))
    if not dbs:
        return False, "目录下无 .db3"
    return True, ""
