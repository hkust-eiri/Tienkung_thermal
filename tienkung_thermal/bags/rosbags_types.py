"""从 ROS 2 消息包目录注册 rosbags 类型（与 scripts/bags/extract_bag_topic_samples.py 同源）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def package_name_from_xml(package_root: Path) -> str:
    xml = package_root / "package.xml"
    if not xml.is_file():
        raise ValueError(f"不是 ROS 包根目录（缺少 package.xml）: {package_root}")
    text = xml.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"<name>\s*([^<]+?)\s*</name>", text)
    if not m:
        raise ValueError(f"无法从 package.xml 解析 <name>: {xml}")
    return m.group(1).strip()


def collect_types_from_package(package_root: Path) -> dict[str, Any]:
    try:
        from rosbags.typesys.msg import get_types_from_msg
    except ImportError as e:
        raise SystemExit(
            "需要安装 rosbags：pip install 'rosbags>=0.9'"
        ) from e

    pkg = package_name_from_xml(package_root)
    msg_dir = package_root / "msg"
    if not msg_dir.is_dir():
        raise ValueError(f"包内无 msg 目录: {package_root}")

    typs: dict[str, Any] = {}
    for f in sorted(msg_dir.glob("*.msg")):
        typname = f"{pkg}/msg/{f.stem}"
        typs.update(get_types_from_msg(f.read_text(encoding="utf-8"), typname))
    return typs


def make_humble_typestore(msg_package_roots: list[Path]) -> Any:
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    for root in msg_package_roots:
        typestore.register(collect_types_from_package(root.resolve()))
    return typestore
