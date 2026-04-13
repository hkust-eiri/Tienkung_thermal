#!/usr/bin/env python3
"""从 ROS 2 rosbag2（sqlite3 .db3）导出少量 /leg/status 与 /leg/motor_status 样本，便于确认字段与类型。

两种模式：

1) **仅原始字节（默认，仅需 Python 标准库）**
   从 bag 内 `topics` 表读取类型名，从 `messages` 表取前 N 条：时间戳、payload 长度、payload 十六进制前缀。
   适合：尚未拉取 `bodyctrl_msgs` 源码、或只想快速核对 bag 内类型字符串与消息密度。

2) **解码为结构化字典（需 `rosbags` + `bodyctrl_msgs` 的 .msg 源码目录）**
   pip install rosbags
   并传入 `--msg-package` 指向 ROS 2 包根目录（内含 `package.xml` 与 `msg/*.msg`），例如 TienKung_ROS 里的
   `.../src/bodyctrl_msgs`。会合并 `rosbags` 自带的 ROS 2 Humble 内置类型后注册该包，再 CDR 反序列化。

示例：

  # 只看类型与原始帧头（推荐先做这一步）
  python3 scripts/bags/extract_bag_topic_samples.py \\
    /path/to/rosbag2_xxx --per-topic 3 --out /tmp/samples_raw.json

  # 在克隆了 bodyctrl_msgs 后解码（路径按你本机仓库调整）
  python3 scripts/bags/extract_bag_topic_samples.py \\
    /path/to/rosbag2_xxx --per-topic 2 --decode \\
    --msg-package /path/to/ws/src/bodyctrl_msgs \\
    --out /tmp/samples_decoded.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

DEFAULT_TOPICS = ("/leg/status", "/leg/motor_status")


def _find_db3(bag_dir: Path) -> Path:
    dbs = sorted(bag_dir.glob("*.db3"))
    if not dbs:
        raise SystemExit(f"在 {bag_dir} 下未找到 .db3 文件")
    if len(dbs) > 1:
        raise SystemExit(f"目录中存在多个 .db3，请指定唯一 bag 目录: {dbs}")
    return dbs[0]


def _package_name_from_xml(package_root: Path) -> str:
    xml = package_root / "package.xml"
    if not xml.is_file():
        raise ValueError(f"不是 ROS 包根目录（缺少 package.xml）: {package_root}")
    text = xml.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"<name>\s*([^<]+?)\s*</name>", text)
    if not m:
        raise ValueError(f"无法从 package.xml 解析 <name>: {xml}")
    return m.group(1).strip()


def _collect_types_from_package(package_root: Path) -> dict[str, Any]:
    try:
        from rosbags.typesys.msg import get_types_from_msg
    except ImportError as e:
        raise SystemExit(
            "解码模式需要安装 rosbags：pip install 'rosbags>=0.9'"
        ) from e

    pkg = _package_name_from_xml(package_root)
    msg_dir = package_root / "msg"
    if not msg_dir.is_dir():
        raise ValueError(f"包内无 msg 目录: {package_root}")

    typs: dict[str, Any] = {}
    for f in sorted(msg_dir.glob("*.msg")):
        typname = f"{pkg}/msg/{f.stem}"
        typs.update(get_types_from_msg(f.read_text(encoding="utf-8"), typname))
    return typs


def _msg_to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {"_bytes_len": len(obj), "_hex_prefix": bytes(obj[:48]).hex()}
    if isinstance(obj, list):
        return [_msg_to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [_msg_to_jsonable(x) for x in obj]
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass
    slots = getattr(obj, "__slots__", None)
    if slots:
        return {k: _msg_to_jsonable(getattr(obj, k)) for k in slots if not k.startswith("_")}
    if hasattr(obj, "__dict__"):
        return {
            k: _msg_to_jsonable(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return str(obj)


def export_raw_sqlite(
    db_path: Path,
    topics: list[str],
    per_topic: int,
) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name, type, serialization_format FROM topics")
        all_topics = [
            {"id": r[0], "name": r[1], "type": r[2], "serialization_format": r[3]}
            for r in cur.fetchall()
        ]
        by_name = {t["name"]: t for t in all_topics}

        samples: dict[str, Any] = {}
        for topic in topics:
            if topic not in by_name:
                samples[topic] = {
                    "error": "bag 中不存在该 topic",
                    "available_topic_names": sorted(by_name.keys()),
                }
                continue
            tid = by_name[topic]["id"]
            cur.execute(
                """
                SELECT timestamp, LENGTH(data) AS nbytes, data
                FROM messages
                WHERE topic_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (tid, per_topic),
            )
            rows = cur.fetchall()
            decoded_rows = []
            for ts, nbytes, blob in rows:
                b = bytes(blob) if blob is not None else b""
                decoded_rows.append(
                    {
                        "timestamp_ns": ts,
                        "payload_bytes": nbytes,
                        "payload_hex_prefix_64": b[:64].hex(),
                    }
                )
            samples[topic] = {
                "ros_type": by_name[topic]["type"],
                "serialization_format": by_name[topic]["serialization_format"],
                "sample_count_in_export": len(decoded_rows),
                "samples": decoded_rows,
            }
        return {
            "mode": "raw_sqlite",
            "db3": str(db_path.resolve()),
            "topics_table": all_topics,
            "samples_by_topic": samples,
        }
    finally:
        conn.close()


def export_decoded_rosbags(
    bag_dir: Path,
    topics: list[str],
    per_topic: int,
    msg_package_roots: list[Path],
) -> dict[str, Any]:
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as e:
        raise SystemExit(
            "解码模式需要安装 rosbags：pip install 'rosbags>=0.9'"
        ) from e

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    for root in msg_package_roots:
        typestore.register(_collect_types_from_package(root.resolve()))

    counts = {t: 0 for t in topics}
    cap = {t: per_topic for t in topics}
    out_samples: dict[str, list[Any]] = {t: [] for t in topics}

    with AnyReader([bag_dir.resolve()], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        if not conns:
            raise SystemExit(
                f"所选 topic 在 bag 中无连接：请求={topics}，实际={sorted({c.topic for c in reader.connections})}"
            )
        for conn, ts, raw in reader.messages(connections=conns):
            topic = conn.topic
            if counts[topic] >= cap[topic]:
                if all(counts[t] >= cap[t] for t in topics):
                    break
                continue
            try:
                msg = reader.deserialize(raw, conn.msgtype)
                payload = _msg_to_jsonable(msg)
            except Exception as err:  # noqa: BLE001
                payload = {"_deserialize_error": repr(err)}
            out_samples[topic].append(
                {
                    "timestamp_ns": ts,
                    "msgtype": conn.msgtype,
                    "msg": payload,
                }
            )
            counts[topic] += 1
            if all(counts[t] >= cap[t] for t in topics):
                break

    return {
        "mode": "decoded_cdr",
        "bag_dir": str(bag_dir.resolve()),
        "msg_packages": [str(p.resolve()) for p in msg_package_roots],
        "samples_by_topic": out_samples,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "bag_dir",
        type=Path,
        help="rosbag2 目录（含 metadata.yaml 与单个 .db3）",
    )
    p.add_argument(
        "--topics",
        nargs="+",
        default=list(DEFAULT_TOPICS),
        help="要导出的 topic 列表",
    )
    p.add_argument(
        "--per-topic",
        type=int,
        default=3,
        help="每个 topic 最多导出多少条消息",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 JSON 路径（默认打印到 stdout）",
    )
    p.add_argument(
        "--decode",
        action="store_true",
        help="使用 rosbags + --msg-package 做 CDR 解码（否则仅导出 sqlite 原始前缀）",
    )
    p.add_argument(
        "--msg-package",
        type=Path,
        action="append",
        default=[],
        metavar="PKG_ROOT",
        help="可重复。ROS 2 消息包根目录（含 package.xml 与 msg/）。解码 bodyctrl 时至少传 bodyctrl_msgs",
    )
    args = p.parse_args()

    bag_dir = args.bag_dir.expanduser().resolve()
    if not bag_dir.is_dir():
        raise SystemExit(f"不是目录: {bag_dir}")

    topics = args.topics

    if args.decode:
        if not args.msg_package:
            raise SystemExit("使用 --decode 时必须至少指定一次 --msg-package（例如 bodyctrl_msgs 包根目录）")
        payload = export_decoded_rosbags(
            bag_dir, topics, args.per_topic, args.msg_package
        )
    else:
        db_path = _find_db3(bag_dir)
        payload = export_raw_sqlite(db_path, topics, args.per_topic)

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
