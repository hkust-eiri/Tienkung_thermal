#!/usr/bin/env python3
"""Rebuild missing/empty metadata.yaml for ROS2 bag directories from db3 files.

Usage::

  cd /path/to/Tienkung_thermal
  python scripts/bags/rebuild_metadata.py data/bags/bag0413
  python scripts/bags/rebuild_metadata.py data/bags/bag0413/rosbag2_2026_04_07-16_40_19
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3

import yaml


def get_db3_info(db3_path: str):
    """Extract topic and message info from a single db3 file.
    Returns None if the file is corrupt or has no topics table."""
    try:
        conn = sqlite3.connect(db3_path)
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        if "topics" not in tables or "messages" not in tables:
            conn.close()
            return None

        cur.execute("SELECT id, name, type, serialization_format, offered_qos_profiles FROM topics")
        topics = {row[0]: row[1:] for row in cur.fetchall()}

        msg_counts = {}
        for tid in topics:
            cur.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (tid,))
            msg_counts[tid] = cur.fetchone()[0]

        cur.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM messages")
        row = cur.fetchone()
        min_ts, max_ts, total = row[0], row[1], row[2]

        conn.close()
        return topics, msg_counts, min_ts, max_ts, total
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
        print(f"WARN: {os.path.basename(db3_path)}: {e}")
        return None


def build_metadata(bag_dir: str):
    """Build metadata.yaml content from all db3 files in a bag directory."""
    db3_files = sorted(glob.glob(os.path.join(bag_dir, "*.db3")))
    if not db3_files:
        return None

    all_topics: dict[str, tuple] = {}
    global_msg_counts: dict[str, int] = {}
    global_min_ts = None
    global_max_ts = None
    global_total = 0
    file_entries = []

    for db3_path in db3_files:
        info = get_db3_info(db3_path)
        if info is None:
            file_entries.append({
                "path": os.path.basename(db3_path),
                "starting_time": {"nanoseconds_since_epoch": 0},
                "duration": {"nanoseconds": 0},
                "message_count": 0,
            })
            continue

        topics, msg_counts, min_ts, max_ts, total = info

        for tid, (name, typ, ser_fmt, qos) in topics.items():
            if name not in all_topics:
                all_topics[name] = (typ, ser_fmt, qos)
                global_msg_counts[name] = 0
            global_msg_counts[name] += msg_counts.get(tid, 0)

        if min_ts is not None:
            if global_min_ts is None or min_ts < global_min_ts:
                global_min_ts = min_ts
            if global_max_ts is None or max_ts > global_max_ts:
                global_max_ts = max_ts

        global_total += total

        duration_ns = (max_ts - min_ts) if (min_ts is not None and max_ts is not None) else 0
        file_entries.append({
            "path": os.path.basename(db3_path),
            "starting_time": {"nanoseconds_since_epoch": min_ts or 0},
            "duration": {"nanoseconds": duration_ns},
            "message_count": total,
        })

    global_duration = (global_max_ts - global_min_ts) if (global_min_ts and global_max_ts) else 0

    topics_with_count = []
    for name in sorted(all_topics.keys()):
        typ, ser_fmt, qos = all_topics[name]
        topics_with_count.append({
            "topic_metadata": {
                "name": name,
                "type": typ,
                "serialization_format": ser_fmt,
                "offered_qos_profiles": qos,
            },
            "message_count": global_msg_counts[name],
        })

    relative_paths = [os.path.basename(f) for f in db3_files]

    metadata = {
        "rosbag2_bagfile_information": {
            "version": 5,
            "storage_identifier": "sqlite3",
            "duration": {"nanoseconds": global_duration},
            "starting_time": {"nanoseconds_since_epoch": global_min_ts or 0},
            "message_count": global_total,
            "topics_with_message_count": topics_with_count,
            "compression_format": "",
            "compression_mode": "",
            "relative_file_paths": relative_paths,
            "files": file_entries,
        }
    }
    return metadata


def _is_bag_dir(path: str) -> bool:
    """True if path looks like a single rosbag2_* directory (contains .db3)."""
    return bool(glob.glob(os.path.join(path, "*.db3")))


def run(bag_root: str) -> None:
    bag_root = os.path.abspath(bag_root)

    if _is_bag_dir(bag_root):
        bag_dirs = [bag_root]
    else:
        bag_dirs = sorted(glob.glob(os.path.join(bag_root, "rosbag2_*/")))
        bag_dirs = [d.rstrip("/") for d in bag_dirs]

    if not bag_dirs:
        print(f"No rosbag2_* directories found in {bag_root}")
        return

    rebuilt = 0
    skipped = 0

    for bag_dir in bag_dirs:
        meta_path = os.path.join(bag_dir, "metadata.yaml")
        dirname = os.path.basename(bag_dir)

        needs_rebuild = not os.path.exists(meta_path) or os.path.getsize(meta_path) == 0

        if not needs_rebuild:
            skipped += 1
            continue

        print(f"Rebuilding: {dirname} ...", end=" ")
        metadata = build_metadata(bag_dir)
        if metadata is None:
            print("SKIP (no db3 files)")
            continue

        with open(meta_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

        print(f"OK ({metadata['rosbag2_bagfile_information']['message_count']} messages)")
        rebuilt += 1

    print(f"\nDone. Rebuilt: {rebuilt}, Skipped (already valid): {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild missing/empty metadata.yaml for ROS2 bag directories from db3 files.",
    )
    parser.add_argument(
        "bag_path",
        help="Path to a bags root (containing rosbag2_* subdirs) or a single rosbag2_* directory.",
    )
    args = parser.parse_args()
    run(args.bag_path)


if __name__ == "__main__":
    main()
