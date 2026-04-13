"""按 bag 目录名选择 ct_scale profile（多版本快照）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tienkung_thermal.bags.mapping import CAN_TO_DEPLOY_J, CAN_TO_T_LEG


def load_ct_scale_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "profiles" not in data:
        raise ValueError(f"无效的 ct_scale 配置: {path}")
    return data


def ct_scale_deploy_to_t_leg(ct_deploy: list[float]) -> np.ndarray:
    """Deploy 顺序 12 维 → Ultra T_leg[0..11] 顺序 12 维（按 CAN 映射置换）。"""
    if len(ct_deploy) != 12:
        raise ValueError(f"ct_scale_deploy_leg 须为长度 12，得到 {len(ct_deploy)}")
    d = np.asarray(ct_deploy, dtype=np.float64)
    t = np.empty(12, dtype=np.float64)
    for can_id, i_ultra in CAN_TO_T_LEG.items():
        j = CAN_TO_DEPLOY_J[can_id]
        t[i_ultra] = d[j]
    return t


def select_profile_for_bag(bag_dir_name: str, data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """返回 (profile_id, profile_dict)。按 profile_rules 顺序匹配 prefix，空前缀为兜底。"""
    rules = data.get("profile_rules") or []
    fallback: tuple[str, dict[str, Any]] | None = None
    for rule in rules:
        prefix = rule.get("prefix", "")
        pid = rule.get("profile")
        if pid is None:
            continue
        prof = data["profiles"].get(pid)
        if prof is None:
            raise KeyError(f"profile_rules 引用未知 profile: {pid}")
        if prefix == "":
            fallback = (str(pid), prof)
            continue
        if bag_dir_name.startswith(prefix):
            return str(pid), prof
    if fallback is not None:
        return fallback
    first = next(iter(data["profiles"].items()))
    return first[0], first[1]


def resolve_ct_scale_t_leg(bag_dir_name: str, config_path: Path) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Ultra 顺序下的 ct_scale 向量 (12,) 与 profile 元数据。"""
    data = load_ct_scale_yaml(config_path)
    pid, prof = select_profile_for_bag(bag_dir_name, data)
    ctd = prof["ct_scale_deploy_leg"]
    t_leg = ct_scale_deploy_to_t_leg(ctd)
    meta = {"profile_id": pid, "profile_description": prof.get("description", "")}
    return t_leg, pid, meta
