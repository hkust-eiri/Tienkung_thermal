"""CAN ID（MotorStatus.name）→ Ultra T_leg 下标与 Deploy 腿向量下标 j。

权威表见 docs/plan.md §1.2.1；与 scripts/check/p0_check.py 中 DEPLOY_LEG_IDS / 语义名一致。
"""

from __future__ import annotations

# 左腿 51–56，右腿 61–66 → Ultra T_leg[i]（hip R–Y–P 每侧）
CAN_TO_T_LEG: dict[int, int] = {
    51: 0,
    52: 2,
    53: 1,
    54: 3,
    55: 4,
    56: 5,
    61: 6,
    62: 8,
    63: 7,
    64: 9,
    65: 10,
    66: 11,
}

# 同一 CAN → Deploy bodyIdMap 腿中间向量下标 j（l_hip_roll=0 … r_ankle_roll=11），用于 ct_scale[j]
CAN_TO_DEPLOY_J: dict[int, int] = {
    51: 0,
    52: 1,
    53: 2,
    54: 3,
    55: 4,
    56: 5,
    61: 6,
    62: 7,
    63: 8,
    64: 9,
    65: 10,
    66: 11,
}

