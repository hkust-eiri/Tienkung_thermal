"""Rosbag2 → HDF5 腿部热建模数据集导出（仅 /leg/status）。"""

from tienkung_thermal.bags.mapping import CAN_TO_DEPLOY_J, CAN_TO_T_LEG

__all__ = ["CAN_TO_DEPLOY_J", "CAN_TO_T_LEG"]
