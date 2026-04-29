"""UltraThermalLSTM — 全关节联合因果 LSTM 热预测模型。

架构与超参数约定见 docs/thermal_lstm_modeling.md §4；
配置文件见 configs/ultra_thermal_lstm.yaml。

输入:
    state_seq  (B, L, D)   — 12 关节 × 3 特征 (q, dq, T) 拼接的历史序列，D=36
输出:
    (B, 12, H)             — 全部 12 关节的未来多视距温度预测 (°C)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UltraThermalLSTM(nn.Module):
    """共享骨干 + 共享解码头的因果 LSTM（全关节联合输入/联合预测）。

    解码头为共享 MLP，一次性输出全部 12 关节 × H 视距的预测，
    使关节间的热耦合关系在解码阶段也被显式建模。
    """

    def __init__(
        self,
        input_dim: int = 36,
        proj_dim: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.15,
        mid_dim: int = 128,
        horizon: int = 9,
        n_joints: int = 12,
    ) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.horizon = horizon

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = n_joints * horizon  # 12 × 9 = 108
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D)  — D = n_joints * d_per_joint = 36

        Returns
        -------
        (B, n_joints, H) — 全部 12 关节的多视距未来温度预测
        """
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        h_last = lstm_out[:, -1, :]  # (B, hidden_dim)
        flat = self.decoder(h_last)  # (B, n_joints * H)
        return flat.view(-1, self.n_joints, self.horizon)  # (B, 12, 9)
