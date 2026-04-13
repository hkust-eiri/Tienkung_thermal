# 腿部 `/leg/status` 500 Hz HDF5 数据集说明

本文描述由 `scripts/bags/export_leg_status_dataset.py` 与 `tienkung_thermal/bags/pipeline.py` 从 **ROS 2 rosbag2** 导出的 **`leg_status_500hz`** 数据集格式、处理规则与下游用途。与工程约定一致时参见 **`docs/plan.md`**、关节顺序参见 **`configs/leg_index_mapping.yaml`**（Ultra `T_leg[0..11]`）。

---

## 1. 定位与数据来源

| 项目 | 说明 |
|------|------|
| **用途** | 将实机录制的 **`/leg/status`**（`bodyctrl_msgs/msg/MotorStatusMsg`）转为 **固定 500 Hz（步长 2 ms）** 的序列，供 Ultra **12 腿关节**热建模 / LSTM 等离线训练。 |
| **原始话题** | 仅 **`/leg/status`**。 |
| **不包含** | **无** `/imu/status`；**无** 上肢/腰等；**无** `MotorStatusMsg1`（如 `/leg/motor_status`）。即 **纯腿、以 `MotorStatus` 单通道温度** 为主的基线包。 |
| **关节顺序** | 第二维下标 **0–11** 为 **`T_leg[0..11]`（Ultra）**，与 `leg_index_mapping.yaml` 中 `motor_names` 一致（髋部 **R–Y–P**）。Deploy 腿向量顺序不同，已通过 **CAN ID → Ultra** 映射落盘。 |

---

## 2. 磁盘布局

| 路径 | 含义 |
|------|------|
| `data/processed/leg_status_500hz/<session_id>.h5` | **单次录制 session** 一个 HDF5；`session_id` 通常与 rosbag 目录名一致（如 `rosbag2_2026_04_07-12_14_44`）。 |
| `data/processed/leg_status_500hz/manifest.csv` | **索引表**：每个成功导出的 session 一行，指向 `.h5`，并记录消息数、500 Hz 帧数、`ct_scale` profile 等；**`split` 列预留** train/val/test。 |

导出命令示例见 **`docs/task_memory.md`** §10 或脚本 `scripts/bags/export_leg_status_dataset.py` 文件头注释。

---

## 3. 单个 HDF5 文件结构

### 3.1 根级属性（`/`）

| 属性名 | 含义 |
|--------|------|
| `export_timestamp_utc` | 本次导出 UTC 时间（ISO 8601）。 |
| `source_rosbag` | 源 rosbag2 **目录**绝对路径。 |
| `sample_rate_hz` | 固定 **500**。 |
| `dt_grid_s` | 固定 **0.002**（秒）。 |
| `t_leg_order` | 12 个 Ultra 关节名字符串（逗号分隔），与 `pipeline.MOTOR_NAMES_ULTRA` 一致。 |
| `ct_scale_profile` | 使用的 `ct_scale` profile 名（如 `default`）。 |
| `ct_scale_config_path` | `configs/ct_scale_profiles.yaml` 的路径。 |
| `ct_scale_per_t_leg_json` | 本文件使用的 **12 维 `ct_scale`（已是 Ultra 顺序）** 快照，用于复现 `tau_est`。 |
| `ct_scale_profile_description` | （可选）profile 描述。 |

### 3.2 数据集 `timestamps`

| 项目 | 说明 |
|------|------|
| **形状** | `(N,)`，`N` 为本 session 在首末有效帧时间 **`t0`…`t1`** 上按 **0.002 s** 划分的采样点数（与 `metadata/n_grid_frames` 一致）。 |
| **单位** | **秒（s）**；值为 **绝对时间**（由 `header.stamp` 经插值落在均匀网格上）。 |
| **对齐** | 第 `k` 行与 `joints/*` 第 `k` 行同一时刻。 |

### 3.3 组 `joints/`

所有数据集形状均为 **`(N, 12)`**，第二维为 **`T_leg[0..11]`**。存储通常带 gzip + shuffle。

| 名称 | 含义 | 单位 / 备注 |
|------|------|----------------|
| `q` | 关节位置 | rad（`MotorStatus.pos`） |
| `dq` | 关节角速度 | rad/s（`speed`） |
| `current` | 电机电流 | A |
| `temperature` | 电机温度（主要监督量） | **°C** |
| `voltage` | 电机端电压 | V |
| `tau_est` | 估计力矩 | `current × ct_scale`（`ct_scale` 为 Ultra 顺序，由 `ct_scale_profiles.yaml` 中录包同期系数按 CAN 映射置换得到） |
| `tau_sq` | \((\tau_{est})^2\) | 焦耳热类代理特征 |
| `dq_abs` | \(\|dq\|\) | 与转速相关的损耗代理 |
| `ddq_abs` | 对 **`dq` 在 500 Hz 网格上**的差分：\(\|\Delta dq / dt\|\)；首行与下一行对齐方式与实现一致（首行复制第一阶差分） |

**重采样方式**：在原始有效帧时间序列上，对各通道使用 **`numpy.interp` 线性插值** 到均匀 2 ms 网格；再在网格上计算 `tau_sq`、`dq_abs`、`ddq_abs`（见 `resample_arrays_to_grid`）。

### 3.4 组 `metadata/`

均为 **HDF5 属性**（非大数组），用于质检与追溯：

| 属性名 | 含义 |
|--------|------|
| `n_raw_messages_leg_status` | 读取的 `/leg/status` 消息总数。 |
| `n_valid_raw_frames` | 解析通过且 **12 电机齐全、CAN 合法、12 路 `error==0`** 的原始帧数。 |
| `n_skipped_bad_status_len` | `status` 长度异常等跳过次数。 |
| `n_skipped_unknown_can` | 未知 CAN 等跳过次数。 |
| `n_skipped_error_nonzero` | **任关节 `error≠0`** 丢弃的原始消息条数。 |
| `n_skipped_incomplete_12` | 重复或缺失导致无法凑齐 12 路的次数。 |
| `n_grid_frames` | 等于 **`len(timestamps)`**，即 500 Hz 序列长度 `N`。 |

---

## 4. 原始帧 → HDF5：处理规则

1. **解码**：仅处理 **`/leg/status`** 的 `MotorStatusMsg`。
2. **映射**：用每条 `MotorStatus` 的 **`name`（CAN ID）** 映射到 Ultra 下标 **`T_leg[i]`**（51–56 左腿、61–66 右腿；髋 pitch/yaw 与 Deploy 顺序交叉已在映射表体现，见 `tienkung_thermal/bags/mapping.py`）。
3. **丢弃整帧**（不进入有效序列）若满足任一条件：  
   - `len(status) != 12`；  
   - CAN 未知或同一 Ultra 槽位重复；  
   - **任一关节 `error != 0`**。  
4. **时间戳**：优先 **`header.stamp`**；否则使用 bag 记录时间（纳秒）换算为秒。  
5. **去重**：按时间排序，**相同时间戳保留最后一条**。  
6. **500 Hz**：在 `[t0, t1]` 上生成步长 **0.002 s** 的网格，对 `q`、`dq`、`current`、`temperature`、`voltage`、`tau_est` 线性插值；再计算派生列。

---

## 5. `manifest.csv` 列说明

每行对应 **一个已成功生成 HDF5 的 session**，便于批量划分与统计。

| 列名 | 含义 |
|------|------|
| `export_timestamp_utc` | 导出批次时间。 |
| `session_id` | 会话标识（通常与 bag 目录名一致）。 |
| `source_bag` | 源 rosbag 目录绝对路径。 |
| `hdf5_path` | 生成的 `.h5` 绝对路径。 |
| `ct_scale_config` | 使用的 `ct_scale_profiles.yaml` 路径。 |
| `ct_scale_profile` | profile 名。 |
| `n_raw_messages` | 原始 `/leg/status` 消息数。 |
| `n_valid_raw_frames` | 有效原始帧数。 |
| `n_grid_frames_500hz` | HDF5 中 **500 Hz 序列长度 `N`**。 |
| `n_skipped_error_nonzero` | 因 **error≠0** 跳过的原始消息条数。 |
| `split` | 预留：可填 `train` / `val` / `test`（当前常为空，需按 session 划分时填写）。 |

---

## 6. 下游使用建议

- **监督学习**：以 **`joints/temperature`** 为温度真值（或预测未来若干步的温度）；输入特征按 **`docs/plan.md` §2.1** 从本文件白名单列组合。  
- **划分数据集**：**按 session 整体** 划分 train/val/test，避免同一段录制泄漏；可用 `manifest.csv` 的 `split` 或外部分组表驱动 `Dataset`。  
- **窗口与 horizon**：在 **500 Hz** 上取输入长度 **`L`**（例如 **2500 ≈ 5 s**）；若对齐 **15 s** 预测视距，在同样网格上约为 **7500 步**（与 `plan.md` 验收口径一致时需统一频率与步数）。  

---

## 7. 注意事项

- **`ct_scale`**：若 `configs/ct_scale_profiles.yaml` 中仍为占位（如全 **1.0**），则 **`tau_est` / `tau_sq` 仅保证公式与顺序正确**，数值需用**与 bag 同期**的机载 **`tg22_config.yaml` 腿段前 12 项**填入对应 profile 后**重新导出**（`ct_scale` 不随帧写入 bag，须由录包同期快照提供，见 `plan.md` §0）。  
- **重采样为线性插值**，不是零阶保持；若实验要求严格「子采样/保持」语义，需在实验配置中单独说明或扩展导出选项。  
- **大文件**：`data/` 通常由 `.gitignore` 忽略；HDF5 体积随 session 时长与 500 Hz 长度增长，建议放在数据盘或共享存储。

---

*与 `docs/plan.md`、`docs/task_memory.md` §7.5、§9、§10 一致；实现以 `tienkung_thermal/bags/pipeline.py` 为准。*
