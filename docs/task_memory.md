# TienKung Ultra 腿部热建模 — 任务记忆（关键信息汇总）

> **关节顺序（唯一准则）**：以 **Ultra** 为准（`TienKung-Lab/legged_lab/envs/ultra/ultra_env.py` 中 `find_joints`）；**禁止**用 Deploy 腿向量下标代替 `T_leg[i]`。  
> **权威接口收紧**：**事实数据**以 **`Tienkung_thermal/data/bags/`** 中 rosbag2 为准；**消息定义与 CDR 解码**以 **`Tienkung/ros2ws`**（如 `install/bodyctrl_msgs/...`）为准。若与 **`Deploy_Tienkung`** 旧版源码冲突，**以 bag + ros2ws 为准**；Deploy 仅作可选行为参考。  
> **仿真与布局**：**TienKung-Lab** 仍用于 Ultra 关节顺序、任务名与 HDF5/时序方法论；**无**与实机一致的温度监督。  
> 详细条款见 `Tienkung_thermal/docs/plan.md` §0、§1。  
> **`temperature` 单位**：**摄氏度 (°C)**，类型 **`float32` 单标量**；IDL 以 **`Tienkung/ros2ws`** 中 `MotorStatus.msg` 为准（与 `plan.md` §0 一致）。

---

## 1. 目标与范围

| 项 | 内容 |
|:---|:-----|
| 机器人 | 天工 **Ultra**（与 Lab `ultra_env` 一致） |
| 自由度 | **12** 腿，顺序 **= Ultra**（`ultra_env`）；`leg_index_mapping.yaml` 仅与之对齐引用，**不以 Deploy 下标代替** |
| 验收 | 12 关节**等权**平均 MAE ≤ 1.5°C（@15s 等视距以 `plan.md` 为准） |
| 权重 | 配置 `w_i` + 可选可学习；Gate 只看**等权** MAE |
| 推理 | ≤ 5 ms FP16 |

---

## 2. 准许使用的数据（白名单）

### 2.0 权威来源（优先）

| 类别 | 权威路径 | 说明 |
|:-----|:---------|:-----|
| 实机观测真值 | `Tienkung_thermal/data/bags/`（rosbag2） | 训练/导出以 bag 内消息为准 |
| 消息字段与解码 | `Tienkung/ros2ws`（`bodyctrl_msgs` 等 `install/.../share/.../msg`） | `--msg-package`、类型与 `plan.md` 字段表均对齐此处 |
| Ultra 关节顺序 | `TienKung-Lab` `ultra_env.py` + `configs/leg_index_mapping.yaml` | `T_leg[0..11]` 唯一编号 |
| 非帧内系数 `ct_scale` | `configs/ct_scale_profiles.yaml` 或录包时 **`tg22_config.yaml` 快照** | 不在 bag 帧内；**不**强制与 Deploy Git 版本一致 |

### 2.1 Deploy_Tienkung（可选参考：历史插件行为；非消息 IDL 权威）

以下摘自 Deploy 文档/源码，**仅**在理解「机载曾如何实现」时参考；**消息形状与取值以 bag + `Tienkung/ros2ws` 为准**。

**Topic**

| Topic | 消息类型（头文件） |
|:------|:-------------------|
| `/leg/status` | `bodyctrl_msgs::msg::MotorStatusMsg` |
| `/arm/status` | 同上 |
| `/imu/status` | `bodyctrl_msgs::msg::Imu` |
| `/sbus_data` | `sensor_msgs::msg::Joy` |
| `/leg/cmd_ctrl`、`/arm/cmd_ctrl`、`/waist/cmd_pos` | 发布侧，见插件 |

**每条 `MotorStatusMsg.status[]` 元素 `one`（插件已读）**

- `pos`, `speed`, `current`, **`temperature`（`float32`，°C）**, `name`（用于 `getIndexById`）；与 **TienKung_ROS** `MotorStatus.msg` 一致

**`tg22_config.yaml`（插件加载）**

- `dt`, `ct_scale`, `motor_num`, …（热相关主要为 **dt、ct_scale**）

**`Imu`（插件已读成员）**

- `euler.{yaw,pitch,roll}`, `angular_velocity.{x,y,z}`, `linear_acceleration.{x,y,z}`（共 9 标量）

**映射表**

- `bodyIdMap.h`：腿 **0–11** 为 Deploy **中间向量**顺序（髋 R–P–Y，**≠** Ultra）；仅用于 **名称/CAN → j**，再 **按语义映射** 到 **Ultra** 的 `T_leg[0..11]`（唯一准则见 `plan.md` §1.2）。

### 2.2 TienKung-Lab

- `README.md`：版本、任务 `ultra_walk`/`ultra_run`、数据集路径、`train`/`play`/`sim2sim_ultra` 命令。  
- `ultra_env.py`：**腿关节名顺序**；`visualize_motion` **44 维**帧说明；`get_amp_obs_for_expert_trans` **AMP 布局**文档字符串。  
- `motion_loader*.py`：JSON `Frames` / `FrameDuration` / `MotionWeight`（**无温度**）。  
- 仿真：`robot.data.joint_pos` / `joint_vel` + `*_leg_ids`（**无温度字段**）。

---

## 3. 待获取备选数据（两仓库未定义或未接入）

| 项 | 说明 |
|:---|:-----|
| ~~`MotorStatus.temperature` 类型与单位~~ | **已确定**：见 **`TienKung_ROS`** `MotorStatus.msg`（`float32`，单通道）及 **`plan.md` §0**（**°C**） |
| `MotorStatusMsg` 与实机固件是否分叉 | 以 bag / `ros2 interface show` 与 **`Tienkung/ros2ws`** 中 `.msg` 交叉核对 |
| 电压、原生 `ddq`、故障/CRC | 插件未读；其它 Topic 未在两仓库列出 |
| BMS、主板温、风扇 | 插件未订阅 |
| 环境温度计 | 非仓库接口 |
| Lab 电机温度仿真 | 仓库内未出现 |
| 第三方机器人 DDS | 本专项基线排除 |

**允许的后处理**：对已白名单的 `speed` 等做**数值差分**、EMA，**不新增 Topic**。

---

## 4. Ultra 顺序为准（Deploy 仅为数据源）

**准则**：**Ultra**（`ultra_env.py` 中 `find_joints` 顺序）是 **`T_leg[0..11]` 的唯一编号准则**；`leg_index_mapping.yaml` 与之对齐。  

Deploy `bodyIdMap.h` 腿中间向量顺序：**l_hip_roll, l_hip_pitch, l_hip_yaw, …**（髋 **R–P–Y**），与 Ultra 每侧 **hip_roll → hip_yaw → hip_pitch → …**（髋 **R–Y–P**）在髋部三轴上**不一致**。  

→ **`T_leg[i]` 绝不等于** Deploy 中间向量第 `i` 个槽位；必须 **按关节语义名** 将 `one.pos/speed/current/temperature` 与 `ct_scale[j]` 对齐到 Ultra 第 `i` 路后再落盘或训练。

---

## 5. 关键路径

### 5.1 权威来源（plan.md §0）

| 路径 | 用途 |
|:-----|:-----|
| `Tienkung_thermal/data/bags/`（rosbag2） | **事实数据的最终依据**；与第三方仓库冲突时以 bag 为准 |
| `Tienkung/ros2ws/install/bodyctrl_msgs/share/bodyctrl_msgs/msg/` | **消息定义与 CDR 解码的唯一权威**（`MotorStatus.msg` 等） |
| `Tienkung_thermal/docs/plan.md` | 完整计划、白名单与权威接口收紧条款 |
| `Tienkung_thermal/configs/leg_index_mapping.yaml` | 与 **Ultra** 同序的 `motor_names`、CAN→T_leg 映射 |
| `TienKung-Lab/legged_lab/envs/ultra/ultra_env.py` | **Ultra 关节顺序权威（T_leg 唯一准则）** |
| `Tienkung_thermal/configs/ct_scale_profiles.yaml` | `ct_scale` 多版本管理（须为录包同期机载快照） |

### 5.2 历史行为参考（Deploy，非强制来源）

| 路径 | 用途 | 限定 |
|:-----|:-----|:-----|
| `Deploy_Tienkung/.../RLControlNewPlugin.cpp` | 了解插件曾如何读取 status/IMU | 不作为消息 IDL 权威 |
| `Deploy_Tienkung/.../bodyIdMap.h` | 了解历史腿索引与 CAN 排列 | 以 `ros2ws` `MotorName.msg` 为准 |
| `Deploy_Tienkung/.../tg22_config.yaml` | `dt`, `ct_scale` 历史参考 | 仅录包同期快照有效；Git 版本不保证与 bag 一致 |

---

## 6. 参考 `thermal_lstm_modeling.md` 搭建 Ultra LSTM 的调整要点

### 6.1 可复用的方法论

- 保留**因果 LSTM** 主线：输入投影 → LSTM 编码器 → 预测头。  
- 保留**滑动窗口**、**session 划分**、**Z-score**、**Huber/MAE**、**ONNX/TensorRT**、**ring buffer 在线推理**等工程流程。  
- 保留派生特征思路：`\tau^2`、`|dq|`、EMA、由 `speed` 数值差分得到的 `|ddq|`。  

### 6.2 必须替换的前提

| G1 规格书假设 | Ultra 基线应改为 |
|:--------------|:-----------------|
| 全身 **29** 关节 | 仅 **12** 条腿关节 |
| 按 `GearboxS/M/L` 分 **3** 个模型 | 先做 **Ultra 12 腿统一基线**，后续再视数据量决定是否分组 |
| 双温度通道：`temperature[0]` / `temperature[1]` | 基线仅承认 **`one.temperature` 单标量 `float32`，单位 °C** |
| 原始 `tau_est` / `ddq` / `vol` 可直接读取 | `tau_est = current * ct_scale`；`ddq` 仅能由 `speed` 数值差分；`vol` 暂不纳入 |
| DDS / SDK 低层接口 | **ROS2** `/leg/status` + 可选 `/imu/status` |
| 关节下标可直接用于训练 | 必须先映射到 **Ultra 顺序 `T_leg[0..11]`**（唯一准则） |

### 6.3 Ultra 基线输入 / 输出建议

**每关节基础输入（建议 v1）**

- `q = one.pos`
- `dq = one.speed`
- `tau_est = one.current * ct_scale[index]`
- `T = one.temperature`（**°C**）
- `tau_sq = tau_est^2`
- `dq_abs = |dq|`
- `ddq_abs = |diff(speed) / dt|`

**可选附加输入**

- 邻域温度：仅 `T_leg` 内相邻关节温度（如同侧 `i-1`, `i+1`）  
- IMU 9 标量：`euler(3) + angular_velocity(3) + linear_acceleration(3)`  

**输出**

- 单头预测未来多步 **标量温度** `y_temp`
- 验收仍以 **12 关节等权平均 MAE** 为准

### 6.4 不应直接沿用 G1 规格书的内容

- `temperature[1]` 外壳温度辅助头  
- `vol` 电压输入  
- `mainboard` / `bms` / `fan_state` / 环境温度默认入模  
- 全身 29 关节 HDF5 shape 与电机类型分桶假设  
- DDS CRC / `motorstate_` 位域这类仅在 G1 低层链路出现的处理

### 6.5 最小可行模型（建议）

- 序列长度：若以 **实机 rosbag 导出** 为准，见 **§9**：**`L = 2500`（5 s @ 500 Hz）**；下文 `L = 100`（5 s @ 20 Hz）仅对应旧版 `thermal_lstm_modeling.md` 叙述，**勿与 §9 混用**。
- Horizon：先对齐 `plan.md` 的 **15 s** 目标（@ 500 Hz 为 **7500 步**），不必保留 G1 的 20 s 远视距
- 架构：`InputProjection -> CausalLSTM -> Single PredictionHead`
- 损失：单通道 `Huber` 或 `MAE`
- 权重：训练可带 `w_0..w_11`，但 Gate 固定看等权 MAE

---

## 7. 需要调整的数据接口 / 数据契约

### 7.1 采集入口

**由 G1 风格接口切换为 Ultra 白名单接口：**

- 必需订阅：`/leg/status`
- 可选订阅：`/imu/status`
- 不把 `mainboard`、`bms`、环境传感器当作现成输入

### 7.2 关节索引与名称映射

所有训练、落盘、评估、在线推理统一使用 **Ultra 顺序 `T_leg[0..11]`**（与 `ultra_env` / `leg_index_mapping.yaml` 一致）。

最低要求：

- 采集阶段就完成 **Deploy 中间向量 → Ultra `T_leg`** 的固定语义映射（**不以位置下标硬拷贝**）
- 在元数据中写明映射表与顺序来源（**Ultra 为准**）
- 禁止下游代码假设 Deploy 下标 `j` = `T_leg` 下标 `i`

### 7.3 标签接口

G1 风格：

- `targets_coil`
- `targets_shell`

Ultra 基线：

- `targets_temp`

即：

- Dataset 返回 `(x, y_temp)` 或 `(x, y_temp, joint_idx)`
- 模型输出从双头改为单头
- Loss 从双通道联合损失改为单通道损失

### 7.4 原始特征接口

建议将原始白名单字段固定为：

- `q`
- `dq`
- `current`
- `temperature`
- `imu/*`（可选）

由此派生：

- `tau_est`
- `tau_sq`
- `dq_abs`
- `ddq_num`

### 7.5 中间数据 / HDF5 Schema

建议 Ultra 版中间格式以 **12 路腿关节** 为核心，不再复用 G1 的 `(N, 29)` 结构。

**当前基线（v1 导出）**：**仅纯腿**，不落盘 IMU 组；若后续做 IMU 消融，再在同一 schema 下增加 `imu/*` 数据集。

建议字段：

- `timestamps`: `(N,)`（**严格 2 ms 网格**，即 **500 Hz**，与 `plan.md` §4 一致）
- `joints/q`: `(N, 12)`
- `joints/dq`: `(N, 12)`
- `joints/current`: `(N, 12)` 或直接存 `tau_est`
- `joints/tau_est`: `(N, 12)`（`ct_scale` **按录制日期选用多版本**，见 **§10**）
- `joints/temperature`: `(N, 12)`
- `joints/voltage`: `(N, 12)`（`plan.md` §2.1.2.3）
- `joints/ddq_num`: `(N, 12)`（由 `speed` 数值差分）
- （可选扩展）`imu/euler`: `(N, 3)`、`imu/angular_velocity`: `(N, 3)`、`imu/linear_acceleration`: `(N, 3)` — **v1 不导出**
- `metadata/sample_rate_hz`（固定 **500**）
- `metadata/t_leg_order`
- `metadata/deploy_to_t_leg_mapping`
- `metadata/ct_scale_source`（路径或标签 + 版本/日期键）

### 7.6 Dataset 接口

Dataset 不再默认：

- 输入维度固定为 `D=8`
- 目标为双通道温度
- 按 `GearboxS/M/L` 分桶

建议改为参数化接口：

- `joint_indices`: 默认 `0..11`
- `feature_config`: 控制是否拼接邻居温度 / IMU
- `target_kind`: 当前仅 `temperature`
- `horizon_steps`: 由 Ultra 验收目标配置

### 7.7 在线推理接口

在线推理链路应改为：

- `/leg/status` 维护 `T_leg` 顺序 ring buffer
- 可选拼接 `/imu/status`
- 在线执行 EMA、数值差分、归一化
- 输出未来温度轨迹供热保护或监控模块消费

---

## 8. 实施计划（建议顺序）

### Phase A：先打通数据链路

1. ~~明确 `one.temperature` 的形态与单位~~ **已约定**：**单标量 `float32`，°C**（`TienKung_ROS` `MotorStatus.msg` + `plan.md` §0）；实机仍建议抽样对照。  
2. 实现并单测 `Deploy -> Ultra T_leg[0..11]` 映射。  
3. 落盘 Ultra 专用中间数据格式（至少含 `q/dq/current/tau_est/temperature`）。  

### Phase B：建立单温度基线模型

1. 用 `q, dq, tau_est, T, tau_sq, |dq|, |ddq|` 训练单头因果 LSTM。  
2. 以 **12 关节等权 MAE** 做主指标。  
3. 先不引入 `temperature[1]`、电压、BMS、主板温、风扇。  

### Phase C：逐步扩展特征

1. 加邻域温度，验证是否优于纯本体特征。  
2. 加 IMU 9 维，验证是否对负载切换和跑跳工况有收益。  
3. 若后续确认 `temperature` 实为多通道，再讨论双头网络。  

### Phase D：部署与闭环

1. 导出 ONNX / TensorRT。  
2. 接入在线 ring buffer 推理。  
3. 若业务需要，再接热保护阈值逻辑。  

### 当前优先级

- ~~**P0**：`one.temperature` 语义确认~~ **已确定：°C、单通道 `float32`**（见上）  
- **P0**：`Deploy → Ultra T_leg` 语义映射实现与单测（**Ultra 顺序为准**）  
- **P1**：Ultra 专用 HDF5 / Dataset 契约定稿  
- **P1**：单温度基线 LSTM 跑通  
- **P2**：邻域温度 / IMU 消融  

---

## 9. 全量 rosbag → HDF5 数据集（仅 `/leg/status`，纯腿 v1）

> **依据**：`plan.md` §1.2–1.3、§2.1、§4（**500 Hz** 工程网格）、§5；`configs/leg_index_mapping.yaml`（Ultra `T_leg[0..11]`）。  
> **范围**：原始 Topic **仅** `/leg/status`（`MotorStatusMsg`）；**不**导出 `/imu/status`**（v1 纯腿基线）；**不**将 `/leg/motor_status`（`MotorStatusMsg1`）纳入基线监督，与 `plan.md` 一致。

### 9.1 已确认导出策略（2026-04-13）

| 项 | 决定 |
|:---|:-----|
| **落盘格式** | **HDF5**（可按 session 多文件 + 全局 manifest，见 §9.5） |
| **特征范围** | **纯腿基线**：仅 `/leg/status` 派生字段；**不含** IMU |
| **时间网格** | **按时间戳重采样到严格 2 ms**（**500 Hz**），与 `plan.md` §4 一致 |
| **`ct_scale`** | **按录制日期多版本**选用对应 `tg22_config.yaml`（或等价快照）；**必须在 HDF5/metadata 中记录**所用版本与路径或哈希 |
| **异常帧** | 任一腿部电机 **`error ≠ 0`**：**整帧丢弃**（该时间样本不进入序列） |
| **`status` 长度 ≠ 12 等** | 见 **§9.6（R3）**：建议 **整帧丢弃**；连续大量异常则 **丢弃 session** 并记录原因 |

### 9.2 窗口长度与预测 horizon（与验收 15 s 对齐）

| 量 | 约定 |
|:---|:-----|
| **训练输入窗口** | **`L = 2500`**（**5 s** @ **500 Hz**），与 `plan.md` §4、`§2.1.5` 一致 |
| **验收目标** | 主温度预测未来 **15 s**（`plan.md` §0） |
| **15 s 对应步数** | @ 500 Hz：**7500 步**（\(15 \times 500\)） |
| **训练监督 horizon** | 实现上可配置 `H` 步；**与验收一致**时取 **`H = 7500`**（15 s）。若显存/稳定性需分阶段，可先训较短 `H` 再延长，但 **Gate / 报数**应以 **15 s @ 500 Hz** 为准 |

### 9.3 推荐实现顺序

1. **Bag 清单**：枚举 `data/bags/rosbag2_*`（`metadata.yaml` + `.db3`）；多 shard 按 **时间顺序拼接**（若尚未定稿，见 §9.6 开放项）。  
2. **解码**：`rosbags` + `bodyctrl_msgs`；流式读取 `/leg/status` 全量消息。  
3. **映射 + 清洗**：CAN `name` → Ultra `T_leg[0..11]`；**error≠0** 整帧丢；重排到 **500 Hz / 2 ms** 网格。  
4. **派生**：`tau_est`、`tau_sq`、`|dq|`、`ddq_num`、`voltage` 等仅来自白名单字段（`plan.md` §2.1.2）。  
5. **落盘**：HDF5 与 `§7.5` 对齐；附 **session manifest**（§9.5）。

### 9.4 Train / Val / Test 划分 — 接口保留与当前用法

**设计意图**：划分逻辑应支持（未来）**比例**（如 70/15/15）、**指定必须进 test 的 session**（如长跑、冷却段），且 **始终以 session 为原子**，避免同一录制泄漏到 train 与 test。

**当前阶段（v1）**：**仅按 session 划分**（例如脚本参数：train session 列表 / val / test，或按比例随机分 session **且固定随机种子**）。**比例、强制 test 工况** 先保留为 **manifest / CLI 占位字段**，待数据与工况标注完善后再启用。

建议在 manifest 中预留列（示例）：

- `session_id`、`hdf5_path`、`duration_s`、`n_frames_500hz`、`ct_scale_key`
- `split`：`train` \| `val` \| `test`（由划分脚本写入）
- （可选，待填）`scenario_tags`：如 `long_run`、`cooldown`，用于未来 **强制进 test**

### 9.5 HDF5 与 manifest 要点

- 每个 session 一个 HDF5 或按约定分块；**全局 `manifest.csv`（或 JSON）** 索引全部 session。  
- **元数据必含**：`sample_rate_hz=500`、`t_leg_order`、`ct_scale` 版本标识、`source_rosbag` 路径、`export_timestamp`。

### 9.6 仍待实现时拍板的细节（非阻塞当前决策）

| # | 项 | 说明 |
|:--|:---|:-----|
| **R1** | 多 `.db3` shard | 默认建议 **按 bag 内时间顺序合并**；若某 bag 异常则整包剔除并记入 manifest |
| **R2** | 输出根目录 | 仓库内 `data/processed/` 或数据盘路径；大文件不入 Git |
| **R3** | `status` 长度异常 | 建议 **整帧丢弃**；连续大量异常则 **丢弃 session** 并记录原因 |

---

## 10. 工作记录（按日期）

### 2026-04-13 — rosbag → 数据集导出决策

- 输出：**HDF5**；**纯腿**、**不重采样 IMU**；时间轴 **严格 2 ms（500 Hz）**。  
- **`ct_scale`**：**按录制日期多版本**，落盘注明来源。  
- **清洗**：**error≠0 → 整帧丢弃**。  
- **划分**：manifest **保留**比例与强制 test 的接口；**当前仅 session 级划分**。  
- **窗口**：**L=2500**（5 s @ 500 Hz）；验收对齐 **15 s → 7500 步** @ 500 Hz；训练 `H` 可配置，**报数与 Gate 以 15 s 为准**。  
- 详见本章 **§9**。

### 2026-04-13 — 全量导出实现与跑批（`data/bags` → `data/processed/leg_status_500hz`）

- **入口**：`scripts/bags/export_leg_status_dataset.py`；库内逻辑：`tienkung_thermal/bags/pipeline.py`（CAN→`T_leg`、`ct_scale` 置换、500 Hz 网格、`h5py` 落盘）。  
- **依赖**：`pip install -e ".[rosbag]"`；`--msg-package` 指向 **`bodyctrl_msgs` 包根**（含 `package.xml`），本机示例：`.../Tienkung/ros2ws/install/bodyctrl_msgs/share/bodyctrl_msgs`。  
- **命令示例**（处理整个 `data/bags`、输出到默认目录、重写 manifest 列）：  
  `python scripts/bags/export_leg_status_dataset.py data/bags --out-dir data/processed/leg_status_500hz --overwrite-manifest --msg-package <bodyctrl_msgs根>`  
- **仅重建 manifest**（已有 `.h5`、不重新解码 bag）：加 `--skip-existing`，会按各 HDF5 内 `metadata` 属性与 `timestamps` 长度写表（勿与「删除全部 h5 后重跑」混淆）。  
- **manifest 列**：含 `ct_scale_profile`、`n_grid_frames_500hz` 等；`split` 留空供后续按 session 填写。  
- **跳过原因**：`metadata.yaml` **缺失或为空**、无 `rosbag2_bagfile_information`、无 `.db3` 时 **不调用 rosbags**（否则会 `NoneType`）；需补全元数据或重新录制。已成功导出的 session 其 HDF5 内 `joints/*` 形状为 `(N, 12)`，`timestamps` 为秒、步长 0.002 s。  
- **`ct_scale` 多版本**：编辑 `configs/ct_scale_profiles.yaml` 的 `profiles` + `profile_rules`（按 `rosbag2_*` 目录名 **prefix** 匹配）；当前占位为 **default 全 1.0**，实机系数需从 **`tg22_config.yaml` 腿段前 12 项** 填入对应 profile。

---

*与 `plan.md` 同步；待获取项澄清后可迁入 `plan.md` §1.3–1.4 白名单。*
