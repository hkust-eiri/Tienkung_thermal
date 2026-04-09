# 天工 Ultra 腿部热建模 — 实机 ROS 2 Bag 录制操作文档

> **关联文档**：`docs/plan.md`（接口白名单、§5 采集协议）、`docs/thermal_lstm_modeling.md`（时序长度、session 划分、HDF5 约定）、`docs/todo.md`（P0 录制与核对项）、`configs/leg_index_mapping.yaml`（`T_leg[0..11]` 顺序）。  
> **适用范围**：Deploy 栈已运行、`/leg/status` 可订阅；监督温度来自 `bodyctrl_msgs/msg/MotorStatus.temperature`（`float32`，**°C**）。

---

## 1. 文档目的

本文件给出从**录制前检查**到**落盘命名、工况设计、与训练划分衔接**的**可执行实操流程**，并汇总与录制相关的命令与约束，避免仅依赖口头约定。

---

## 2. 核心原则（录制前必读）

| 原则 | 说明 |
|:-----|:-----|
| **一次录制 = 一个 session** | 训练集划分按 **整段 bag（session）** 分配 Train / Val / Test，**禁止**同一 session 内再随机拆到不同集合（见 `thermal_lstm_modeling.md` §6.4）。 |
| **关节顺序以 Ultra 为准** | 原始 bag 存 Deploy 侧消息；离线转换时必须按**关节语义名**映射到 `T_leg[0..11]`，**禁止**把 Deploy 腿向量下标与 Ultra 下标按位置等同（见 `plan.md` §1.2、`leg_index_mapping.yaml` 文首说明）。 |
| **白名单 Topic** | 基线至少 **`/leg/status`**；若使用 IMU 上下文特征，同时录 **`/imu/status`**（见 `plan.md` §1.3、§2.1.3）。 |
| **环境温度** | 实验室室温等仅作**实验笔记**，不写入 ROS 白名单字段（`plan.md` §1.5）。 |

---

## 3. 录制前检查清单

按顺序完成；未完成前不建议开始正式采集。

1. **环境与工作空间**  
   - 与机器人控制端一致的 ROS 2 发行版（如 Humble）。  
   - 已 `source` 包含 `bodyctrl_msgs` 的工作空间（通常与 **Deploy_TienKung** / **TienKung_ROS** 一致）。

2. **Topic 可用性**  
   - `ros2 topic list` 中存在 `/leg/status`（及计划使用的 `/imu/status`）。  
   - `ros2 topic echo /leg/status --once` 能收到 `MotorStatusMsg`，且 `status` 数组中每条含 `name`、`pos`、`speed`、`current`、`temperature`。

3. **接口一致性（P0，可与首条短 bag 并行）**  
   - 对照 `ros2 interface show bodyctrl_msgs/msg/MotorStatus` 与 **TienKung_ROS** 中 `MotorStatus.msg`。  
   - 更细项见 `docs/todo.md` P0；可用 `scripts/check/p0_check.py` 在实机核对频率与时间戳。

4. **磁盘与路径**  
   - 确认 `ros2 bag record` 的 `-o` 输出目录所在分区空间充足（长时录制增长快）。

5. **实验记录准备**  
   - 准备简单表格或笔记：session 名、日期、工况类型、室温（可选）、操作员、备注。

---

## 4. 详细实操流程（分步）

### 4.1 阶段 A：链路冒烟（建议最先做）

**目的**：验证能录、能回放、消息字段齐全；不追求工况覆盖。

| 步骤 | 操作 |
|:----:|:-----|
| A1 | 机器人上电，Deploy / 底层驱动按现场规程运行，确保 `/leg/status` 在发布。 |
| A2 | 在新终端 `source` 工作空间。 |
| A3 | 录制 **仅 `/leg/status`**，时长 **≥ 30 s**（与 `todo.md` 建议一致）：见 **§5.1** 命令，输出目录建议含 `p0_leg_30s` 等字样。 |
| A4 | （可选）再录一条 **≥ 30 s** 的 **`/leg/status` + `/imu/status`**：见 **§5.2**，用于时间对齐分析。 |
| A5 | `ros2 bag info <bag目录>` 确认消息类型、时长、无零消息异常。 |
| A6 | 将结论记入实验笔记；P0 勾选见 `todo.md`。 |

### 4.2 阶段 B：正式工况采集（按 session 规划）

**目的**：为 LSTM 提供 **多种负载、升温、冷却、动态切换** 的统计覆盖，并满足 **按 session 划分** 时 Val 对「高负载 + 冷却」的要求。

| 步骤 | 操作 |
|:----:|:-----|
| B1 | 根据 **§6 工况矩阵** 选定本次 session 的**单一主工况**（或主次分明的组合），在笔记中写好 **session 命名**（建议 **§7** 规范）。 |
| B2 | 确认机器人处于该工况所需安全状态（护栏、急停、人员距离等按现场规程）。 |
| B3 | `source` 工作空间，在**工况稳定起始时刻**启动 `ros2 bag record`（建议统一用 **§5.2** 含 IMU，便于后续消融与对齐）。 |
| B4 | 按工况设计执行动作：**尽量包含** 升温段 →（如适用）高负载稳态 → **冷却段**（负载降低或静置）。单次建议 **数分钟**，见 **§8** 时长说明。 |
| B5 | 用 **Ctrl+C** 正常结束 `ros2 bag record`，检查输出目录完整。 |
| B6 | 立即在笔记中补全：结束时间、异常中断说明、室温（若记录）。 |
| B7 | 对下一工况 **重复 B1–B6**，**每次新起一个输出目录**（新 session），勿在同一目录续录混用。 |

### 4.3 阶段 C：离线处理与训练衔接（录制结束后）

**目的**：与 `thermal_lstm_modeling.md` §6 流水线一致；此处仅列操作要点，实现以代码为准。

| 步骤 | 操作 |
|:----:|:-----|
| C1 | 对每个 bag：**解析** `MotorStatusMsg`，按 **语义名** 映射到 `T_leg[0..11]`，生成中间表或 HDF5（字段建议见 `thermal_lstm_modeling.md` §6.2）。 |
| C2 | 将序列 **重采样到 20 Hz**（与 `plan.md` §4、`thermal_lstm_modeling.md` §1.2 一致）。 |
| C3 | 计算派生特征（`tau_est`、`tau_sq`、`|dq|`、数值 `|ddq|` 等），仅使用白名单字段。 |
| C4 | **按 session** 划分 Train（约 70%）/ Val（约 15%）/ Test（约 15%）：**整段 session 不得拆开**；Val 集合中应包含 **含完整高负载与冷却段** 的 session（`thermal_lstm_modeling.md` §6.4）。 |
| C5 | 用滑动窗口生成样本：`L = 100`（约 5 s），多视距标签至 **15 s**（`thermal_lstm_modeling.md` §1.2）。 |

---

## 5. `ros2 bag record` 命令

以下假设已 `source` 工作空间，且 topic 名称与 Deploy 一致（`plan.md` §1.3）。

### 5.1 仅腿部状态（最小集）

含每关节 `temperature`（°C）及 `pos` / `speed` / `current` / `name`：

```bash
ros2 bag record -o ~/bags/thermal_leg_$(date +%Y%m%d_%H%M%S) /leg/status
```

### 5.2 腿部 + IMU（推荐用于正式数据集）

与 `plan.md` §2.1.3 可选 IMU 上下文一致，便于负载切换、跑跳等工况：

```bash
ros2 bag record -o ~/bags/thermal_leg_imu_$(date +%Y%m%d_%H%M%S) /leg/status /imu/status
```

**说明**：`-o` 后为输出目录；可按项目规范改为统一数据盘路径。录制中 **Ctrl+C** 结束写入。

---

## 6. Session 设计与工况矩阵（建议）

下列不是唯一标准，但与 **`thermal_lstm_modeling.md`** 中「多种负载、冷却动态、工况切换 / `|ddq|`、IMU 辅助」及 **`plan.md` §3**「膝、髋 pitch 等热工况优先」的表述一致，便于落地。

### 6.1 建议的 session 类型（每类可多条、多天重复）

| 类型代号 | 内容要点 | 训练价值 |
|:--------:|:---------|:---------|
| **S0** | 中低负载、长时间稳态行走或站立微调 | 常见工况覆盖 |
| **S1** | 中高负载、持续数分钟；**刻意让膝、髋 pitch 等大扭矩关节处于较高负载** | 升温与高热标签 |
| **S2** | **S1 类高负载结束后** 不接新动作，**静置冷却**足够长时间 | 冷却段动态；Val 建议包含此类完整过程 |
| **S3** | 步态/速度/地形切换多，或跑跳类（安全前提下） | `|ddq|`、切换；若录 IMU，利于上下文特征 |

### 6.2 与 Train / Val / Test 的对应关系（规划时自检）

- **Train**：多种 **S0–S3** session，覆盖不同负载水平。  
- **Val**：至少部分 session 为 **完整「高负载 → 冷却」**（例如一次录制内包含 **S1+S2** 连续过程，或明确一次长录包含两段）。  
- **Test**：**独立 session**，在训练与调参过程中**不可见**，仅最终评估使用。

### 6.3 命名与 Lab 对齐（可选）

`plan.md` 建议工况命名可与 **TienKung-Lab** 的 `ultra_walk` / `ultra_run` 等对齐；可在 `metadata/notes` 或目录名中标注 **任务名 + 速度/地形**，便于追溯。

---

## 7. 输出目录与 session 命名规范（建议）

统一命名可减少后期整理成本，示例：

```text
thermal_<工况简码>_<YYYYMMDD>_<HHMMSS>_<短备注>
```

示例：`thermal_S1_highload_20260408_143022_lab`  

**要求**：

- **每次独立录制使用新目录**，对应一个 **session_id**（与 HDF5 `session_name` 或划分表一致）。  
- 笔记中记录：目录名、类型（S0–S3）、是否含 IMU topic、异常与中断位置。

---

## 8. 单次录制时长与时间尺度

| 项目 | 约定来源 | 实操建议 |
|:-----|:---------|:---------|
| 训练网格 | `20 Hz`（`thermal_lstm_modeling.md` §1.2） | 离线重采样到此频率 |
| 输入窗口 | `L = 100`（约 **5 s**） | — |
| 最远监督 | **15 s** 视距 | 切片时需足够未来帧 |
| 最小有效长度（重采样后） | 约需 **`n_frames > 100 + 300 = 400`** 点（约 **20 s**）才便于生成最长视距样本（`thermal_lstm_modeling.md` §6.3 思路） | **单次 session 建议数分钟**，避免仅贴下限 |
| P0 短 bag | `todo.md` | **≥ 30 s** 用于频率与 stamp 检查 |

---

## 9. 安全与质量注意事项

1. **安全**：所有动作符合现场机器人安全规程；高负载与跑跳类 session 需额外评审。  
2. **冷却**：避免连续高强度录制导致过热保护触发；**S2 冷却段**对模型与 `MAE_cooling` 类评估有意义（`thermal_lstm_modeling.md` §7.3）。  
3. **一致性**：同一次实验批次内，尽量固定 **鞋子/负重/地面** 等未在 ROS 中出现的因素，并在笔记中记录。  
4. **中断**：若 bag 中途异常停止，在笔记中标注；该 session 是否纳入 Val/Test 由负责人根据完整性决定。  
5. **回放**：定期 `ros2 bag play` 抽查，确认与录制时 topic 一致。

---

## 10. 命令与文档速查

| 用途 | 命令或位置 |
|:-----|:-----------|
| 看 topic | `ros2 topic list` |
| 看消息定义 | `ros2 interface show bodyctrl_msgs/msg/MotorStatus` |
| bag 信息 | `ros2 bag info <目录>` |
| P0 检查脚本 | `scripts/check/p0_check.py`（见 `docs/todo.md`） |
| 映射配置 | `configs/leg_index_mapping.yaml` |
| 实施总纲 | `docs/plan.md` |
| 模型与 HDF5 | `docs/thermal_lstm_modeling.md` §6 |

---

*文档结束 — 若与实机 topic 名或消息定义不一致，以实机 `ros2 interface show` 与 bag 为准，并回写 `plan.md` §1.3。*
