# Ultra 热 LSTM — 待实机或另行确认项（详细版）

## 文档目的

本文件列出：**在把采集、训练、在线推理写进代码之前**，必须通过实机、`ros2 interface show`、Deploy 源码或日志**核实**的参数与接口。

- **原则**：与 `plan.md` §0–§1.5、`task_memory.md` §3 一致——两仓库未写明的字段不得默认存在；`bodyctrl_msgs` 的 `.msg` 不在仓库内时，必须以实机接口定义为准。
- **用法**：自上而下按 **P0 → P1 → P2** 勾选；P0 未完成前，不建议大规模录数或锁死 HDF5 schema。
- **记录建议**：每勾一项，在团队 wiki 或 PR 里补一行「结论 + 证据」（接口输出截图、bag 统计、`interface show` 粘贴）。

---

## 优先级说明

| 级别 | 若不确认会怎样 | 典型动作 |
|:-----|:---------------|:---------|
| **P0** | 关节错位、标签错关节、力矩代理错、时间基准错 → 数据**整体不可信** | 先做完再批量录数 |
| **P1** | 特征物理意义漂移、IMU 与腿不同步、在线与离线分布不一致 | 特征定型前完成 |
| **P2** | 主要影响**效果与工程取舍**，可迭代 | 可与基线训练并行 |

---

## P0 — 阻塞数据采集与标签正确性

### P0 说明

P0 解决三件事：**（1）消息里到底有什么字段、什么类型**；**（2）第 `i` 个训练槽位是否真是 `T_leg[i]` 那条腿**；**（3）`current × ct_scale` 是否真是可用的力矩代理、时间步长怎么取**。任何一条错了，LSTM 学的是噪声或错标签。

### P0 两仓库静态可答项（`Deploy_Tienkung` / `TienKung-Lab`）

以下仅通过**阅读两仓库源码与配置**可归纳的结论，用于勾 P0 时对照；**不能**替代实机 `ros2 interface show`、bag 统计与物理对照。路径均相对各仓库根目录。

**`Deploy_Tienkung`**

| P0 TODO 主题 | 静态可答内容 | 证据位置 |
|:-------------|:-------------|:---------|
| 话题与消息类型 | 订阅 `/leg/status`，类型为 `bodyctrl_msgs::msg::MotorStatusMsg` | `rl_control_new/src/plugins/rl_control_new/src/RLControlNewPlugin.cpp`（`onInit` 中 `create_subscription`） |
| 每条 `status` 读哪些字段 | 对 `msg->status` 遍历；使用 `one.pos`、`one.speed`、`one.current`、`one.temperature`；`one.name` 传入 `idMap.getIndexById(one.name)` 得到 `index` | 同上，`LegMotorStatusMsg` 回调循环 |
| `name` / 中间向量下标 | `getIndexById` 与 CAN id 表对应；腿部 `index` 0–11 与 `l_hip_roll`…`r_ankle_roll` 固定顺序见映射表初始化 | `rl_control_new/src/plugins/rl_control_new/include/bodyIdMap.h`（`legIds` / `legNames`） |
| \(\tau_{est}\) 与 `ct_scale` 对齐 | `tau_fed_midVec(index) = one.current * ct_scale_midVec(index)`；腿部标定：`ct_scale_midVec.head(12) << ct_scale.head(12)`，与腿 `index` 同一下标 | `RLControlNewPlugin.cpp`（`onInit` 与状态回调） |
| `temperature` 在插件侧形态 | 写入 `temperature_midVec(index) = one.temperature`，与 `pos/speed/tau` 同一 `index`（**C++ 侧按标量赋值**；**IDL 是否数组仍以 `bodyctrl_msgs` 为准**） | 同上 |
| `dt` 含义（配置层） | `tg22_config.yaml` 中 `dt`；`LoadConfig` 读入 `dt`、`ct_scale`；控制循环里用 `dt` 做周期（与 `/leg/status` 发布频率是否一致**无法**在仓库内证明） | `rl_control_new/config/tg22_config.yaml`、`RLControlNewPlugin.cpp`（`LoadConfig`、`rlControl` 内定时） |
| `ct_scale` 与 `motor_num` | 配置中声明 `motor_num`；`ct_scale` 为 YAML 数组；**当前仓库中 `ct_scale` 元素个数与 `motor_num` 可能不一致**（需与机载一致配置核对） | `tg22_config.yaml` |

**`TienKung-Lab`**

| P0 TODO 主题 | 静态可答内容 | 证据位置 |
|:-------------|:-------------|:---------|
| 训练用 `T_leg` 槽位顺序（**唯一准则**） | **以 Ultra 为准**：左腿 / 右腿各 6 关节 `hip_roll` → `hip_yaw` → `hip_pitch` → `knee_pitch` → `ankle_pitch` → `ankle_roll`。Deploy 腿中间向量髋序为 R–P–Y，**不得**与 `T_leg[i]` 按下标混用；须按关节语义映射到 Ultra 第 `i` 槽位 | `legged_lab/envs/ultra/ultra_env.py`（`left_leg_ids` / `right_leg_ids` 的 `find_joints` `name_keys` 顺序）；`configs/leg_index_mapping.yaml` 与之对齐 |
| 与 USD/MJCF 关节名 | 上述 `*_joint` 名称与 Ultra 资产一致（用于和 `leg_index_mapping.yaml` 对齐） | 同上 |

**两仓库仍无法静态回答（须实机或 `bodyctrl_msgs` 包内 `.msg`）**

- `MotorStatusMsg` **完整** IDL：`status` 定长或变长、`temperature`/`current` 的 ROS 类型、是否另有字段。
- **物理单位与符号**：电流安培/毫安、温度 °C、与力矩符号约定。
- **`/leg/status` 实际频率**、stamp 抖动、与 `dt` 是否一致：须 bag 或 `scripts/check/p0_check.py` 在实机运行。

### P0 核对脚本（`scripts/check/p0_check.py`）

**定位**：仅在**实机**（已 `source` ROS2、话题有数据）运行，用于确认 `ros2 interface show` 与 `/leg/status` 上**离线无法确定**的接口类型与样本形态。**不**读取 `Deploy_Tienkung` / `TienKung-Lab` 源码做静态审计；仓库与配置的一致性请在开发机通过文档、代码审阅与 `leg_index_mapping.yaml` 自行完成。

脚本内嵌与 `configs/leg_index_mapping.yaml` 一致的 `T_leg` 关节名列表，仅用于核对**实机一帧** `status` 中的 `name` 能否映射满 12 路腿电机。

**用法（在机器人或已接入实机 ROS 域的机器上）**：

```bash
cd Tienkung_thermal

python scripts/check/p0_check.py

# 与 Deploy tg22_config.yaml 中的 dt 对比消息 stamp 间隔（可选）
python scripts/check/p0_check.py --dt 0.0025

python scripts/check/p0_check.py --json
```

常用参数：`--topic`（默认 `/leg/status`）、`--dt-tolerance-ratio`（与 `--dt` 对比时的相对容差，默认 `0.2`）、`--echo-multi-timeout`（连续 echo 采集秒数，默认 `8`）。

**预期终端输出（非 `--json`）**：

1. 标题：`TienKung Thermal — P0 实机确认`，分隔线，一行统计：`统计: FAIL=… PASS=… PENDING=…`。
2. 每条检查：`[PASS|FAIL|PENDING] p0.xxx`，一行摘要，可选 `  - ` 明细（接口字段解析、echo 样本统计、stamp 间隔等）。
3. **退出码**：存在任一 `FAIL` 时为 **1**，否则为 **0**（`PENDING` 不导致失败）。

**状态含义**：

| 状态 | 含义 |
|:-----|:-----|
| `PASS` | 实机侧该项检查通过 |
| `FAIL` | 无 `ros2`、interface 失败、话题无数据、字段与基线假设不一致、或（若提供 `--dt`）间隔与 `dt` 偏差过大等 |
| `PENDING` | 脚本不判定物理量（电流安培/符号、温度 °C 等），或未能采到足够多帧计算间隔 |

**与下方 P0 TODO**：脚本对应「实机 `interface show`、单帧 `/leg/status` 形态、消息时间间隔」；映射表与插件代码级结论不在此脚本中重复验证。

### P0 TODO List

- [ ] **拿到 `MotorStatusMsg` 的权威定义并归档**
  - **为何重要**：插件按成员名读 `pos/speed/current/temperature/name`；若类型是数组、整型、或单位与假设不符，预处理会静默错。
  - **待确定**：
    - [ ] `status` 是固定长度数组还是变长序列
    - [ ] 每条 `status` 元素是否带独立时间戳或仅依赖消息 `header`
    - [ ] 除已文档化的字段外是否还有电流环/错误码等字段可后续纳入白名单
  - **建议**：在实机环境执行 `ros2 interface show bodyctrl_msgs/msg/MotorStatusMsg`（或包内实际类型名），把完整定义粘贴到仓库 issue / 设计笔记。

- [ ] **确认 `one.temperature` 的形态、单位与语义**
  - **为何重要**：热 LSTM 的监督就是温度；若为双通道却按标量读，或单位不是 °C，MAE 与阈值无意义。
  - **待确定**：
    - [ ] 单标量（基线假设）
    - [ ] 多元素数组（需记录：下标 0/1 各代表绕组/外壳等，并决定是否改双头网络）
    - [ ] 单位（°C / 0.1°C / 内部码值）
    - [ ] 是否已做滤波或仅原始 ADC 阶梯
  - **建议**：静止与负载两段 bag，对比温度上升曲线与手持测温或驱动器上位机（若有）。

- [ ] **确认 `one.current` 的单位与符号约定**
  - **为何重要**：`tau_est = current * ct_scale` 是核心特征；单位或符号错会导致 `tau_sq` 与真实负载无关。
  - **待确定**：
    - [ ] 安培还是毫安或定点整数
    - [ ] 是否与关节力矩方向约定一致（正电流对应正 `tau` 定义）
  - **建议**：单关节小幅度正弦运动，对照 `pos/speed` 与 `current` 符号是否合理。

- [ ] **实机验证 `Deploy → Ultra T_leg[0..11]` 映射**
  - **为何重要**：**`T_leg` 以 Ultra 顺序为唯一准则**（见 `plan.md` §1.2、`leg_index_mapping.yaml`）。Deploy 中间向量髋序为 **R–P–Y**，Ultra 为 **R–Y–P**；若按下标硬拷贝会错三条髋轴。
  - **待确定**：
    - [ ] 每条 `status` 经 `name`/id 先到 Deploy 中间向量，再**按语义**对齐到 Ultra 的 12 个 `motor_names` 槽位，对应表已固定且无歧义
    - [ ] 左/右腿、膝/踝与 Ultra USD/MJCF 关节名一致
  - **建议**：一次只动一条腿一个关节，看哪一路 `pos` 变化，打印 `name` / index，与 **Ultra** 表核对。

- [ ] **确认 `ct_scale` 与 `current` 下标对齐方式**
  - **为何重要**：`plan.md` 写明 `tau_est` 用的是 `ct_scale[j]`，`j` 为 Deploy 中间向量下标，必须与映射后的关节一致。
  - **待确定**：
    - [ ] `ct_scale` 长度是否等于 `motor_num` 或仅腿子集
    - [ ] 向量顺序与 `getIndexById` 得到的 `index` 是否一致
    - [ ] 乘积量级是否接近 N·m（与关节额定对比数量级）
  - **建议**：对比官方或标定文档；若无，用已知负载工况做粗校验。

- [ ] **明确 `tg22_config.yaml` 中 `dt` 的含义**
  - **为何重要**：数值 `ddq` 用 `diff(speed)/dt`；若 `dt` 不是实际控制/采样周期，加速度特征全错。
  - **待确定**：
    - [ ] `dt` 是否等于 `/leg/status` 有效更新间隔
    - [ ] 若 Topic 频率与 `dt` 不一致，以哪一个为「真」时间步（通常以消息时间戳间隔为准）
  - **建议**：用 bag 统计相邻消息 `stamp` 差分的中位数，与 `dt` 对比。

- [ ] **测量 `/leg/status` 实际发布特性**
  - **为何重要**：`plan.md` 建议 20 Hz 统一网格；若实际 100 Hz 且抖动大，抽帧策略必须写清楚。
  - **待确定**：
    - [ ] 平均频率、最小/最大间隔、丢帧率
    - [ ] 是否适合「每 k 帧取 1 帧」还是必须按时间戳重采样
  - **建议**：10–60 s bag，`ros2 bag info` 或自写脚本统计间隔直方图。

---

## P1 — 特征语义与时间对齐

### P1 说明

P1 保证：**除温度电流外，其余输入在物理上与训练假设一致**；**多 Topic 拼在一起时同一时刻语义对齐**。否则模型在实机上会「离线很好、在线漂移」。

### P1 TODO List

- [ ] **确认 `one.name`（或等价 id 字段）与 `getIndexById` 的输入语义**
  - **为何重要**：映射链路的起点错了，后面全错。
  - **待确定**：
    - [ ] 传入的是字符串名、CAN id 还是二者之一由配置决定
  - **建议**：读 `RLControlNewPlugin.cpp` 与 `bodyIdMap.h` 调用点，与实机一帧原始 `status` 对照。

- [ ] **确认 `one.pos` / `one.speed` 的单位与参考系**
  - **为何重要**：与 Lab 仿真或日志对齐、与 `|dq|` / 数值 `ddq` 一致。
  - **待确定**：
    - [ ] `pos` 是否为弧度、是否连续无跳变（或是否已包裹到 ±π）
    - [ ] `speed` 是否为 rad/s
  - **建议**：与仿真同一动作对比曲线形状（尺度与符号）。

- [ ] **确认 `temperature` 与 `pos/speed/current` 在同一中间向量下标上对齐**
  - **为何重要**：避免出现「温度是腿 3、电流却是腿 5」的拼接错误。
  - **待确定**：
    - [ ] 插件写入 `temperature_midVec` 与 `pos_fed_midVec` 等是否同一套 index
  - **建议**：单关节扰动同时打印四者，确认只有一路同步变化。

- [ ] **确认 `/leg/status` 消息体与 `motor_num` 的关系**
  - **为何重要**：若一条消息含全身电机，采集端必须只取腿 12 路，不能按数组前 12 个元素截取。
  - **待确定**：
    - [ ] 每条消息是否只含腿
    - [ ] 若含多段，如何按 id 过滤出 `T_leg` 对应元素
  - **建议**：打印单帧 `status` 长度与所有 `name`/id。

- [ ] **拿到 `Imu` 消息权威定义并对照插件读取字段**
  - **为何重要**：可选 9 维特征；欧拉角顺序、陀螺/加计坐标系错会导致无效甚至有害输入。
  - **待确定**：
    - [ ] `euler` 是 yaw-pitch-roll 还是其它顺序，单位是否为弧度
    - [ ] `angular_velocity`、`linear_acceleration` 是否在机体系，重力是否已从加计中补偿
  - **建议**：`ros2 interface show` + 静止/绕单轴旋转标定小实验。

- [ ] **测量 `/imu/status` 频率并与 `/leg/status` 对齐策略定稿**
  - **为何重要**：两路频率不同则必须在预处理中明确「谁向谁对齐、是否允许外推」。
  - **待确定**：
    - [ ] IMU 与 leg 的中位时间差
    - [ ] 采用最近邻、线性插值，还是只使用 leg 时间栅格上的 IMU 采样
  - **建议**：同一 bag 提取两者 stamp，画差分分布。

- [ ] **明确在线推理的触发周期与 ring buffer 深度**
  - **为何重要**：训练用长度 `L` 与步长 Δt 必须与在线一致，否则分布偏移。
  - **待确定**：
    - [ ] 推理是否每个控制周期调用，还是固定 50 ms / 100 ms
    - [ ] ring buffer 是否按 wall clock 还是按消息 stamp 推进
  - **建议**：在目标部署线程打时间戳日志，统计调用间隔。

---

## P2 — 训练与工程超参（可迭代）

### P2 说明

P2 不阻塞「是否正确」类问题，但影响 **Horizon、窗口、平滑、消融与延迟预算**；建议在基线跑通后按实验表推进。

### P2 TODO List

- [ ] **确定统一时间网格与落盘采样策略**
  - **为何重要**：`L`、Horizon 的「步」必须对应同一 Δt。
  - **待确定**：
    - [ ] 目标网格频率（如 20 Hz）如何从原始 `/leg/status` 得到
    - [ ] 降采样用抽帧、平均池化还是插值
  - **建议**：在文档中写死一种策略并版本化预处理脚本。

- [ ] **确定预测 Horizon 形态**
  - **为何重要**：`plan.md` 验收为约 15 s；需明确是单点还是多档监督。
  - **待确定**：
    - [ ] 仅预测 t+15 s 一个点
    - [ ] 或多档 {0.5, 1, …, 15} s 多任务头（类似 G1 文档中的多视距）
  - **建议**：与算力、标签噪声权衡；多档通常更稳但更重。

- [ ] **确认序列长度 `L` 是否足够**
  - **为何重要**：热惯性长时，`L` 过短会欠拟合慢升温。
  - **待确定**：
    - [ ] 在典型负载下，自相关或肉眼观察温升曲线，判断 5 s @ 20 Hz 是否够
  - **建议**：对比 `L ∈ {50, 100, 200}` 的验证集 MAE。

- [ ] **确定温度 EMA 的 `α`（若使用）**
  - **为何重要**：uint8 或低分辨率温度在 20 Hz 下台阶明显，EMA 改善梯度但引入滞后。
  - **待确定**：
    - [ ] 是否对输入温度做 EMA；`α` 取值（如 0.05）是否固定
    - [ ] 在线与离线是否同一套平滑
  - **建议**：小网格搜索 + 可视化预测曲线。

- [ ] **确定数值 `|ddq|` 的计算方式**
  - **为何重要**：差分对噪声敏感，方法不同特征分布不同。
  - **待确定**：
    - [ ] 前向差分 vs 中心差分
    - [ ] 是否先对 `speed` 做低通再差分
  - **建议**：消融「有/无 ddq」「两种差分」对 Val MAE 的影响。

- [ ] **消融：邻域温度特征是否加入**
  - **为何重要**：`plan.md` 允许仅 `T_leg` 相邻索引；需数据证明收益。
  - **待确定**：
    - [ ] 是否加入及 `K` 邻居数（如 2）
  - **建议**：与仅本体特征对比 Val MAE 与在线延迟。

- [ ] **消融：IMU 9 维是否加入**
  - **为何重要**：增加维度和同步复杂度。
  - **待确定**：
    - [ ] 基线不含 IMU；是否第二阶段加入
  - **建议**：跑跳、上下坡等 session 上对比。

- [ ] **实测端到端推理延迟分解**
  - **为何重要**：`plan.md` 要求单次前向 ≤ 5 ms（FP16）；预处理若占大头需优化或改 CPU 路径。
  - **待确定**：
    - [ ] 各阶段耗时：解析消息 → EMA/归一化 → 拷入 GPU → TensorRT → 反归一化
    - [ ] 目标硬件（机载 Orin / x86）上的 P50/P99
  - **建议**：用 CUDA event 或高精度计时包裹各段。

---

## 验证手段 TODO（按需勾选执行）

- [ ] 已执行：`ros2 interface show` 并保存 `MotorStatusMsg`、`Imu` 全文
- [ ] 已录制：≥30 s 的 `/leg/status` bag，并统计频率与 stamp 间隔
- [ ] 已录制：同时含 `/leg/status` 与 `/imu/status` 的 bag，用于对齐分析
- [ ] 已在采集节点打印：至少一帧完整 `status` 的 `name`、温度、电流、位置、速度与内部 index
- [ ] 已做：单关节运动学识别（确认 `T_leg` 映射）
- [ ] 已做：静止 / 缓动 / 高负载 三类短录，人工目检温度-电流-速度合理性

---

## 相关文档

- `Tienkung_thermal/docs/plan.md` — 白名单 Topic、特征清单、验收与推理预算
- `Tienkung_thermal/docs/task_memory.md` — Ultra 相对 G1 的调整要点与阶段计划
- `Tienkung_thermal/configs/leg_index_mapping.yaml` — `T_leg[0..11]` 关节名顺序
- `Tienkung_thermal/scripts/check/p0_check.py` — P0 实机核对脚本（用法见上文「P0 核对脚本」）
