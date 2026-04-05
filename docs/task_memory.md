# TienKung Ultra 腿部热建模 — 任务记忆（关键信息汇总）

> **关节顺序**：以 **`TienKung-Lab`** 为准（`legged_lab/envs/ultra/ultra_env.py` 中 `find_joints`）。  
> **数据**：**仅**使用 **`Deploy_Tienkung`** 与 **`TienKung-Lab`** 两仓库内**已定义**的接口（源码/配置/README/注释中的布局）；其余一律为 **待获取备选**，不得默认存在。  
> 详细条款见 `Tienkung_thermal/docs/plan.md` §0、§1。

---

## 1. 目标与范围

| 项 | 内容 |
|:---|:-----|
| 机器人 | 天工 **Ultra**（与 Lab `ultra_env` 一致） |
| 自由度 | **12** 腿，顺序 = Lab；`Tienkung_thermal/configs/leg_index_mapping.yaml` 与 Lab **对齐引用** |
| 验收 | 12 关节**等权**平均 MAE ≤ 1.5°C（@15s 等视距以 `plan.md` 为准） |
| 权重 | 配置 `w_i` + 可选可学习；Gate 只看**等权** MAE |
| 推理 | ≤ 5 ms FP16 |

---

## 2. 准许使用的数据（白名单）

### 2.1 Deploy_Tienkung（`RLControlNewPlugin.cpp` + `tg22_config.yaml` + `bodyIdMap.h`）

**Topic**

| Topic | 消息类型（头文件） |
|:------|:-------------------|
| `/leg/status` | `bodyctrl_msgs::msg::MotorStatusMsg` |
| `/arm/status` | 同上 |
| `/imu/status` | `bodyctrl_msgs::msg::Imu` |
| `/sbus_data` | `sensor_msgs::msg::Joy` |
| `/leg/cmd_ctrl`、`/arm/cmd_ctrl`、`/waist/cmd_pos` | 发布侧，见插件 |

**每条 `MotorStatusMsg.status[]` 元素 `one`（插件已读）**

- `pos`, `speed`, `current`, `temperature`, `name`（用于 `getIndexById`）

**`tg22_config.yaml`（插件加载）**

- `dt`, `ct_scale`, `motor_num`, …（热相关主要为 **dt、ct_scale**）

**`Imu`（插件已读成员）**

- `euler.{yaw,pitch,roll}`, `angular_velocity.{x,y,z}`, `linear_acceleration.{x,y,z}`（共 9 标量）

**映射表**

- `bodyIdMap.h`：腿 **0–11** 名称顺序（**≠** Lab 髋轴顺序）；用于 **名称/CAN → 中间向量下标**，再 **置换** 到 Lab 的 `T_leg[0..11]`。

### 2.2 TienKung-Lab

- `README.md`：版本、任务 `ultra_walk`/`ultra_run`、数据集路径、`train`/`play`/`sim2sim_ultra` 命令。  
- `ultra_env.py`：**腿关节名顺序**；`visualize_motion` **44 维**帧说明；`get_amp_obs_for_expert_trans` **AMP 布局**文档字符串。  
- `motion_loader*.py`：JSON `Frames` / `FrameDuration` / `MotionWeight`（**无温度**）。  
- 仿真：`robot.data.joint_pos` / `joint_vel` + `*_leg_ids`（**无温度字段**）。

---

## 3. 待获取备选数据（两仓库未定义或未接入）

| 项 | 说明 |
|:---|:-----|
| `bodyctrl_msgs` `.msg` 全文 | 两仓库无包源码；`temperature` 类型/通道数/单位待澄清 |
| 电压、原生 `ddq`、故障/CRC | 插件未读；其它 Topic 未在两仓库列出 |
| BMS、主板温、风扇 | 插件未订阅 |
| 环境温度计 | 非仓库接口 |
| Lab 电机温度仿真 | 仓库内未出现 |
| 第三方机器人 DDS | 本专项基线排除 |

**允许的后处理**：对已白名单的 `speed` 等做**数值差分**、EMA，**不新增 Topic**。

---

## 4. Deploy 腿索引 vs Lab 顺序（映射提醒）

`bodyIdMap.h` 腿名顺序：**l_hip_roll, l_hip_pitch, l_hip_yaw, …**（髋 **R–P–Y**）。  
Lab / `T_leg`：**hip_roll_l_joint, hip_yaw_l_joint, hip_pitch_l_joint, …**（髋 **R–Y–P**）。  

→ **`T_leg[i]` 不得等于 Deploy `pos_fed_midVec(i)`**；必须 **按语义重排**。

---

## 5. 关键路径

| 路径 | 用途 |
|:-----|:-----|
| `Tienkung_thermal/docs/plan.md` | 完整计划与白名单 |
| `Tienkung_thermal/configs/leg_index_mapping.yaml` | 与 Lab 同序的 `motor_names` |
| `TienKung-Lab/legged_lab/envs/ultra/ultra_env.py` | **关节顺序权威** |
| `Deploy_Tienkung/.../RLControlNewPlugin.cpp` | Topic 与 status/IMU 读取 |
| `Deploy_Tienkung/.../bodyIdMap.h` | Deploy 腿索引与 CAN |
| `Deploy_Tienkung/rl_control_new/config/tg22_config.yaml` | `dt`, `ct_scale` |

---

*与 `plan.md` 同步；待获取项澄清后可迁入 `plan.md` §1.3–1.4 白名单。*
