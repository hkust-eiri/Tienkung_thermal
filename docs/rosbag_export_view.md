# ROS 2 Bag 导出与离线查看

本文说明如何在**不依赖实机**的情况下，查看本仓库 `data/bags/` 下 **rosbag2**（`metadata.yaml` + `*.db3`）中的 **topic 列表**与**消息字段**，并给出推荐命令。

**相关脚本**：`scripts/bags/extract_bag_topic_samples.py`  
**依赖说明**：仅「原始导出」只需 Python 3 标准库；「解码导出」需 `rosbags` 以及对应消息包的 `.msg` 定义（见下文）。

---

## 1. Bag 目录里有什么

- 每个 `**rosbag2_*`** 子目录 = 一次录制的 **rosbag2** 数据集。
- 必备文件：`**metadata.yaml`**（摘要信息、topic 列表）+ **一个 `*.db3`**（sqlite 存储消息）。
- 同目录下的 `***.txt**` 等为文本日志，**不是** rosbag，用编辑器直接打开即可。

---

## 2. 查看有哪些 Topic

### 2.1 读 `metadata.yaml`（最通用）

打开对应 bag 目录下的 `metadata.yaml`，在 `topics_with_message_count` 中可以看到：

- 每个 topic 的 `**name`**
- 消息类型 `**type**`（如 `bodyctrl_msgs/msg/MotorStatusMsg`）
- `**message_count**`

无需安装 ROS，适合快速浏览。

### 2.2 使用 ROS 2 命令（需已安装并 `source` 工作空间）

```bash
ros2 bag info /path/to/rosbag2_xxx
```

会打印时长、消息总数、各 topic 及类型。

---

## 3. 导出样本：脚本两种模式

脚本从 rosbag2 中按 topic 抽取**少量消息**，便于核对字段与类型。

### 3.1 安装可选依赖（仅解码模式需要）

```bash
cd /path/to/tienkung_thermal
pip install 'rosbags>=0.9'
# 或：pip install -e ".[rosbag]"
```

### 3.2 模式 A：原始导出（默认，无需 `rosbags`）

从 `.db3` 中读取：时间戳、payload 长度、**十六进制前缀**。适合快速确认 bag 内类型字符串、消息是否非空。

```bash
python3 scripts/bags/extract_bag_topic_samples.py \
  data/bags/rosbag2_2026_04_07-16_50_48 \
  --topics /leg/status /leg/motor_status \
  --per-topic 3 \
  --out /tmp/samples_raw.json
```

未加 `--decode` 即为该模式。输出 JSON 中含 `topics_table`（全表 topic 与类型）及各 topic 的原始样本。

### 3.3 模式 B：CDR 解码（结构化字段）

将消息反序列化为可读的嵌套结构（与 `ros2 topic echo` 类似），需：

1. 已安装 `**rosbags**`
2. 至少一个 `**--msg-package**`，指向 **ROS 2 消息包根目录**（该目录下须有 `package.xml` 与 `msg/*.msg`）

**示例**（`bodyctrl_msgs` 常见路径二选一，按本机实际修改）：

- 源码包：`<TienKung_ROS>/src/bodyctrl_msgs`
- 安装 share：`<ros2ws>/install/bodyctrl_msgs/share/bodyctrl_msgs`

```bash
python3 scripts/bags/extract_bag_topic_samples.py \
  data/bags/rosbag2_2026_04_07-16_50_48 \
  --topics /leg/status \
  --per-topic 10 \
  --decode \
  --msg-package /home/js/robot/Tienkung/TienKung_ROS/src/bodyctrl_msgs \
  --out /tmp/leg_status_decoded.json
```

**多自定义类型**：若 bag 中还有其它包定义的类型（例如 `hric_msgs`、`monitor_msgs`），可对每个包各加一行 `--msg-package`，指向对应包的根目录。

---

## 4. 常用参数说明


| 参数              | 含义                                                    |
| --------------- | ----------------------------------------------------- |
| 第一个位置参数         | rosbag2 **目录**（含 `metadata.yaml` 与单个 `.db3`）          |
| `--topics`      | 要导出的 topic 列表；默认仅 `/leg/status` 与 `/leg/motor_status` |
| `--per-topic`   | 每个 topic 最多导出多少条消息                                    |
| `--decode`      | 使用 `rosbags` 做 CDR 解码                                 |
| `--msg-package` | 可重复；消息包根目录（含 `package.xml`）                           |
| `--out`         | 输出 JSON 路径；不写则打印到标准输出                                 |


---

## 5. 类型与 Topic 的对应关系（简要）

- **Topic**：通道名字（如 `/leg/status`）。
- **消息类型**：该通道上数据的结构体名称（如 `bodyctrl_msgs/msg/MotorStatusMsg`）。

同一**类型**可被多个 topic 使用；不同 topic 也可使用不同类型。本仓库常见腿部相关：


| Topic               | 常见类型              | 说明                                          |
| ------------------- | ----------------- | ------------------------------------------- |
| `/leg/status`       | `MotorStatusMsg`  | `MotorStatus[]`，含位置、速度、电流、`temperature`、电压等 |
| `/leg/motor_status` | `MotorStatusMsg1` | `MotorStatus1[]`，字段更少，偏温度（如电机温与 MOS 温）      |


具体字段以对应 `.msg` 为准（`TienKung_ROS` 或安装空间下的 `share/bodyctrl_msgs/msg/`）。

---

## 6. 补全缺失的 `metadata.yaml`

部分 `rosbag2_`* 目录可能缺失或存在空的 `metadata.yaml`（录制中断、拷贝不完整等）。缺失 metadata 会导致 `ros2 bag info` 和导出脚本无法识别该 bag。

### 6.1 检查哪些 bag 缺失 metadata

```bash
cd data/bags/bag0413
for d in rosbag2_*/; do
  if [ -f "$d/metadata.yaml" ] && [ -s "$d/metadata.yaml" ]; then
    echo "OK: $d"
  else
    echo "MISSING: $d"
  fi
done
```

### 6.2 使用 `rebuild_metadata.py` 从 db3 重建

脚本位于 `scripts/bags/rebuild_metadata.py`，直接读取 `.db3`（SQLite）中的 `topics` / `messages` 表，重建完整的 `metadata.yaml`。适用于 `ros2 bag reindex` 不可用（Humble 已知 segfault）的情况。

```bash
# 批量处理某个 bags 目录下所有 rosbag2_*
python scripts/bags/rebuild_metadata.py data/bags/bag0413

# 单个 bag 目录
python scripts/bags/rebuild_metadata.py data/bags/bag0413/rosbag2_2026_04_07-16_40_19
```

脚本会自动跳过已有有效 metadata 的目录，仅处理缺失或空文件的。损坏的 `.db3`（`database disk image is malformed`）会被跳过并打印警告，对应 file entry 记录为 0 条消息。

### 6.3 使用 `ros2 bag reindex`（备选）

若 ROS 2 环境正常且未触发 segfault：

```bash
source /opt/ros/humble/setup.bash
ros2 bag reindex /path/to/rosbag2_xxx -s sqlite3
```

---

## 7. 将 Bag 导出为 500 Hz HDF5 数据集

入口脚本：`scripts/bags/export_leg_status_dataset.py`；详细数据格式见 `docs/dataset_leg_status_h5.md`。

### 7.1 安装依赖

```bash
cd /path/to/Tienkung_thermal
pip install -e ".[rosbag]"
```

### 7.2 批量处理整个 bags 目录

```bash
python scripts/bags/export_leg_status_dataset.py \
  --bags-root data/bags/bag0413 \
  --out-dir data/processed/leg_status_500hz \
  --msg-package /home/js/robot/Tienkung/ros2ws/install/bodyctrl_msgs/share/bodyctrl_msgs \
  --ct-scale-config configs/ct_scale_profiles.yaml \
  --manifest data/processed/leg_status_500hz/manifest.csv
```

### 7.3 处理单个 bag

```bash
python scripts/bags/export_leg_status_dataset.py \
  data/bags/bag0413/rosbag2_2026_04_07-12_14_44 \
  --out-dir data/processed/leg_status_500hz \
  --msg-package /home/js/robot/Tienkung/ros2ws/install/bodyctrl_msgs/share/bodyctrl_msgs
```

### 7.4 参数说明


| 参数                     | 含义                                                  |
| ---------------------- | --------------------------------------------------- |
| `--bags-root`          | 批量模式，扫描目录下所有 `rosbag2_*` 子目录                        |
| 第一个位置参数                | 单个 bag 模式，指向一个 `rosbag2_*` 目录                       |
| `--out-dir`            | HDF5 输出目录                                           |
| `--msg-package`        | `bodyctrl_msgs` 消息定义路径（含 `package.xml`）；脚本会自动探测默认路径 |
| `--ct-scale-config`    | 力矩系数配置，影响 `tau_est` 计算                              |
| `--manifest`           | 输出索引 CSV，记录每个 session 的统计信息                         |
| `--skip-existing`      | 跳过已有 `.h5` 的 session，仅补写 manifest                   |
| `--overwrite-manifest` | 覆盖已有 manifest                                       |


输出为 `data/processed/leg_status_500hz/<session_id>.h5`，每个 session 一个文件，包含 500 Hz 重采样后的 12 关节数据（`q`、`dq`、`current`、`temperature`、`voltage`、`tau_est` 等）。

---

## 8. 与本仓库其它文档的关系

- 录制流程、白名单 topic、session 约定：见 `**docs/recording_operations.md**`。
- 数据格式与训练侧约定：见 `**README.md**` 与 `**docs/plan.md**`。

README 中若提到 `scripts/extract_from_rosbag.py`，请以本仓库实际提供的 `**scripts/bags/extract_bag_topic_samples.py**` 为准。