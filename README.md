# 360 Camera 6DoF Dataset Pipeline

将 360 相机原始视频自动处理为带时间戳的 6DoF 轨迹数据集。当前生产入口优先支持 DJI Osmo 360，并为 Insta360 X6 官方 SDK 接入保留统一接口。

核心流程：

```text
原始视频 → 相机识别 → 工厂标定拼接 → AprilTag/AprilGrid 6DoF
        → 光流恢复 → Kalman + RTS 平滑 → 标准数据集
```

## 能力

- 自动识别 DJI `.OSV`、Insta360 `.insv/.lrv` 和已拼接的 2:1 MP4；
- DJI 使用 PanoForge 从 `djmd` 提取工厂标定和 IMU，生成真实 remap 与接缝融合全景；
- 输出带时间戳的位置、姿态四元数、识别来源和质量状态；
- 支持 AprilGrid 以及非连续 ID 的独立大 Tag 地图；
- 使用双向光流降低重复解码开销，使用 Kalman + RTS 生成连续轨迹；
- 自动按机型和分辨率选择 CPU/CUDA 处理路径；
- 可选导出夹爪 CAD 轨迹视频、逐帧图片和 OptiTrack 真值评估。

光流、插值和预测只用于连续轨迹与可视化。正式精度统计只接受满足要求的直接多 Tag 视觉测量。

## 部署

推荐环境：Ubuntu、Python 3.11–3.13、`uv`。CUDA 为可选加速项。

```bash
sudo apt update
sudo apt install -y ffmpeg gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-ugly

curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/zcxixixi/osmo-360-apriltag-pose.git
cd osmo-360-apriltag-pose

# CPU 部署
uv sync --extra test

# NVIDIA GPU 节点额外安装
uv sync --extra test --extra gpu
```

DJI 原始 `.OSV` 拼接依赖本机 PanoForge，默认查找与本仓库同级的 `panoforge-test/`。也可以在运行时指定：

```bash
./camera-to-dataset input.OSV \
  --panoforge-root /opt/PanoForge \
  --run-name production-001
```

### 自动处理策略

| 输入 | 自动处理路径 |
| --- | --- |
| DJI 3K 全景 | CPU 解码 + CPU 投影 + 4 路检测 |
| DJI 高分辨率全景 | CPU 解码 + CUDA 投影（可用时） |
| Insta360 X6 8K | CPU 解码 + CUDA 投影（可用时） |
| 未知 2:1 MP4 | 按文件名与分辨率选择保守配置 |

NVDEC 目前保留为显式实验选项；由于帧仍需下载到 CPU 进行 AprilTag 检测，当前实测不一定比 CPU 解码快。部署时可使用 `--camera-model`、`--decoder` 和 `--projection-backend` 覆盖自动策略。

## 单命令生成数据集

### DJI Osmo 360

```bash
./camera-to-dataset /data/CAM_xxx_D.OSV \
  --output-root /data/processed \
  --run-name robot-motion-001
```

默认 AprilGrid 参数为 6×6、黑色编码区 88 mm、间距比例 0.30。使用独立大 Tag 时必须传入实际地图：

```bash
./camera-to-dataset /data/CAM_xxx_D.OSV \
  --tag-map mocap-evaluation/config/insta360_x6_tag_map.json \
  --output-root /data/processed \
  --run-name four-tag-motion-001
```

建议先做小规模验收：

```bash
./camera-to-dataset /data/CAM_xxx_D.OSV \
  --run-name smoke-test \
  --max-processed-frames 60
```

确认 Tag 尺寸、地图、坐标方向和识别率后，去掉 `--max-processed-frames` 执行全量。加 `--extract-frames` 才会额外保存 JPEG；默认使用 MP4 + 时间戳 CSV，避免数据集膨胀。

### 输入规则

- `.OSV`：执行 DJI 工厂标定拼接、IMU 提取、6DoF 解算和数据集封装；
- `.insv/.lrv`：识别为 Insta360 原始文件，当前返回 `waiting_for_insta360_sdk`，不会用通用投影伪造结果；
- 2:1 MP4：跳过原始鱼眼拼接，直接进行轨迹解算和数据集封装。

## 数据集结构

默认输出到 `camera-datasets/<run-name>/dataset/`：

```text
dataset/
├── media/
│   └── panorama.mp4
├── annotations/
│   ├── trajectory_6dof.csv
│   ├── pose_direct.csv
│   └── detections.jsonl
├── calibration/
├── sensor/
├── previews/
│   └── trajectory_overlay.mp4
└── metadata.json
```

- `trajectory_6dof.csv`：时间戳、第一帧原点坐标、位置、四元数、测量来源和状态；
- `pose_direct.csv`、`detections.jsonl`：可审计的直接视觉结果；
- `calibration/`：相机工厂标定与坐标系信息；
- `sensor/`：DJI IMU 数据；
- `metadata.json`：数据集版本、输入身份、参数、坐标约定和质量统计；
- `trajectory_overlay.mp4`：可选的 6DoF/夹爪模型质检视频。

状态字段区分 `direct`、`optical_flow`、`recovered`、`predicted` 和 `lost`，下游训练或评估时不能把恢复帧当作直接真值。

## Insta360 X6 与 OptiTrack 评估

使用关闭 FlowState、方向锁定和地平线校正的官方 2:1 MP4：

```bash
./x6-mocap-evaluate \
  /data/VID_NO_FLOWSTATE.mp4 \
  /data/motive.csv \
  --confirm-flowstate-off \
  --run-name x6-evaluation-001
```

管线会完成 Motive 多行表头解析、异常刚体分支隔离、时间同步、前 30% 外参标定和后 70% 独立评估，并输出 ATE、姿态误差、RPE、漂移、失锁恢复和 Bootstrap 置信区间。

只有运动相关性、同步不确定度和测试段直接双 Tag 样本数均满足门槛时，结果才标记为 `FORMAL_ACCURACY`。启用光流参与误差计算时会强制标记为 `DIAGNOSTIC_ONLY`。

## 运维与复现

- `--dry-run`：检查输入并打印将执行的命令；
- `--force`：重跑已经存在的阶段；
- `--output-root`：将运行产物写入指定数据盘；
- `pipeline.log` / `processor.log`：完整处理日志；
- `metadata.json` / `pipeline_manifest.json`：输入身份、参数和结果清单。

代码更新后执行：

```bash
uv sync --extra test
uv run pytest -q
```

## 精度注意事项

轨迹精度取决于相机工厂标定、Tag 实际尺寸与安装地图、运动模糊、Tag 可见数量、观察距离和标定板平整度。发布数据集前应检查直接视觉覆盖率、重投影 RMSE、最长失锁时间和坐标跳变。
