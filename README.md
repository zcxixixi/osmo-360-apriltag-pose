# 360 Camera 6DoF Dataset Pipeline

将 360 相机原始视频自动处理为带时间戳的 6DoF 轨迹数据集。当前生产入口优先支持 DJI Osmo 360，并为 Insta360 X5 官方 SDK 接入保留统一接口。

核心流程：

```text
原始视频 → 相机识别 → 工厂标定拼接 → AprilTag/AprilGrid 6DoF
        → 光流恢复 → Kalman + RTS 平滑 → 标准数据集
```

问题根因、坐标变换链、质量门禁和双夹爪共同世界坐标的完整说明见
[`docs/LOCALIZATION_PIPELINE.md`](docs/LOCALIZATION_PIPELINE.md)。

## 能力

- 自动识别 DJI `.OSV`、Insta360 `.insv/.lrv` 和已拼接的 2:1 MP4；
- DJI 使用 PanoForge 从 `djmd` 提取工厂标定和 IMU，生成真实 remap 与接缝融合全景；
- 输出带时间戳的位置、姿态四元数、识别来源和质量状态；
- 支持 AprilGrid 以及非连续 ID 的独立大 Tag 地图；
- 使用双向光流降低重复解码开销，使用 Kalman + RTS 生成连续轨迹；
- 自动按机型和分辨率选择 CPU/CUDA 处理路径；
- 可选导出夹爪 CAD 轨迹视频、逐帧图片和 OptiTrack 真值评估。

## 统一入口

已注册的采集不再手工选择根目录脚本，而是由一个不可变 manifest 锁定输入、
硬件、角度算法、渲染器和输出：

```bash
./umi inspect manifests/captures/x5-20260829-114845-iahea2606kmurq-sdk-r3.json
./umi process manifests/captures/x5-20260829-114845-iahea2606kmurq-sdk-r3.json
./umi review manifests/captures/x5-20260829-114845-iahea2606kmurq-sdk-r3.json
./umi review manifests/captures/x5-20260829-114845-iahea2606kmurq-sdk-r3.json --publish
./umi progress /path/to/pipeline_status.json
./umi review-ui /path/to/processed-dataset-root
```

`inspect` 校验全部哈希；`process` 运行 manifest 管线；`review` 生成不可变审核包；
`progress` 显示各 stage 的执行节点、进度、产物和错误；`review-ui` 提供面向非技术
审核员的中文界面，支持双路画面、3D/真实视频同步审核入口、长视频动作分段、
通过/重处理/拒绝、审核历史、SQLite 持久化和复处理/UMI 导出队列。没有发布
3D 世界姿态审核包的数据不能标记为通过。底层 `render_*`、`calibrate_*`、
`fuse_*` 工具仅用于内部实现或历史复现，其状态可用 `./umi commands --legacy` 查看。

双 X5 数据集的对外入口固定为一个参数：
```bash
./run_pipeline.sh /absolute/path/to/dataset-root
```

调用者只提供数据集根目录；不创建 JSON，不修改厂商文件名，也不执行 `tools/`
内部步骤。输入格式固定为：

```text
dataset-root/
└── raw/
    ├── left/
    │   └── *.insv
    └── right/
        └── *.insv
```

`left`、`right` 始终表示物理左右手。Pipeline 仍会读取 INSV 内嵌序列号并与设备
注册表交叉验证，不能用目录名掩盖放反的相机。当前固定配对为左
`IAHEA2606M5WSK`/BaseTag2、右 `IAHEA2606KKUKF`/BaseTag3。短于 30 秒的误触
录制自动列入忽略清单；其余左右原片按录制时间一一配对，再用音频相关求精确同步。

纯 CPU 的四路鱼眼 MP4 v5 入口同样使用上述单参数命令。若根目录包含
`dataset.h5` 以及 `video/{Left,Right}_{back,forward}.mp4`，入口会直接读取 H5 中的
设备身份与逐帧对齐时间轴，并自动切换到 `dual-x5-four-mp4-cpu-v5`；也继续支持
`raw/left` 与 `raw/right` 各两个 MP4 的通用布局。该流程跳过 SDK 拼接，直接取
HEVC 灰度平面，用 30 Hz LK、局部 ROI 和低频切面恢复缓存四路鱼眼观测。InstaUMI
极速档默认四个进程、每进程四线程；i7-14790F 上四路 10 秒样例 30 Hz
首次无缓存实测 21.77 秒，完整缓存复用约 1.27 秒（v3 检测基线；v5 复用相同
观测算法，保留 0.25 秒可信插值上限，同时为每个时间轴帧保留数值位姿和独立可信度）。
输入描述、资源参数和缓存目录见
[`docs/FOUR_MP4_CPU_PIPELINE.md`](docs/FOUR_MP4_CPU_PIPELINE.md)。

默认由当前工作站和 `current@192.168.109.124` 并行处理。协调器按采集对轮询分配，
两台主机从同一个 NAS 数据集路径读取只读原片；缓存、双镜头 remux、拼接和优化
只写各自主机的 `/tmp/osmo-pipeline/`。节点或 scratch 可在机器部署时通过
`OSMO_PIPELINE_NODES`、`OSMO_PIPELINE_SCRATCH`、`OSMO_REMOTE_REPO` 配置，
日常运行命令不增加参数。

NAS 数据集始终只保留 `raw/` 和 `final/`：

```text
dataset-root/
├── raw/
│   ├── left/*.insv
│   └── right/*.insv
└── final/
    └── dual-x5-raw-v1/
        ├── manifest.lock.json
        ├── status.json
        └── pairs/
            └── pair-<NN>-<HHMMSS>/
                ├── calibration/
                ├── trajectories/
                ├── gates/
                ├── review/
                └── export/
```

`manifest.lock.json` 由 Pipeline 自动生成，不是用户输入。它记录原片相对路径、
内嵌序列号、实际分辨率/FPS、录制配对、设备 revision 和各 stage 身份。失败任务
只向 `final/` 写状态、同步和门禁小报告，不上传大缓存；通过任务保留训练数据、
相机/TCP 轨迹、每视频 A3 地图、完整门禁、审核项目和可复现锁。

命名与坐标规则：

- 数据集根目录：`<task>-<YYYYMMDD>-v<revision>`；
- 采集对：`pair-<两位序号>-<HHMMSS>`；双镜头：`lens-0`、`lens-1`；
- 当前 `20260831` 数据保留实际 2880×2880 双鱼眼、59.94 FPS；`4K30` 只约束后续新采集；
- 时间戳为同步公共时钟秒；位置单位米；四元数顺序 `xyzw`；
- 位姿统一为 `T_parent_child`，把 child 坐标中的点映射到 parent；
- 每个视频独立估计并冻结 `T_session_grid_A_grid_B`，禁止跨视频复用 A/B 摆位；
- 左相机观测墙 Tag 与右 BaseTag3；右相机观测墙 Tag 与左 BaseTag2；
- 官方 2:1 全景仅用于 RGB/审核；BaseTag/TCP 度量必须使用两路原始鱼眼四角。

任一身份、同步、A3 双板、BaseTag 闭环或 TCP 门禁失败时，不生成训练就绪标记。

仓库根目录只保留项目配置和 `umi` 主入口。正式产品代码统一放在
`src/osmo360/`，离线实验工具放在 `tools/`，兼容命令放在 `bin/`，专项说明
放在 `docs/`。新流程优先使用 `./umi`；只有历史复现或专项诊断才运行
`python -m tools.<module>`。
分支、PR、Squash、版本和发布规则见
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)。

光流、插值和预测只用于连续轨迹与可视化。正式精度统计只接受满足要求的直接多 Tag 视觉测量。

## 部署

推荐环境：Ubuntu、Python 3.11–3.13、`uv`。CUDA 为可选加速项。

```bash
sudo apt update
sudo apt install -y gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-ugly

curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/zcxixixi/osmo-360-apriltag-pose.git
cd osmo-360-apriltag-pose

# CPU 部署
uv sync --extra test
# 四 MP4 流水线再安装项目内锁定的离线运行时
.venv/bin/python -m tools.install_ffmpeg_runtime --archive /path/to/ffmpeg-9.0.1-linux-x86_64.tar.xz

# NVIDIA GPU 节点额外安装
uv sync --extra test --extra gpu
```

UMI/VLA 训练节点使用 NVIDIA GPU 时，再安装独立锁定的 PyTorch 环境：

```bash
uv pip install --python .venv/bin/python -r requirements-train-cu130.txt
```

`dual_gripper_3d/` 的浏览器三维渲染器需要 Node.js：

```bash
.venv/bin/python -m tools.install_node_runtime
.venv/bin/python -m tools.install_node_dependencies
```

安装器按 `config/runtime_revisions/node_linux_x64_24_20_0.json` 校验官方归档
SHA-256；运行时也可通过 `OSMO_NODE_BINARY` 显式指定兼容的 Node >=22.12.0。

DJI 原始 `.OSV` 拼接依赖本机 PanoForge，默认查找与本仓库同级的 `panoforge-test/`。也可以在运行时指定：

```bash
./bin/camera-to-dataset input.OSV \
  --panoforge-root /opt/PanoForge \
  --run-name production-001
```

Insta360 `.insv/.lrv` 原片使用官方 Linux MediaSDK。当前固定修订为
CameraSDK 2.1.1 / MediaSDK 3.1.1，清单位于
`config/sdk_revisions/insta360_linux_camera_2_1_1_media_3_1_1.json`。
MediaSDK 与 CameraSDK 分别本地部署到版本化的 gitignored `work/` 目录；
运行时会校验平台、二进制、动态库和模型哈希。拼接阶段默认关闭 FlowState
和方向锁定，以保留用于 6DoF 解算的原始相机运动：

```bash
./bin/camera-to-dataset /data/VID_xxx.insv \
  --insta-sdk-revision config/sdk_revisions/insta360_linux_camera_2_1_1_media_3_1_1.json \
  --run-name insta-x5-001 \
  --max-processed-frames 60
```

若官方硬件编解码与当前驱动不兼容，可追加
`--insta-soft-decode --insta-soft-encode`；若 CUDA 路径不兼容，可追加
`--insta-disable-cuda`。

CameraSDK 通过 USB 控制 X5 时需要安装 udev 权限规则，并将机身 USB
模式设为 Android/SDK（不是 U-Disk）：

```bash
sudo install -m 0644 config/udev/99-insta360-camera-sdk.rules \
  /etc/udev/rules.d/99-insta360-camera-sdk.rules
sudo udevadm control --reload-rules
```

当前物理右 X5 已由 CameraSDK 2.1.1 DeviceDiscovery 验证：serial
`IAHEA2606KMURQ`，型号 `Insta360 X5`，固件 `v1.7.8`。CameraSDK GetFileList
同时确认该设备持有 `/DCIM/Camera01/VID_20260829_114845_00_002.insv`，
因此已有视频也完成了 serial 来源绑定。

多设备不需要逐台重跑视频流水线。udev 规则每台工作站只安装一次，然后用
CameraSDK 批量发现并增量登记序列号。
不想使用命令行时，双击桌面的 `X5设备管理`，或运行：

```bash
./umi devices ui
```

页面提供“扫描已连接 X5”“登记全部”“同步到服务器”和“保存分配”按钮，并显示
serial、固件、物理角色、BaseTag 和设备标签。服务器库存接口是
`http://192.168.111.62:7865/api/devices`。
平台 GET 审阅可在局域网读取，所有写入需要 Bearer 令牌；部署、轮换和客户端用法见
[`docs/VISUALIZATION_PLATFORM_SECURITY.md`](docs/VISUALIZATION_PLATFORM_SECURITY.md)。


```bash
./umi devices scan
./umi devices register
./umi devices assign IAHEA2606KMURQ \
  --role physical_right --base-tag-id 3 --label right-gripper-basetag3
./umi devices list
```

若 20 台同时接在有供电的 USB Hub 上，`scan/register` 一次登记全部设备；若逐台
连接，同一个 `register` 命令会增量合并并保留既有角色分配。设备登记只查询
serial、型号和固件，通常数秒完成，不运行角度、力或视频处理。

### 固定比例相对力

当前方案不改夹爪硬件，继续使用原有三黄点和黑点。黑点间距先减去开口角对应的
固定基线，再减去固定噪声门限，最后映射到该硬件版本统一的 0–100% 相对力。
它不是牛顿值，也不会对每段视频单独归一化。

当前清单：

```text
manifests/captures/x5-20260829-114845-fixed-relative-force-r4.json
```

当前可视化输出为 `force-angle-v16-fixed-relative-scale/` 和
`webgl-v13-fixed-relative-scale/`。该方案只需要现有标记，不要求 TPU 打印件。

最新直接从 SD 卡导入的高分辨率演示为
`manifests/captures/x5-20260830-162856-iahea2606km43a-one-sided-r8.json`：
`IAHEA2606KM43A`、物理左夹爪、BaseTag2、2880×2880、29.97 FPS。双侧完整时
使用双侧结果；遮挡一侧时使用可见侧并标低置信度；两侧都不完整时保留 `N/A`。
该演示使用本次视频的局部 0–100% 形变尺度，不与其他硬件版本直接比较。
冻结基线为
`config/baselines/x5_left_one_sided_force_src_accepted_20260830.json`，使用
`./umi verify` 校验全部当前基线。后续算法修改必须新建 revision 和输出目录，
不能覆盖当前接受版。

未来 Insta360 采集型号为 X5。开始大规模采集前，必须使用实际 X5
序列号完成 CameraSDK 设备发现/录制测试，以及 MediaSDK 原始 INSV/LRV
到官方 2:1 输出的端到端测试；二进制帮助和 dry-run 不能替代实机验证。

### 自动处理策略

| 输入 | 自动处理路径 |
| --- | --- |
| DJI 3K 全景 | CPU 解码 + CPU 投影 + 4 路检测 |
| DJI 高分辨率全景 | CPU 解码 + CUDA 投影（可用时） |
| Insta360 X5 8K | CPU 解码 + CUDA 投影（可用时） |
| 未知 2:1 MP4 | 按文件名与分辨率选择保守配置 |

NVDEC 目前保留为显式实验选项；由于帧仍需下载到 CPU 进行 AprilTag 检测，当前实测不一定比 CPU 解码快。部署时可使用 `--camera-model`、`--decoder` 和 `--projection-backend` 覆盖自动策略。

## 单命令生成数据集

### DJI Osmo 360

```bash
./bin/camera-to-dataset /data/CAM_xxx_D.OSV \
  --output-root /data/processed \
  --run-name robot-motion-001
```

默认 AprilGrid 参数为 6×6、黑色编码区 88 mm、间距比例 0.30。使用独立大 Tag 时必须传入实际地图。双夹爪采集必须使用包含两面墙全部唯一ID的共同世界地图：

```bash
./bin/camera-to-dataset /data/CAM_xxx_D.OSV \
  --tag-map config/room_corner_10tag_world_provisional.json \
  --output-root /data/processed \
  --run-name four-tag-motion-001
```

建议先做小规模验收：

```bash
./bin/camera-to-dataset /data/CAM_xxx_D.OSV \
  --run-name smoke-test \
  --max-processed-frames 60
```

确认 Tag 尺寸、地图、坐标方向和识别率后，去掉 `--max-processed-frames` 执行全量。`PROVISIONAL` 卷尺地图只会生成诊断结果；完成房间标定并冻结地图后才会标记训练就绪。加 `--extract-frames` 才会额外保存 JPEG；默认使用 MP4 + 时间戳 CSV，避免数据集膨胀。

### 双夹爪共同世界坐标

双机轨迹必须直接输出 `T_room_world_panorama_camera`，再按
`T_world_tcp = T_world_camera · T_camera_tcp` 转为夹爪TCP。禁止分别把左右第一帧归零后再用手动动画布局拼接。

```bash
# 编译、校验10 Tag世界地图并查看稳定哈希
uv run python -m osmo360.localization.world_frames \
  config/room_corner_10tag_world_provisional.json \
  --output /data/calibration/room_world_tags.compiled.json

# UMI诊断/导出；episode.json必须声明coordinate_frame与同一地图
./bin/vla-dataset episode.json output/
```

UMI中的 `robot*_eef_pos` 和姿态是共同世界坐标，`robot*_eef_delta_from_start_*` 是附加的单爪起点增量。地图哈希、父子坐标系、相机到TCP方向或标定状态不匹配时，管线会阻止Zarr训练文件输出。

### 输入规则

- `.OSV`：执行 DJI 工厂标定拼接、IMU 提取、6DoF 解算和数据集封装；
- `.insv`：必须解码两路原始 2880×2880 鱼眼轨道；四角先转换为同一 X5 rig
  的单位射线，再参与 A3 双板、BaseTag 和相机/TCP 联合解算。MediaSDK 另行生成
  无 FlowState、无方向锁定的官方 2:1 RGB，仅供审核和导出；
- `.lrv`：仅作快速预览或发现，不能作为度量位姿输入；
- 2:1 MP4：可用于 RGB、人工审核和诊断轨迹；由于近场拼接视差，不能替代 X5
  双原始鱼眼的 BaseTag/TCP 精度门禁。

## 单相机兼容输出
DJI/单相机兼容流程继续输出 `media/panorama.mp4`、`annotations/{trajectory_6dof.csv,pose_direct.csv,detections.jsonl}`、`calibration/`、`sensor/`、`previews/trajectory_overlay.mp4` 和 `metadata.json`。状态仍区分 `direct`、`optical_flow`、`recovered`、`predicted`、`lost`；恢复帧不能作为直接真值。

## Insta360 X5 与 OptiTrack 评估

使用关闭 FlowState、方向锁定和地平线校正的官方 2:1 MP4：

```bash
./bin/x5-mocap-evaluate \
  /data/VID_NO_FLOWSTATE.mp4 \
  /data/motive.csv \
  --confirm-flowstate-off \
  --run-name x5-evaluation-001
```

管线会完成 Motive 多行表头解析、异常刚体分支隔离、时间同步、前 30% 外参标定和后 70% 独立评估，并输出 ATE、姿态误差、RPE、漂移、失锁恢复和 Bootstrap 置信区间。

只有运动相关性、同步不确定度和测试段直接双 Tag 样本数均满足门槛时，结果才标记为 `FORMAL_ACCURACY`。启用光流参与误差计算时会强制标记为 `DIAGNOSTIC_ONLY`。

## 双相机配对与坐标对齐审核

输入左右两台相机的视频及6DoF CSV，以左相机为参考坐标系，自动估计时间偏移和刚性安装外参 `T_left_right`：

```bash
./bin/dual-camera-align-audit \
  left.mp4 left/annotations/trajectory_6dof.csv \
  right.mp4 right/annotations/trajectory_6dof.csv \
  --left-source left_original.OSV \
  --right-source right_original.OSV \
  --output-dir dual-camera-pairs/capture-001
```

工具会生成一个共享的 UUIDv4 `capture_pair_id`，并额外为左右视频生成独立 `asset_id`。同一个 `capture_pair_id` 会写入配对清单、审核报告、资产 sidecar 和逐帧合并 CSV，用于明确两个视频属于同一次同步采集。复算时可使用 `--capture-pair-id <UUIDv4>` 保持配对身份不变。

建议始终传入两路原始容器。工具会读取相机写入的 `creation_time` 和时长并报告绝对时间关系。由于两台运动相机的内部时钟可能相差数分钟，默认仅警告时间区间不重叠，并继续使用音频/运动信号同步；只有确认相机时钟已经校准时，才使用 `--require-wall-clock-overlap` 将其设为硬门槛。

主要输出：

- `capture_pair.json`：双视频配对关系和共享 UUIDv4；
- `alignment_report.json`：时间偏移、同步可信度、左右相机刚性外参和审核结果；
- `aligned_trajectories.csv`：左轨迹、右原始轨迹、右对齐轨迹和逐帧残差；
- `alignment_audit.png`：同一左坐标系中的两条轨迹及位置/姿态误差；
- `left_asset.json` / `right_asset.json`：可随视频一起交付的身份 sidecar。

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

## UMI / VLA 数据集封装

视觉解算完成后，用一个 Episode 清单把视频、6DoF、夹爪角度、双机时间偏移和任务文本统一封装：

```bash
cp config/episode.template.json episode.json
cp config/hardware.template.json hardware.json
./bin/vla-dataset episode.json dataset-output
```

输出包含：

- `episode_arrays.npz`：同步后的 TCP 位姿、轴角姿态、夹宽、测量掩码和双臂 action；
- `episode_metadata.json`：任务、频率、时长、UUID 和是否可训练；
- `quality_report.json`：可信视觉轨迹（Tag刷新＋光流）、Tag定期刷新成功率、最长失锁、同步及硬件标定门槛；
- `dataset.zarr.zip`：通过全部门槛后生成的 UMI 兼容 replay buffer。

相机→TCP 外参或“开合角→实际夹宽”尚未实测时仍可加 `--skip-rgb` 快速验证数据链，但结果会明确标为 `DRAFT_HARDWARE_OR_QUALITY_PENDING`，不会冒充可训练数据。硬件到位后只需补全 `hardware.json` 并重新执行同一命令。

### 训练链路冒烟测试

先用一个很小的数据集确认“双目图像＋双夹爪状态→下一帧动作”能够被模型读取和学习：

```bash
uv run --with-requirements requirements-train-cu130.txt \
  python -m tools.train_zarr_overfit_smoke dataset-output/dataset.zarr.zip \
  --output-dir dataset-output/overfit-smoke
```

脚本严格按照 `episode_ends` 构造转移，不会跨 Episode 连接动作，并输出 checkpoint、预测 CSV、Loss/轨迹图和视频审计。它是数据管线的记忆测试，不是策略泛化精度；正式训练仍需要独立验证集和足够数量的成功 Episode。

### 本地审核网页

```bash
DUAL_GRIPPER_DATA_ROOT=/absolute/path/to/episode-review \
  ./bin/dual-gripper-calibrator
```

浏览器访问 `http://127.0.0.1:7861/umi` 可逐帧检查双目画面、原始/滤波轨迹、异常剔除点、夹宽和训练有效率。网页只读取本地产物，不上传原视频。
审核目录中放置 `dual_gripper_timeline.json`，并将 UMI 产物放在其
`vla-episode/` 子目录；未配置本地数据时 API 会明确返回 404，而不会依赖仓库里的某个历史采集 UUID。

## 精度注意事项

轨迹精度取决于相机工厂标定、Tag 实际尺寸与安装地图、运动模糊、Tag 可见数量、观察距离和标定板平整度。加速模式每隔若干帧才重新解码 Tag，中间帧由可信光流传播，因此发布数据集前应分别检查可信视觉轨迹覆盖率、Tag 定期刷新成功率、重投影 RMSE、最长失锁时间和坐标跳变，不能要求每一帧都直接解码 Tag。
