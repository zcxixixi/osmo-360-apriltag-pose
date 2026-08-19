# Osmo 360 AprilGrid 离线位姿 Demo

从 DJI Osmo 360 双镜头录像估计相机相对 6×6 AprilGrid 的 XYZ、Roll/Pitch/Yaw，并生成轨迹图。默认正式入口是 **DJI Studio 导出的 2:1 等距柱状全景 MP4**。旧单镜头 RTMP 程序仍保留为 Legacy/实验模式。

> 当前 Tag 边长 `0.088 m`、`tagSpacing=0.30` 和全景投影模型均未最终标定。所有结果必须视为 **APPROXIMATE / DEMO-GRADE**，不能作为测量级坐标。

## 为什么不直接“实现 OSV 拼接”

`.OSV` 是 DJI 原片，通常包含两个未拼接圆形鱼眼和可能的私有标定/方向数据。本项目不会猜测或重造 DJI 私有拼接器。FFmpeg `v360=input=dfisheye:output=equirect` 只能在实际圆心、直径、FOV、朝向和接缝都验证后用于实验；出现拉伸、重影或接缝误差时，不得用于坐标测量。`.LRF` 仅供快速预览，不能替代原片。

## Ubuntu 安装

```bash
sudo apt update
sudo apt install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh   # 尚未安装 uv 时
git clone https://github.com/zcxixixi/osmo-360-apriltag-pose.git
cd osmo-360-apriltag-pose
uv sync --extra test
```

相机素材必须先复制，不能在相机挂载目录直接运行写操作：

```bash
mkdir -p work/input
find "/run/user/$(id -u)/gvfs" "/media/$USER" /mnt \
  -type f \( -iname '*.OSV' -o -iname '*.LRF' \) 2>/dev/null
cp --preserve=timestamps /只读挂载/DCIM/示例.OSV work/input/
cp --preserve=timestamps /只读挂载/DCIM/示例.LRF work/input/
```

## OSV 只读检查

```bash
uv run python inspect_osv.py work/input/example.OSV \
  --output-dir work/osv_inspection/example
```

检查视频轨数、HEVC profile、像素格式/色深、分辨率、帧率、时长、metadata/side data，以及两个鱼眼圆的布局、圆心、直径、旋转、AprilGrid 可见区间和运动模糊。若 DJI Studio 可用，优先导出 2:1 MP4 到 `work/input/`。

## 离线处理

```bash
uv run python osmo_360_offline.py work/input/stitched.mp4 \
  --tag-size 0.088 --spacing 0.30 --rows 6 --cols 6 --first-id 0 \
  --sample-fps 5 --output-dir sessions --official-stitched
```

处理器用 `py360convert` 生成每 45° 一个的重叠水平透视视角，另加上下视角；在各视角用 OpenCV `tag36h11` 检测和中心点 RANSAC。Kalibr ID 按列优先排列（0–5 是第一列，6–11 是第二列）。各视角 PnP 结果旋回统一全景相机坐标系，重复 ID 在视角内去重，再按内点数和 RMSE 选择最可靠候选。异常速度只影响过滤轨迹，原始坐标仍写入 CSV。

每次运行创建独立 session，包含：

- `pose.csv`：过滤坐标、原始坐标、姿态、ID、内点、RMSE、视角和质量状态
- `detections.jsonl`：逐帧逐视角检测及候选位姿
- `summary.json`：覆盖率、有效位姿比例、RMSE、跳变和精度标记
- `relative_coordinates.png`：3D/XY/XZ/YZ、起终点和 AprilGrid 原点
- `processor.log`

## 本地控制页

```bash
uv run python control_app.py
```

打开 <http://127.0.0.1:7860>。默认页扫描 `/media/$USER`、`/run/media/$USER`、`/mnt`、`work/input`，后端直接读取文件而非浏览器上传。可以开始/安全停止处理、查看进度和日志，并下载结果。OSV/LRF 会显示但不会被误当成已拼接全景。

Legacy 实时原型在第二个标签页，也可直接运行：

```bash
uv run python osmo_apriltag_demo.py --tag-size 0.088 --spacing 0.30
```

## 坐标与精度限制

AprilGrid 坐标：原点在整板中心，X 向右，Y 向上，Z 垂直离开板面。输出描述全景相机中心在该坐标系中的位置，姿态为全景相机坐标到板坐标的 Roll/Pitch/Yaw。

透视视图虽然具有精确的虚拟针孔内参，但精度仍受 DJI 拼接、双镜头外参、滚动快门、曝光、模糊、Tag 尺寸和打印比例影响。识别到 36/36 只表示覆盖，不表示轨迹正确；必须同时检查有效位姿比例、RMSE、跳变连续性和几何趋势。

## 测试

```bash
uv run pytest
```

合成测试覆盖列优先映射、视角旋转正交性，以及不同透视视角恢复同一全景相机中心。
