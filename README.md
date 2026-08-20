# Osmo 360 AprilTag 轨迹 Demo

从 DJI Osmo 360 全景视频中识别 AprilGrid，估计相机相对标定板的三维位置与姿态，并生成带滤波轨迹的可视化视频。

> 当前结果用于方案验证和模型数据准备，属于 Demo 级轨迹，不作为测量级坐标。

## 真实轨迹演示

动画会在 GitHub 页面中自动播放。点击动画可查看对应的原始 MP4。

### 8

[![8 轨迹演示](sessions/8-trajectory/trajectory_demo.gif)](sessions/8-trajectory/8_trajectory_overlay.mp4)

有效位姿 43/53（81.1%），中位 RMSE 0.404 px。

[轨迹数据 CSV](sessions/8-trajectory/pose.csv) · [结果摘要 JSON](sessions/8-trajectory/summary.json) · [轨迹图 PNG](sessions/8-trajectory/relative_coordinates.png)

### round

[![round 轨迹演示](sessions/round-trajectory/video_preview.gif)](sessions/round-trajectory/round_trajectory_overlay.mp4)

有效位姿 36/54（66.7%），中位 RMSE 0.392 px。

[轨迹数据 CSV](sessions/round-trajectory/pose.csv) · [结果摘要 JSON](sessions/round-trajectory/summary.json) · [轨迹图 PNG](sessions/round-trajectory/relative_coordinates.png)

### round2

[![round2 轨迹演示](sessions/round2-trajectory/video_preview.gif)](sessions/round2-trajectory/round2_trajectory_overlay.mp4)

有效位姿 1/32（3.1%），中位 RMSE 0.779 px。该片段中标定板倾角较大、有效 Tag 不足，主要用于展示识别失败与轨迹预测状态。

[轨迹数据 CSV](sessions/round2-trajectory/pose.csv) · [结果摘要 JSON](sessions/round2-trajectory/summary.json) · [轨迹图 PNG](sessions/round2-trajectory/relative_coordinates.png)

### w

[![w 轨迹演示](sessions/w-trajectory/video_preview.gif)](sessions/w-trajectory/w_trajectory_overlay.mp4)

有效位姿 30/37（81.1%），中位 RMSE 0.723 px。

[轨迹数据 CSV](sessions/w-trajectory/pose.csv) · [结果摘要 JSON](sessions/w-trajectory/summary.json) · [轨迹图 PNG](sessions/w-trajectory/relative_coordinates.png)

## 项目能做什么

- 从工厂标定拼接后的 2:1 全景视频识别 AprilGrid
- 输出相机 XYZ、Roll、Pitch、Yaw 和识别质量
- 对短时断点进行轨迹预测和平滑
- 生成轨迹图与真实画面叠加视频
- 保留逐帧结果，方便后续模型训练和轨迹回放

输入可以是 PanoForge 使用相机工厂标定生成的全景 MP4，也可以是 DJI Studio 导出的 2:1 全景 MP4。原始 OSV、中间全景、remap 和缓存不包含在仓库中。

## 快速开始

```bash
sudo apt install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/zcxixixi/osmo-360-apriltag-pose.git
cd osmo-360-apriltag-pose
uv sync --extra test
```

运行离线轨迹识别：

```bash
uv run python osmo_360_offline.py input_360.mp4 \
  --tag-size 0.088 \
  --spacing 0.30 \
  --sample-fps 5 \
  --output-dir sessions \
  --official-stitched
```

生成轨迹叠加视频：

```bash
uv run python render_trajectory_overlay_video.py \
  input_360.mp4 sessions/example/pose.csv sessions/example/overlay.mp4 \
  --fps 20 --smooth 0.55 --tail-seconds 2
```

也可以启动本地控制页面：

```bash
uv run python control_app.py
```

然后打开 <http://127.0.0.1:7860>。

## 输出文件

- `pose.csv`：逐帧位置、姿态和识别质量
- `summary.json`：有效率、RMSE 和整体统计
- `relative_coordinates.png`：三维及平面轨迹图
- `overlay.mp4`：真实视频与三维轨迹叠加结果

## 注意

轨迹精度会受到全景拼接、运动模糊、标定板倾角、可见 Tag 数量和拍摄距离影响。使用时应同时关注有效位姿比例、RMSE、轨迹连续性和整体运动趋势。

## 测试

```bash
uv run pytest
```
