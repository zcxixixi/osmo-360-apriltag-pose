# Osmo 360 AprilGrid 坐标 Demo

当前原型输入 Osmo 360 的单镜头 RTMP 直播，经本机 MediaMTX 转为 RTSP，检测 6×6 `tag36h11` AprilGrid，并显示相机相对标定板的 XYZ、Roll、Pitch、Yaw。

下一阶段将切换到双镜头工作流：Osmo 360 将全景原片录入 SD 卡，经 DJI Studio 导出 2:1 等距柱状全景 MP4，再转换为重叠透视视图完成 AprilGrid 位姿估计。

## 一键控制界面

```bash
uv run python control_app.py
```

浏览器打开 <http://127.0.0.1:7860>。点击一次开始实时处理，再点击一次停止并生成相对坐标轨迹 PNG。

## 1. 相机推流

在 DJI Mimo 中选择：单镜头 → 直播 → RTMP。

当前 Mac 的推流地址：

```text
rtmp://<MAC_LAN_IP>:1935/osmo/live
```

手机、相机和 Mac 必须处于同一个局域网。Mac 的地址变化后需要更新上述地址。

## 2. 安装

```bash
cd osmo-360-apriltag-pose
uv sync
```

## 3. 运行

先实测单个黑色 Tag 的边长。假设为 8.8 cm：

```bash
uv run python osmo_apriltag_demo.py --tag-size 0.088 --spacing 0.30
```

按 `q` 或 `Esc` 退出。坐标同时写入 `pose.csv`。

如果标定板不是从 ID 0 开始：

```bash
uv run python osmo_apriltag_demo.py --tag-size 0.088 --first-id 0
```

## 坐标系

- 原点：整块 AprilGrid 的中心
- X：标定板向右
- Y：标定板向上
- Z：垂直离开标定板表面

## 精度说明

没有提供相机内参时，Demo 使用 `--hfov 90` 构造近似内参，窗口会显示 `APPROX INTRINSICS`。此时功能链路可验证，但 XYZ 不是测量级结果。

得到 OpenCV 相机标定文件后运行：

```bash
uv run python osmo_apriltag_demo.py \
  --tag-size 0.088 \
  --camera-yaml camera.yml
```

`camera.yml` 至少包含：

```yaml
%YAML:1.0
camera_matrix: !!opencv-matrix
  rows: 3
  cols: 3
  dt: d
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
dist_coeff: !!opencv-matrix
  rows: 1
  cols: 5
  dt: d
  data: [k1, k2, p1, p2, k3]
```
