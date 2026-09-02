# InstaUMI X5 Pipeline

Linux pipeline for converting synchronized dual Insta360 X5 recordings into
InstaUMI datasets, calibrated trajectories, gripper signals, and review assets.

## Input

```text
dataset-root/
└── raw/
    ├── left/*.insv
    └── right/*.insv
```

Camera identity comes from embedded INSV metadata and must match
`config/devices/x5_pairs.json`. Filenames never determine physical left/right.

## Run

```bash
./run_pipeline.sh /absolute/path/to/dataset-root
```

The pipeline:

1. verifies X5 serials, source hashes, SDK revision, and pair timing;
2. estimates the left/right audio offset and common time window;
3. extracts both 1920×1920 fisheye tracks from each camera;
4. extracts both X5 IMU streams from recorded INSV telemetry;
5. detects AprilTags once per lens and reuses signed caches;
6. calibrates the A/B panels and solves left/right camera and TCP trajectories;
7. writes derived trajectory and gripper CSV files;
8. after quality gates pass, writes 1024×1024 H.265 rear-view MP4s and
   `dataset.h5`.

Set `INSTAUMI_PIPELINE_ALIGNMENT_ONLY=1` only for diagnostics. It retains the four
1920 processing videos in scratch and does not publish a complete dataset.

## InstaUMI output

The raw dataset contract is defined by:

`/home/cenxi/Downloads/instaumi_dataset_format.md`

```text
instaumi_xxxxxx/
├── dataset.h5
└── video/
    ├── Left.mp4
    └── Right.mp4
```

`dataset.h5` contains:

- left/right camera packet timelines and immutable source timestamps;
- separate `/sensor/imu/left` and `/sensor/imu/right` gyro/acceleration tracks;
- X5 camera calibration, serials, firmware, hashes, and synchronization;
- optional speaker/audio data when present.

Trajectories, poses, detections, gripper opening, and contact signals are
derived data. They remain CSV files in a separate processing directory and are
never written back into the raw HDF5.

## X5 SDK and telemetry

Pinned vendor revision:

`config/sdk_revisions/insta360_linux_camera_2_1_1_media_3_1_1.json`

- CameraSDK 2.1.1: device discovery, serial, firmware, and live camera control.
- MediaSDK 3.1.1: official X5 file parsing and optional stitched review output.
- telemetry-parser/gyro2bb 0.3.0: recorded INSV gyro/acceleration extraction;
  the executable commit and SHA-256 are checked before every use.

The pipeline fails closed on missing telemetry, wrong camera serial, unexpected
axis layout, non-monotonic source time, invalid SI conversion, or empty data.

## Device management

```bash
./umi devices scan
./umi devices register
./umi devices assign <serial> --role left --base-tag-id 2
./umi devices assign <serial> --role right --base-tag-id 3
./umi devices list
./umi devices sync
./umi devices ui
```

## Review

Review consumes completed outputs; it never recomputes or mutates trajectories.

```bash
./umi review <manifest>
./umi review <manifest> --publish
./umi progress <pipeline_status.json>
```

## Verification

```bash
./umi verify
./.venv/bin/pytest -q
```

Superseded DJI Osmo/OSV implementations and their frozen reproduction assets
live under `archive/legacy_dji_osmo/` and are excluded from the active pipeline.
