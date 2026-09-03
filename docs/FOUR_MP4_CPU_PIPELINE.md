# Four-MP4 CPU pipeline v10

`dual-x5-four-mp4-cpu-v13` accepts the four independent raw fisheye MP4 streams
produced by two X5 cameras. It does not import INSV and does not invoke the
Insta360 stitching SDK. The official panorama is therefore no longer a
localization prerequisite.

## InstaUMI HDF5 input contract

The native input accepted by the optimized profile is:

```text
dataset-root/
├── dataset.h5
└── video/
    ├── Left_back.mp4
    ├── Left_forward.mp4
    ├── Right_back.mp4
    └── Right_forward.mp4
```

`dataset.h5` supplies the dataset ID, camera serial numbers, exact per-frame
aligned timestamps, original source timestamps, alignment uncertainty, and the
original right/left time offset. The exported MP4 files are already aligned,
so the source offset is retained for audit but is not applied again. The H5
rear preview declares source stream 0 and frame-matches `*_back`; consequently
`back=stream-0` and `forward=stream-1`.

Declared sub-frame first-frame offsets are preserved rather than independently
zeroing each sensor clock. Left/right H5 frame indices are treated as the
already aligned pairing, a maximum 10 ms paired-timestamp difference is
enforced, and the stable right-camera 29.97 Hz timeline is published as the
joint timestamp. The observed maximum pairing delta is recorded in the report.

When present, the rear-lens Kannala-Brandt intrinsics and
`T_rig_camera_left/right` rotations in H5 are used directly for stream 0.
Stream 1 uses the serial-bound X5 factory lens record because the current H5
schema exposes only the active rear lens. The complete H5 hash and calibration
hash are cache inputs, so replacing H5 invalidates stale bearings even when the
four MP4 files are unchanged. `T_right_left` is not used: the two cameras move
independently and are localized against the same fixed AprilGrid map.

The observation worker launches the verified project FFmpeg executable as a
bounded subprocess and transports selected `gray8` luma frames to Python over
`rawvideo` stdout. FFmpeg performs accurate timestamp seeking at chunk starts,
selects only the requested decode cadence, and emits an exact expected frame
count. Python validates every `width x height` payload, the final byte boundary,
and the subprocess exit code. No BGR/RGB frame is produced.

Optional IMU assistance is fail-closed. Because the two hand cameras move
independently, usable H5 data must contain `/sensor/imu/left/...` and
`/sensor/imu/right/...` (or `/sensor/{left,right}/imu/...`). A singular
`/sensor/imu` stream is never silently assigned to both hands. IMU and visual
data are matched by their H5 `timestamp_ns` values and interpolated in the
shared `dataset_start` clock; frame-index matching and fixed baseline time
offsets are forbidden.

IMU rotation calibration has an explicit precedence. Valid non-identity
`T_rig_camera_left/right` and `T_rig_imu_left/right` matrices in
`calibration_full` win. Identity or null placeholder matrices use the immutable,
rig-side-checked, serial-bound rotation-only baseline in
`config/imu_revisions/x5_kmdgp_kmurq_visual_gyro_20260902_r1.json`. An unknown
serial never receives another camera's baseline. If neither source is usable,
v12 may perform the existing fail-closed capture-local visual/gyro fit; it is
enabled only with at least 200 excited pairs, speed-norm correlation of at least
0.70, held-out median/p95 residuals no greater than 0.50/2.0 degrees, and
cross-side rotation agreement within 10 degrees.

Calibrated gyro propagation supplies a short-horizon attitude check and shapes
orientation between exact visual endpoints. Timestamp-aligned accelerometer
samples also shape translation inside a visually bounded gap. The mean
world-frame specific force is removed over that interval to cancel gravity and
constant bias, both visual positions remain exact metric anchors, and deviation
from linear visual interpolation is capped at 0.15 m. A gyro bridge whose visual
endpoint closure exceeds 20 degrees is rejected before acceleration is used.
This is not unbounded raw
double integration and is not used beyond the last visual anchor. Gaps no longer
than 0.25 s are `IMU_ASSISTED`; longer gaps remain
`IMU_ASSISTED_UNTRUSTED`. Missing, empty, ambiguous, poorly sampled, or invalid
IMU data falls back to visual interpolation/SLERP and is recorded in
`report.json`.

## Generic four-MP4 input contract

The default layout is:

```text
dataset-root/
└── raw/
    ├── four-mp4.json
    ├── left/
    │   ├── lens-0.mp4
    │   └── lens-1.mp4
    └── right/
        ├── lens-0.mp4
        └── lens-1.mp4
```

`left` and `right` mean physical left/right. The two MP4s under one side must
have the same size, FPS, duration, and frame count. If the MP4 files retain the
camera serial and X5 `m2_...`/`n2_...` lens-offset record, `four-mp4.json` is
optional. Otherwise use:

```json
{
  "schema_version": "dual-x5-four-mp4-input/1.0",
  "pair_id": "pair-01-session-name",
  "cameras": {
    "left": {
      "serial": "IAHEA2606M5WSK",
      "base_tag_id": 2,
      "x5_offset": "<the complete m2_/n2_ record from the source camera>",
      "lenses": [
        "raw/left/lens-0.mp4",
        "raw/left/lens-1.mp4"
      ]
    },
    "right": {
      "serial": "IAHEA2606KKUKF",
      "base_tag_id": 3,
      "x5_offset": "<the complete m2_/n2_ record from the source camera>",
      "lenses": [
        "raw/right/lens-0.mp4",
        "raw/right/lens-1.mp4"
      ]
    }
  },
  "sync": {
    "offset_s": 0.0
  },
  "processing": {
    "cache_workers": 1,
    "threads_per_worker": 2,
    "trajectory_observation_fps": 30,
    "cache_chunk_duration_s": 120
  }
}
```

Omit `sync.offset_s` to estimate left/right offset from the first 120 seconds
of lens-0 audio. If lens-0 has no audio, an explicit offset is required.

This descriptor remains available for four-MP4 exports that do not use the
InstaUMI HDF5 schema.

## Verified FFmpeg runtime

The v6 discovery/audio path and the joint review-video encoder require the
project runtime `ffmpeg-linux-x64-9.0.1-osmo1`. Both executables are checked
against the hashes in
`config/runtime_revisions/ffmpeg_linux_x64_9_0_1.json`; legacy FFmpeg versions
fail closed instead of silently taking part in a run. Install the offline,
prebuilt archive once per checkout:

```bash
.venv/bin/python -m tools.install_ffmpeg_runtime \
  --archive /path/to/ffmpeg-9.0.1-linux-x86_64.tar.xz
```

The revision also records the signed upstream source hash, signing-key
fingerprint, license, compiler, configure flags, and the fact that network
protocols were disabled. The runtime lives under gitignored `work/tools/`, so
it does not alter the host FFmpeg package. `OSMO_FFMPEG_BIN` may override it
only with a complete `ffmpeg`/`ffprobe` pair at version 9.0.1 or newer.

## Execution and resource limits

Run the same one-argument entry point:

```bash
./run_pipeline.sh /absolute/path/to/dataset-root
```

For the native InstaUMI `dataset.h5 + video/` layout, the one-argument CSV
product entry point is:

```bash
./bin/process_instaumi_dataset.sh /absolute/path/to/dataset-root
```

To publish only the joint trajectory and skip gripper analysis:

```bash
./bin/process_instaumi_dataset.sh --trajectory-only /absolute/path/to/dataset-root
```

This mode writes `processed/trajectory.csv` and preserves unrelated existing files
under `processed/`.

The default full export first runs or resumes the v12 shared-map trajectory
pipeline, then reads the H5 serials/timestamps and the two 1920x1920
`*_back.mp4` gripper views. Camera serial numbers are output provenance rather
than a detector gate: the yellow-dot detector measures the image on every
capture, while the opening-width calibration is bound to the physical
left/right gripper and BaseTag role. `metadata.csv` records both the actual
dataset camera serial and the source camera used to establish the diagnostic
calibration. A cross-serial result remains `training_ready=0`; detector coverage
must be audited rather than treating camera identity as accuracy evidence.

The H5 timeline is the bounded processing range: a source MP4 may retain
verified trailing encoded frames, but missing source frames or any request past
the H5 endpoint remains an error. The
role/BaseTag/mount-bound jaw calibration produces angle and calibrated jaw
width while preserving direct, low-confidence one-sided, short-gap recovered,
and unavailable states.  It never derives a zero from each input episode.

The atomically published CSV product is written directly under the dataset:

```text
processed/
├── trajectory.csv  # v12 joint trajectory re-expressed in world FLU
├── gripper.csv     # synchronized left/right opening angle, width and state
├── processed.csv   # trajectory and gripper columns joined at v12 timestamps
├── metadata.csv    # revisions, source/target frames, rate and quality status
└── time_alignment.csv  # preserved when already present
```

All published camera pose fields use one right-handed FLU contract.  The world
origin is the midpoint of the geometric centers of `grid_A` and `grid_B`; world
`+X` follows the AprilTag corner-winding normal through the printed grids
toward their rear, world `+Y` is left when looking along `+X`, and world `+Z`
is physical up.  The child frame remains `hand_camera_flu_back_x`, so each
quaternion represents `T_world_flu_hand_camera_flu`.  `metadata.csv` records
the native source frame, source-frame origin and complete source-to-world-FLU
rotation so the conversion is auditable. The trajectory CSV also carries
`left/right_parent_frame=world_flu_aprilgrid_midpoint` and
`left/right_child_frame=hand_camera_flu_back_x` explicitly on every row.

The shell entry shows the active stage, elapsed time, aggregate four-stream
frame/chunk progress, percentage and ETA in the terminal. After all CSV files
are published successfully it removes the generated v10 `final/` work tree;
failed runs retain it for diagnosis. The hidden `.osmo-cache` is retained so a
repeat run can reuse decoded observations.

Empty opening values are intentional when a visual gap exceeds 0.25 seconds.
This jaw signal remains diagnostic (`training_ready=0`); the pose CSV retains
the v10 `hand_camera_flu_back_x` camera frame and does not silently claim a TCP
trajectory.

The generic conservative profile defaults to one detector process and two
OpenCV/FFmpeg decoder threads. The InstaUMI fast profile runs four lens processes with
four threads each (bounded at 16 active CPU threads), uses about 1 GiB for four
resident decoders, and requires neither CUDA nor panorama stitching.

Deployment overrides:

```bash
OSMO_CPU_WORKERS=4 \
OSMO_THREADS_PER_WORKER=2 \
OSMO_MAX_CONCURRENT_JOBS=1 \
OSMO_JOB_SLOT_TIMEOUT_S=3600 \
OSMO_TRAJECTORY_FPS=30 \
OSMO_DECODE_FPS=30 \
OSMO_CACHE_CHUNK_SECONDS=120 \
OSMO_PIPELINE_CACHE=/fast-local-disk/osmo-cache \
./run_pipeline.sh /data/session
```

`OSMO_CPU_WORKERS` is capped at 4 and `OSMO_THREADS_PER_WORKER` at 8. Each job
is additionally capped at 16 logical threads and the profile is automatically
scaled down on smaller hosts. A user-scoped host lock admits one job by default;
`OSMO_MAX_CONCURRENT_JOBS` may be raised only when its aggregate thread budget
does not exceed the host's logical CPUs. Separate repository clones owned by
the same user share the lock. For a shared CPU server, `1 x 2` is the
conservative setting; the tested 9950X speed setting is `4 x 4` with one job.
BLAS, OpenMP, OpenCV and FFmpeg decoder thread pools inherit the same bound;
the parallel pose-graph uses one math thread per Python worker to avoid nested
oversubscription. `OSMO_TRAJECTORY_FPS` controls measured corner samples written
to the cache. H5 timestamps remain 59.94 Hz even though images are retrieved
for detection/tracking at 30 Hz.

On the supplied four-stream, 1920×1920, 59.94 FPS, 10-second dataset, the v3
30 Hz pipeline measured 21.77 seconds uncached and 1.27 seconds fully cached on
an Intel i7-14790F with the `4 x 4` fast profile. The uncached measurement covers
H5 ingest, hashing, four-stream Tag observation caches, shared-map
self-calibration, and synchronized trajectory export. The earlier 4.75-second
Ryzen 9 9950X observation-cache result belonged to the v2 15 Hz cadence and is
not a v3 end-to-end benchmark.

## Resume and cache identity

Every lens is split into independently committed chunks (120 seconds by
default). A completed chunk is reused only when all of the following still
match:

- source MP4 SHA-256;
- complete H5 SHA-256, H5 rear calibration, lens stream, and factory offset;
- synchronized clock mapping;
- trajectory output stride and the complete temporal-search signature;
- decoded frame range and CPU thread setting.

The source hash is computed once and reused while file size and nanosecond
mtime remain unchanged. Outputs and JSON sidecars are atomically renamed, so a
killed process leaves the last incomplete chunk invalid and reruns only that
chunk.

The InstaUMI fast detector obtains the HEVC luma plane directly without an RGB
conversion. Half-resolution pyramidal LK updates known corners at 30 Hz;
forward/backward validation and merged local ROI decoding run at 5 Hz and 2 Hz
respectively. A 0.35-scale grayscale frame is searched globally at 2 Hz to
discover new Tags. The 11 tangent-view rectifications run at most 0.5 Hz and
only while BaseTag 2/3 or sufficient wall support is missing. The sidecar
records every flow, redetection, scout, fallback, and rejection count.

Persistent cache defaults to:

```text
dataset-root/.osmo-cache/<dataset-name>/dual-x5-four-mp4-cpu-v13/<pair-id>/
```

Set `OSMO_PIPELINE_CACHE` to a server-local SSD if the dataset itself is on a
NAS.

## Tracking stage

The four per-lens caches are merged into one calibrated dual-fisheye bearing
cache per physical camera. The existing cached joint pose-graph optimizer then
uses those two caches; it does not decode or stitch video.

For native InstaUMI input, the worker automatically estimates the rigid
relationship between the fixed A3 grids (IDs 200-205 and 210-215) from both
cameras' overlapping bearing observations. Both moving cameras are then
localized in `session_grid_A`; they are not independently rebased. The result
is accepted only after calibration-inlier, angular-residual, and coverage
gates pass. The report explicitly labels this as capture-local
self-calibration, not external ground truth.

The principal outputs are:

```text
final/dual-x5-four-mp4-cpu-v13/pairs/<pair-id>/tracking/
├── session_world_map.json
├── left_pose.csv
├── right_pose.csv
├── joint_trajectory.csv
└── report.json
```

`joint_trajectory.csv` has one shared H5 timestamp and one shared map ID per
row, followed by both left and right 6DoF poses. Direct bearing measurements
are marked `MEASURED`; gaps bounded by measurements are filled and retain a
numeric pose on every common-timeline frame. Gaps up to 0.25 seconds use a
timestamp-aligned, visual-endpoint-anchored accelerometer/gyro bridge when
available and are marked `IMU_ASSISTED`, otherwise they use visual interpolation
and SLERP and are marked `INTERPOLATED`. Longer gaps use the same bounded bridge
when safely available and are marked
`IMU_ASSISTED_UNTRUSTED`; otherwise they are marked
`INTERPOLATED_UNTRUSTED`. Leading or
trailing gaps use the nearest accepted pose and are marked `HELD_UNTRUSTED`.
`joint_has_pose` therefore describes numeric availability independently from
`joint_valid`, which remains false whenever either side is untrusted. The audit
video shows every pose and its confidence state instead of hiding it. The
report separately records numeric-pose, trusted, measured, long-gap, and held
coverage.

Published camera poses use child frame `hand_camera_flu_back_x`. Its `+X` is
the optical direction of the `back`/stream-0 video, `+Y` points left, and `+Z`
points up. PnP continues to operate in its internal stream-0 OpenCV frame; only
the serialized child basis is re-expressed, so the world-frame camera origin
is unchanged.

Sparse planar observations receive an additional confidence-aware temporal
gate. A pose supported by only two co-planar Tags carried entirely by LK flow
is rejected when it implies more than 1.5 m/s or 180 deg/s from the previous
accepted pose. Every visual solve also has a generous absolute ceiling of
3 m/s or 540 deg/s, including same-lens direct detections. After a rejection,
reacquisition requires at least five inlier Tags and consistency with the last
accepted pose; this prevents a wrong low-Tag planar branch from becoming the
new anchor merely because it persists. Weak visual attitudes that differ by
more than 15 degrees from calibrated gyro propagation are also rejected.

Each selected inlier also retains its calibrated `lens_stream` provenance. At
an unambiguous dominant-lens handoff, only the first candidate measurement is
rejected when it crosses the same 1.5 m/s or 180 deg/s limit; the observed-lens
state then advances so subsequent same-lens measurements are not rejected in a
chain. Slow handoffs and physically plausible same-lens direct measurements
remain accepted. The pose CSV records per-lens inlier counts, dominant lens,
measured temporal speeds, gyro residual/status, and the rejection reason for
auditing.

Generic four-MP4 input can still provide an external world map and initial
poses to the existing held-out joint pose-graph optimizer:

```json
{
  "tracking": {
    "enabled": true,
    "left_initial_pose_common_time": "bootstrap/left_pose.csv",
    "right_initial_pose_common_time": "bootstrap/right_pose.csv",
    "initial_world_map": "bootstrap/session_world_map.json",
    "start_common_s": 0,
    "alternations": 4
  }
}
```

No stitched RGB video is needed for either tracking path.

Render the four source views beside the synchronized shared-map 3D tracks:

```bash
.venv/bin/python -m tools.render_joint_four_mp4_trajectory \
  /data/session \
  /data/session/final/dual-x5-four-mp4-cpu-v13/pairs/<pair-id>/tracking \
  /data/session/processed/joint_trajectory_comparison.mp4 \
  --reframe-world-flu \
  --view-preset flu-front-above
```

The comparison view includes a metric world grid and XYZ axes, camera
frustums, measured/interpolated state, live XYZ and RPY values, linear and
angular speeds, and full-clip XYZ/RPY trend plots for both cameras.

For data explicitly re-expressed in the right-handed FLU world above (`+X`
toward the AprilGrid rear, `+Y` left, `+Z` up), use
`--view-preset flu-front-above`. This fixes a perspective camera on the
negative-X printed-front side and above both AprilGrids, draws the vertical Tag
plane, labels the world and camera FLU axes, and adds dashed camera-to-wall `X`
depth guides. `--reframe-world-flu` performs and audits the same conversion
used by the published CSV exporter; the view preset by itself remains
display-only and does not silently reinterpret input coordinates.

When only the hand-camera child frame is FLU and the parent remains the native
`tag_map`, use `--view-preset tag-map-front-above`. For the four-MP4 X5 rig,
hand-camera `+X` is the optical direction of the `back`/stream-0 lens, `+Y` is
left, and `+Z` is up. The corresponding source-camera basis bridge is
`X_hand -> +Z_source`, `Y_hand -> -X_source`, `Z_hand -> -Y_source`. This
preset views the native `Z_map=0` Tag wall from its front and physical `-Y_map`
upper side; it does not rotate the world positions.
