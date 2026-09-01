# Four-MP4 CPU pipeline v2

`dual-x5-four-mp4-cpu-v2` accepts the four independent raw fisheye MP4 streams
produced by two X5 cameras. It does not import INSV and does not invoke the
Insta360 stitching SDK. The official panorama is therefore no longer a
localization prerequisite.

This is a new output revision. It does not overwrite or reinterpret the v50
accepted diagnostic baseline.

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

The current example H5 contains null camera intrinsics and identity placeholder
rig extrinsics. The pipeline therefore uses the serial-bound X5 factory lens
records in `config/devices/x5_factory_lens_offsets.json` for raw-pixel bearings,
but correctly stops at `OBSERVATIONS_READY` instead of inventing a calibrated
dual-camera trajectory.

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

## Execution and resource limits

Run the same one-argument entry point:

```bash
./run_pipeline.sh /absolute/path/to/dataset-root
```

The generic conservative profile defaults to one detector process and two
OpenCV/decoder threads. The InstaUMI fast profile runs four lens processes with
four threads each (bounded at 16 active CPU threads), uses about 1 GiB for four
resident decoders, and requires neither CUDA nor panorama stitching.

Deployment overrides:

```bash
OSMO_CPU_WORKERS=4 \
OSMO_THREADS_PER_WORKER=2 \
OSMO_TRAJECTORY_FPS=15 \
OSMO_DECODE_FPS=15 \
OSMO_CACHE_CHUNK_SECONDS=120 \
OSMO_PIPELINE_CACHE=/fast-local-disk/osmo-cache \
./run_pipeline.sh /data/session
```

`OSMO_CPU_WORKERS` is capped at 4 and `OSMO_THREADS_PER_WORKER` at 8. For a
shared CPU server, `1 x 2` is the conservative setting; the tested 9950X speed
setting is `4 x 4`. `OSMO_TRAJECTORY_FPS` controls measured corner samples
written to the cache. H5 timestamps remain 59.94 Hz even though images are
retrieved for detection/tracking at 15 Hz.

On the supplied four-stream, 1920×1920, 59.94 FPS, 10-second dataset, the first
uncached observation pass measured 4.75 seconds on a Ryzen 9 9950X with the
`4 x 4` fast profile. Four-stream decode alone measured 2.89 seconds, so this
is close to the software HEVC decode floor. A cached repeat is about one second.
These measurements cover the complete H5 ingest, hashing, four-stream Tag
observation caches, and merging; they do not claim trajectory completion while
the H5 rig calibration remains a placeholder.

## Resume and cache identity

Every lens is split into independently committed chunks (120 seconds by
default). A completed chunk is reused only when all of the following still
match:

- source MP4 SHA-256;
- lens stream and factory offset;
- synchronized clock mapping;
- trajectory output stride and the complete temporal-search signature;
- decoded frame range and CPU thread setting.

The source hash is computed once and reused while file size and nanosecond
mtime remain unchanged. Outputs and JSON sidecars are atomically renamed, so a
killed process leaves the last incomplete chunk invalid and reruns only that
chunk.

The InstaUMI fast detector obtains the HEVC luma plane directly without an RGB
conversion. Half-resolution pyramidal LK updates known corners at 15 Hz;
forward/backward validation and merged local ROI decoding run at 5 Hz and 2 Hz
respectively. A 0.35-scale grayscale frame is searched globally at 2 Hz to
discover new Tags. The 11 tangent-view rectifications run at most 0.5 Hz and
only while BaseTag 2/3 or sufficient wall support is missing. The sidecar
records every flow, redetection, scout, fallback, and rejection count.

Persistent cache defaults to:

```text
dataset-root/.osmo-cache/<dataset-name>/dual-x5-four-mp4-cpu-v2/<pair-id>/
```

Set `OSMO_PIPELINE_CACHE` to a server-local SSD if the dataset itself is on a
NAS.

## Tracking stage

The four per-lens caches are merged into one calibrated dual-fisheye bearing
cache per physical camera. The existing cached joint pose-graph optimizer then
uses those two caches; it does not decode or stitch video.

Until a capture-specific session world map and common-time initial pose files
exist, the worker ends successfully with `OBSERVATIONS_READY`. It never labels
an uncalibrated trajectory as complete. To enable the existing optimizer, add:

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

These three capture-specific bootstrap products will be derived directly from
the new data after its real format and calibration visibility are inspected.
No stitched RGB video is needed for the cached joint optimization.
