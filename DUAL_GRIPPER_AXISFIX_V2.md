# Dual-gripper axis/frame fix v2

## What is fixed in code

- The PanoForge/DJI axis bridge is a proper SO(3) rotation (`det=+1`).
- Hardware transforms are typed and composed only as
  `T_common_tcp = T_common_camera @ T_camera_base @ T_base_tcp`.
- Common-frame export rejects missing physical chain fields and mount revisions containing
  `flat`, `table`, or `shared-a` display/task patches.
- The capture-calibrated nine-Tag map is now honestly named `tag_map`, not `room_world`.
- The map declares physical up as `-Y`; renderer camera, grid, provisional table, labels,
  TCP markers, and TCP `+X` arrows all consume this metadata instead of hard-coded axes.
- Web/video replay consumes the same TCP arrays as UMI. The renderer subtracts the CAD
  base-to-TCP translation exactly once only to place the base-link STL.
- An uncalibrated table plane and provisional map block training Zarr output.

## Diagnostic artifacts

U disk:

- `demo-output/upan-20260823/hardware-axisfix-v2.json`
- `demo-output/upan-20260823/dataset-axisfix-v2/`
- `demo-output/upan-20260823/upan_axisfix_v2_diagnostic_timeline.json`

Earphone:

- `/home/cenxi/Videos/umi-captures/20260823/regression-earphone/dataset-axisfix-v2/`
- `/home/cenxi/Videos/umi-captures/20260823/regression-earphone/earphone_axisfix_v2_diagnostic_timeline.json`

The v2 U-disk timeline reports the independently replayed DROP/PICK A-point gap as
`231.5 mm`, not the overfit `0.38 mm`. This is intentionally a failed validation and proves
that the remaining shared-space error is no longer hidden by `shared-a`.

At the first diagnostic frame, both gripper-plane normals agree with physical up to about
`14 degrees` or better (absolute dot product `0.95+`), so the former horizontal-vs-vertical
renderer failure is removed.

## Remaining physical calibration gate

The project still has no measured rigid transform from the capture Tag map to a frozen
Z-up room frame, no calibrated tabletop plane, and no independent multi-anchor validation.
A least-squares attempt to align the provisional capture map to the tape-measured room map
has about `208 mm` corner RMSE, so those maps must not be bridged automatically.

Do not mark `camera_to_tcp_verified=true`, do not produce training Zarr, and do not claim
A-point accuracy until a simultaneous two-wall calibration plus at least three distributed,
non-collinear handoff points (with held-out validation points) passes.

## Validation

```text
uv run pytest -q
97 passed
```

A one-frame headless Three.js/FFmpeg smoke render also completes successfully.
