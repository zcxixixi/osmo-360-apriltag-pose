# Dual-gripper axis/frame fix v3

## Right-gripper heading correction

The v2 right mount used Tag quarter-turn `2`, which kept the gripper horizontal but
reversed its in-plane fingertip direction. It is now changed to quarter-turn `0`, IPPE
branch `0`. This is a physical `camera→base_link` correction, not a renderer rotation.
It reverses the right gripper `+X/+Y` directions by 180 degrees while preserving its
already-correct plane normal (`+Z`). The selection is frozen in
`config/gripper_mount_selections_axisfix_v3.json`.

The new U-disk diagnostic A-point gap is `125.1 mm` (previous honest v2 value was
`231.5 mm`). It remains a failed common-space validation, but is not fitted or hidden.
The v3 interactive diagnostic is served on port 7867.

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

- `demo-output/upan-20260823/hardware-axisfix-v3.json`
- `demo-output/upan-20260823/dataset-axisfix-v3/`
- `demo-output/upan-20260823/upan_axisfix_v3_diagnostic_timeline.json`

Earphone:

- `/home/cenxi/Videos/umi-captures/20260823/regression-earphone/dataset-axisfix-v3/`
- `/home/cenxi/Videos/umi-captures/20260823/regression-earphone/earphone_axisfix_v3_diagnostic_timeline.json`

The v3 U-disk timeline reports the independently replayed DROP/PICK A-point gap as
`125.1 mm`, not the overfit `0.38 mm`. This is intentionally still a failed validation and
proves that the remaining shared-space error is no longer hidden by `shared-a`.

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
