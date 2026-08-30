# Dual-gripper v50 accepted baseline

> Status: **accepted visual diagnostic baseline, not external ground truth**.
>
> Machine lock: `config/baselines/dual_gripper_v50_src_accepted_baseline.json`
>
> Historical predecessor: `config/baselines/dual_gripper_v50_accepted_baseline.json`
>
> Verifier: `./umi verify`

## Why this baseline exists

The calibration-capture v15 trajectory looked coherent, but later asymmetric-fusion
outputs became progressively worse. The main regression was not a renderer camera
problem: the fusion code discarded an already audited cross-pose world rotation and
replaced it with a raw IMU delta rebased from the first pose. That shortcut omitted
the calibrated IMU/body/camera transform chain. A rotation error was then amplified
at the TCP by the 135.6 mm base-to-TCP lever arm, corrupting both apparent angle and
TCP trajectory.

The old fusion also interpolated across observation gaps as long as 2.002 seconds
while rendering those samples as trusted. The v50 fix restores the useful v15
principle: keep the constrained visual/world rotation, permit only bounded SLERP,
and fail closed across long gaps.

## Non-negotiable semantics

1. The world frame is `tag_map`; physical up is `[0, -1, 0]`.
2. The frozen world-map file and compiled hashes are recorded in the machine lock.
3. Physical left is serial `95SXN9H0423SGG`, BaseTag ID2.
4. Physical right is serial `95SXNAD0425JCY`, BaseTag ID3.
5. In the calibration capture only, the source folders were worn on opposite hands:
   physical left is `source_right`, and physical right is `source_left`.
6. In the claw-to-claw action capture, `left/action_stream1.mp4` and
   `right/action_stream1.mp4` already have physical left/right meaning.
7. A role swap must move the complete serialized rig binding: video, camera serial,
   BaseTag ID, camera calibration, gripper angle, and hardware extrinsic. Never swap
   only pose columns.
8. Weak-side attitude is the audited cross-pose world rotation. Never replace it
   with `first_pose * raw_imu_relative` unless the complete calibrated transform
   chain is represented and independently regression-tested.
9. Maximum interpolation gap is 0.25 s. A longer gap is
   `INTERPOLATED_UNTRUSTED`, invisible, and breaks the trail.
10. Timeline positions are TCP positions. The STL root is `base_link`; rendering
    must apply `p_world_base = p_world_tcp - R_world_base * [0.1356, 0, 0.0101]`.
11. Screen copies of duplicate Tag IDs, synthetic frames, and contact constraints
    are forbidden in this accepted baseline.

## Accepted evidence

### Calibration capture

Baseline:

- `dual_gripper_3d_200mm_joint_v15_measured_angles_role_corrected_diagnostic_timeline.json`

Accepted v50 comparison:

- left position P95 after constant rigid alignment: 26.61 mm
- right trusted position P95: 19.66 mm
- left orientation P95 after constant frame alignment: 5.32 deg
- right orientation P95: 4.79 deg
- orientation-motion correlation: left 0.901, right 0.869

The v15 and v50 calibration products remain diagnostic; v15 itself was holdout-failed.
Their purpose here is regression detection, not a claim of metric ground truth.

### Claw-to-claw action

Accepted video:

- `/home/cenxi/Videos/umi-captures/20260825/dual-0057-0030-new/dual_gripper_claw_to_claw_action_v50_fixed.mp4`

Accepted timeline properties:

- 13.5 s at 30 fps, 406 timeline samples
- weak tracked ratio 1.0
- no untrusted long-gap frames
- minimum TCP separation 14.98 mm at 10.90 s
- median TCP separation 31.45 mm
- left/right orientation-step P95: 5.15 / 1.71 deg

These values are task-specific regression signals. A future version does not need to
match every decimal, but it must remain inside the gates in the machine lock and be
visually compared at approach, contact, and departure.

## Before changing the pipeline

Run:

```bash
./umi verify
./.venv/bin/pytest -q
```

Then make changes into a **new output directory and version number**. Do not overwrite
v15 or v50 artifacts. After the change:

1. rerun all tests;
2. rerun the verifier to prove the frozen baseline was not mutated;
3. generate a new calibration comparison against v15/v50;
4. generate the full 13.5 s claw-to-claw video with synchronized raw-camera insets;
5. compare approach/contact/departure visually;
6. create a new baseline lock such as v51 only after explicit human acceptance.

A report field named `VERIFIED` is insufficient by itself. The old v49 report passed
position/coverage gates while its right orientation P95 had regressed to about 58
degrees. Orientation regression and role-binding checks are mandatory.

## Reproducing accepted action fusion

```bash
P=/home/cenxi/Videos/umi-captures/20260825/dual-0057-0030-new

./.venv/bin/python -m osmo360.localization.fuse_asymmetric_gripper_world_pose \
  --strong-camera-csv "$P/world-pose-v45-multicapture-map/right_camera_pose.csv" \
  --weak-cross-base-csv "$P/world-pose-v45-multicapture-map/left_base_pose.csv" \
  --weak-instance-cache "$P/right/action_apriltag_factory_instance_id2_v1.npz" \
  --weak-tag-id 2 \
  --hardware config/hardware_20260825_serial_bound_v1.json \
  --world-map /home/cenxi/Videos/umi-captures/20260825/world-map-room-corner-10tag-200mm-multicapture-v1.json \
  --weak-role left --strong-role right \
  --maximum-interpolation-gap-s 0.25 \
  --output-dir "$P/fused-world-v50-v15-regression-fixed"

./.venv/bin/python -m osmo360.visualization.render_fused_world_audit \
  --fusion-dir "$P/fused-world-v50-v15-regression-fixed" \
  --template-timeline "$P/dual_gripper_training_ready_v49_timeline.json" \
  --left-video "$P/left/action_stream1.mp4" \
  --right-video "$P/right/action_stream1.mp4" \
  --output "$P/dual_gripper_claw_to_claw_action_v50_fixed.mp4" \
  --fps 30 --duration 13.5 --default-view operator --view-roll-deg -90 \
  --sync-offset-s 0 --sync-correlation 0.95296
```

The lock file contains SHA-256 values for code, inputs, maps, timelines, reports, and
accepted outputs. If a hash changes, investigate; do not casually update the hash to
make the verifier pass.
