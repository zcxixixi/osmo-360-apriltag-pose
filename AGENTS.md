# Repository instructions

Keep this file short and repository-specific. General coding discipline is
provided by the harness and must not be repeated here. Do not create additional
or case-variant agent instruction files.

## Guarded v50 baseline

Before changing localization, calibration, fusion, role mapping, world frames,
hardware extrinsics, smoothing, timeline export, 3-D rendering, or UMI export:

1. Read `DUAL_GRIPPER_V50_BASELINE.md` and
   `config/baselines/dual_gripper_v50_accepted_baseline.json`.
2. Run `./.venv/bin/python verify_dual_gripper_v50_baseline.py`.
3. After the change, run the focused check, the real-data path, the verifier,
   and `./.venv/bin/pytest -q`.

The accepted v15/v50 artifacts and `assets/gripper/` meshes are immutable.
Never overwrite them or change pinned hashes. New experiments use a new
version/output directory. If the verifier fails, stop and report the mismatch.

Do not:

- substitute raw IMU rotation for audited cross-pose attitude;
- trust interpolation gaps longer than 0.25 s;
- swap pose columns without swapping the complete rig binding;
- mix TCP coordinates with the `base_link` mesh origin;
- use screen-copy Tags, synthetic frames, or hidden contact constraints in
  accepted metric outputs.

## Active hardware and revisions

Until the user declares a revision:

- left: serial `95SXN9H0423SGG`, BaseTag ID2;
- right: serial `95SXNAD0425JCY`, BaseTag ID3;
- wall Tags: IDs 128-137, 200 mm outer size;
- BaseTags: 20 mm outer size;
- world frame: `tag_map`, physical up `[0, -1, 0]`.

Read the camera serial from OSV metadata; never infer hardware from filenames.
Any mount, geometry, Tag, or wall-layout change requires a new immutable
revision. New interfaces take a rig revision, verify its hashes and serial, and
must not silently fall back to legacy calibration.

Current new-gripper revisions:

- CAD: `config/rig_revisions/gripper_cad_v52_new_r1.json`;
- geometry: `config/rig_revisions/gripper_geometry_v52_new_r2.json`;
- rig: `config/rig_revisions/dual_gripper_v52_new_gripper_20260826_r2.json`;
- assets: `assets/gripper_v52_new_r1/`;
- marker layout: `config/rig_revisions/gripper_marker_layout_umi_iii_dxf_r2.json`.

The newer editable source package `/home/cenxi/Downloads/UMI-III.zip` is not a
renderable export. Do not claim it matches v52 meshes. Preserve
`/home/cenxi/Downloads/标定.DXF` as marker-layout authority; r1 misidentified
five auxiliary circles and is rejected.

The CAD update does not change `T_base_basetag`:
translation `[0.02625, 0, 0.0196]` m, RPY `[0, 0, 0]`. Jaw/contact geometry is
a separate revision. The pad STL uses millimetres; URDF meshes use metres.

## Single-gripper diagnostics

Active implementations:

- diagnostics: `render_gripper_force_angle_demo.py`,
  `render_single_gripper_motion_demo.py`, `render_single_gripper_webgl_demo.py`;
- review platform: `dual_gripper_3d/platform_server.mjs`,
  `dual_gripper_3d/platform.html`, `upload_visualization_bundle.py`;
- focused tests: `tests/test_gripper_force_angle_demo.py`,
  `tests/test_single_gripper_motion_demo.py`,
  `tests/test_single_gripper_webgl_demo.py`,
  `tests/test_visualization_platform.py`.

Current capture:
`/home/cenxi/Videos/umi-captures/20260827/single-0063-smoke100-v1/`,
source `CAM_20260827134904_0063_D.OSV`, left serial, BaseTag ID2. Use
`pose-world-v1-direct/`, `force-angle-v3/`, and the reviewed
`webgl-final-v5/` timeline/output. Earlier 0063 pose/force/WebGL variants are
rejected.

Latest X5 diagnostic:
`/home/cenxi/Videos/umi-captures/20260829/insta360-x5-114845-v1/`,
physical right serial `IAHEA2606KMURQ`, verified by CameraSDK 2.1.1
DeviceDiscovery (`Insta360 X5`, firmware `v1.7.8`) and corroborated by BaseTag
ID3 in 828/852 frames (ID2 in 0/852). Use angle revision
`config/rig_revisions/x5_jaw_angle_yellow_dots_20260829_r2.json`, rig
`config/rig_revisions/x5_right_basetag3_gridAB_20260829_r4.json`, force output
`force-angle-v15-sdk-identity-r4/`, timeline
`webgl-v12-sdk-identity-r4/`, and review bundle
`review-bundle-v5-sdk-identity-r4/`. The angle is defined by the three
highlighted yellow circular dots on each black jaw pad; do not substitute
yellow-body PCA or capture-percentile zeroing. Bilateral measurement is
preferred. A right-pad-only quadratic fallback is explicitly low-confidence
(blocked holdout MAE 0.82 deg, P95 2.03 deg) and never produces force. Gaps over
0.25 s remain N/A and hide the CAD jaw links. The visible Grid subset did not
yield a valid world pose, so this product is camera-local BaseTag3 diagnostic,
not `tag_map`.

Registered captures must use `./umi` with a file under `manifests/captures/`.
Do not invoke internal processing scripts or assemble timelines manually for a
registered capture. `umi inspect` verifies all hashes, `umi process` runs or
reuses the locked outputs, and `umi review` creates an immutable bundle with a
project-versioned renderer. The current X5 manifest is
`manifests/captures/x5-20260829-114845-iahea2606kmurq-sdk-r3.json`.

Fleet identity is stored in `config/devices/x5_inventory.json`. Use
`umi devices scan/register/assign/list`; never create one-off serial discovery
scripts or rerun video processing merely to register another X5. The udev rule
`config/udev/99-insta360-camera-sdk.rules` applies to every X5 on the host.

`contact_intensity` is capture-local pad deformation, not force in Newtons and
not cross-episode comparable. Preserve direct, recovered (maximum 0.25 s), and
unavailable observations separately. Keep per-pad deformation before combining.
Use the opening-conditioned lower envelope in `force-angle-v3`; do not restore
the rejected global MAD deadband.

For a training field, define one immutable scale per hardware revision. Never
normalize each episode independently.

## Single-gripper WebGL invariant

Use `dual_gripper_3d/single_gripper_scene.html`; do not modify the guarded
`dual_gripper_3d/scene.html` for single-gripper behavior.

The accepted view is `正对 Tag 墙` / `view-all`: complete Tag walls centred,
floor horizon visible, gripper viewed from the workspace side. Offline rendering
must keep timeline `default_view=human_corner`, pass
`--view-preset human_corner`, call `fitView(all)`, and disable the redundant
world inset.

Apply a display-only 180-degree roll around base-local `+X` to the rendered
gripper. Base-local `+Z` yaw is wrong. Preserve the source quaternion and never
write the display correction into metric `tag_map` poses.
For `camera_local_basetag` diagnostics, do not apply the world-view 180-degree
display roll. Use the BaseTag-bound source rotation and a BaseTag-normal
diagnostic camera view; never claim it is a world pose.

## Agent review platform

The platform is a fast visual check of processed results, not a processing
pipeline. It accepts `single_gripper_webgl_timeline.json` plus synchronized
`front-video.mp4`; it must not recompute or alter poses, angles, or contact
intensity. Agents upload with `upload_visualization_bundle.py` and consume its
JSON `view_url`. Capabilities are at `http://192.168.111.62:7865/api/capabilities`.

`render_single_gripper_webgl_demo.py` derives `capture_pair_id` from the source
OSV stem and actual FPS. Never hardcode a capture filename or ID.

## Pose, cache, and evidence invariants

- Serialize whether each pose is measured, recovered, predicted, smoothed, or
  rejected.
- Planar angular residual alone does not validate metric depth.
- Treat fisheye lenses as distinct calibrated optical centres.
- Rendering transforms are display-only.
- Every timeline declares frame, units, transform direction, and physical up.
- Store source/revision/calibration hashes and complete parameters in caches;
  reuse only on an exact identity match.
- Mark failed experiments `REJECTED` or remove them.
- Diagnostic video quality is not UMI training readiness.

Do not claim millimetre accuracy without independent metric ground truth. A
capture used for tuning is not a holdout.

## Insta360 SDK

Use the pinned revision
`config/sdk_revisions/insta360_linux_camera_2_1_1_media_3_1_1.json` and
`--insta-sdk-revision`; verify binary/library/model/platform hashes. Never scan
an unversioned SDK directory. X5 support remains unverified until tested with an
actual device and raw capture.
