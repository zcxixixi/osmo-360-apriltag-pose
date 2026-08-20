# Motion-capture ground-truth evaluation

`evaluate_mocap_ground_truth.py` compares an `osmo_360_offline.py` pose CSV
with an OptiTrack/Vicon rigid-body trajectory. It reports:

- position ATE (mean, RMSE, median, p95, maximum);
- orientation error in degrees;
- translational and rotational RPE;
- endpoint drift and drift as a percentage of path length;
- visual-loss intervals and error on reacquisition;
- optional position/orientation drift during known stationary intervals.

## Mocap CSV

Export the rigid body with these column names:

```csv
timestamp,x,y,z,qx,qy,qz,qw
0.000,12.3,-45.6,901.2,0.0,0.0,0.0,1.0
```

`timestamp` is seconds and must be increasing. Position may be metres or
millimetres. Quaternion order is `x,y,z,w` and represents body-to-world
orientation. Remove any text/header rows added by Motive or Vicon before the
CSV header shown above.

## Rigid-body to camera transform

The mocap marker origin normally differs from the panoramic camera optical
centre. Measure or calibrate `T_body_camera` and save it as a JSON 4x4 matrix:

```json
[
  [1, 0, 0, 0.000],
  [0, 1, 0, 0.000],
  [0, 0, 1, 0.000],
  [0, 0, 0, 1.000]
]
```

The translation is in metres. It is the camera pose expressed in the mocap
rigid-body frame. An identity matrix is only valid when the mocap pose already
describes the camera optical frame.

## Run

```bash
uv run python evaluate_mocap_ground_truth.py \
  sessions/robot-rotation-0021-final/pose.csv \
  mocap/rotation-0021.csv \
  --mocap-unit mm \
  --body-to-camera mocap/T_body_camera.json \
  --static-interval 0:2 \
  --output-dir evaluations/rotation-0021
```

The evaluator estimates a small timestamp offset from translation and angular
speed. For a known hardware-synchronised offset, pass `--time-offset SECONDS`.

Outputs:

- `mocap_evaluation.json`: metrics and loss/recovery events;
- `mocap_errors.csv`: per-frame position and orientation errors;
- `mocap_evaluation.png`: aligned 3D trajectories and error plots.

Do not use Sim(3) scale alignment for the reported accuracy. AprilGrid has a
metric scale, so allowing scale would hide tag-size or calibration errors.
