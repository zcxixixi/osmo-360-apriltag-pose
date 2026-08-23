#!/usr/bin/env python3
"""Small behavior-cloning overfit test for an exported dual-gripper Zarr dataset.

This is deliberately a memorization/smoke test, not a generalization benchmark.
Transitions are built inside each episode so an action never targets the first
frame of the following episode.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from scipy.spatial.transform import Rotation


STATE_KEYS = (
    "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width",
    "robot1_eef_pos", "robot1_eef_rot_axis_angle", "robot1_gripper_width",
)


class CameraEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 192), nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class DualCameraPolicy(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.encoder = CameraEncoder()
        self.head = nn.Sequential(
            nn.Linear(192 * 2 + state_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, state_dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.encoder(left), self.encoder(right), state), dim=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    store = zarr.ZipStore(str(path), mode="r")
    root = zarr.group(store=store)
    arrays = {key: np.asarray(root[f"data/{key}"][:]) for key in ("camera0_rgb", "camera1_rgb", *STATE_KEYS)}
    arrays["episode_ends"] = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    store.close()
    return arrays


def state_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(arrays[key]).reshape(len(arrays[key]), -1) for key in STATE_KEYS], axis=1).astype(np.float32)


def transition_indices(episode_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current: list[int] = []
    target: list[int] = []
    episodes: list[int] = []
    start = 0
    for episode, end in enumerate(episode_ends.tolist()):
        current.extend(range(start, end - 1))
        target.extend(range(start + 1, end))
        episodes.extend([episode] * max(0, end - start - 1))
        start = end
    return np.asarray(current), np.asarray(target), np.asarray(episodes)


def images_to_tensor(images: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    tensor = F.interpolate(tensor, size=(56, 56), mode="area")
    return tensor.to(device)


def rotation_errors_deg(prediction: np.ndarray, truth: np.ndarray, offset: int) -> np.ndarray:
    predicted = Rotation.from_rotvec(prediction[:, offset:offset + 3])
    actual = Rotation.from_rotvec(truth[:, offset:offset + 3])
    return np.degrees((predicted.inv() * actual).magnitude())


def render_prediction_video(
    output: Path, arrays: dict[str, np.ndarray], current: np.ndarray, episodes: np.ndarray,
    truth: np.ndarray, predicted: np.ndarray, fps: float = 20.0,
) -> None:
    width, height = 1280, 720
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output}")
    xy = np.concatenate((truth[:, 0:2], truth[:, 7:9]), axis=0)
    low, high = xy.min(0), xy.max(0)
    span = np.maximum(high - low, 1e-4)

    def point(value: np.ndarray) -> tuple[int, int]:
        normalized = (value[:2] - low) / span
        return int(80 + normalized[0] * 1120), int(690 - normalized[1] * 260)

    colors = ((255, 180, 55), (90, 225, 85))
    for row, source_index in enumerate(current):
        canvas = np.full((height, width, 3), (13, 20, 28), dtype=np.uint8)
        for camera, x in ((0, 25), (1, 655)):
            rgb = arrays[f"camera{camera}_rgb"][source_index]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = cv2.resize(bgr, (600, 338), interpolation=cv2.INTER_AREA)
            canvas[55:393, x:x + 600] = bgr
            cv2.putText(canvas, f"CAMERA {camera}", (x, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, colors[camera], 2)
        cv2.putText(canvas, "OVERFIT SMOKE TEST - NOT GENERALIZATION", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, .72, (230, 230, 230), 2)
        cv2.rectangle(canvas, (25, 410), (1255, 705), (45, 60, 72), 1)
        for robot, base in ((0, 0), (1, 7)):
            for episode in np.unique(episodes[:row + 1]):
                mask = np.flatnonzero(episodes[:row + 1] == episode)
                truth_segment = np.asarray([point(truth[index, base:base + 3]) for index in mask], np.int32)
                predicted_segment = np.asarray([point(predicted[index, base:base + 3]) for index in mask], np.int32)
                if len(truth_segment) > 1:
                    cv2.polylines(canvas, [truth_segment], False, colors[robot], 3, cv2.LINE_AA)
                    cv2.polylines(canvas, [predicted_segment], False, tuple(int(c * .55) for c in colors[robot]), 1, cv2.LINE_AA)
            truth_points = np.asarray([point(value) for value in truth[:row + 1, base:base + 3]], np.int32)
            predicted_points = np.asarray([point(value) for value in predicted[:row + 1, base:base + 3]], np.int32)
            cv2.circle(canvas, tuple(truth_points[-1]), 6, colors[robot], -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(predicted_points[-1]), 5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"EPISODE {episodes[row] + 1}  TRANSITION {row + 1}/{len(current)}", (40, 438), cv2.FONT_HERSHEY_SIMPLEX, .62, (220, 225, 230), 2)
        cv2.putText(canvas, "thick = truth   thin/white marker = prediction", (760, 438), cv2.FONT_HERSHEY_SIMPLEX, .52, (170, 180, 190), 1)
        writer.write(canvas)
    writer.release()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this smoke test")
    device = torch.device("cuda")

    arrays = load_dataset(args.dataset)
    all_states = state_matrix(arrays)
    current, target, episodes = transition_indices(arrays["episode_ends"])
    states = all_states[current]
    truth = all_states[target]
    state_mean, state_std = states.mean(0), states.std(0).clip(1e-5)
    target_mean, target_std = truth.mean(0), truth.std(0).clip(1e-5)

    left = images_to_tensor(arrays["camera0_rgb"][current], device)
    right = images_to_tensor(arrays["camera1_rgb"][current], device)
    state_t = torch.from_numpy((states - state_mean) / state_std).to(device)
    truth_t = torch.from_numpy((truth - target_mean) / target_std).to(device)

    model = DualCameraPolicy(states.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    losses: list[float] = []
    count = len(current)
    model.train()
    for step in range(args.steps):
        indices = torch.randint(count, (min(args.batch_size, count),), device=device)
        prediction = model(left[indices], right[indices], state_t[indices])
        loss = F.mse_loss(prediction, truth_t[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step and step % 500 == 0 and np.mean(losses[-100:]) < 1e-5:
            break

    model.eval()
    with torch.inference_mode():
        predicted_norm = model(left, right, state_t).cpu().numpy()
    predicted = predicted_norm * target_std + target_mean

    metrics: dict[str, dict[str, float]] = {}
    for robot, base in ((0, 0), (1, 7)):
        position_mm = np.linalg.norm(predicted[:, base:base + 3] - truth[:, base:base + 3], axis=1) * 1000.0
        rotation_deg = rotation_errors_deg(predicted, truth, base + 3)
        gripper_mm = np.abs(predicted[:, base + 6] - truth[:, base + 6]) * 1000.0
        metrics[f"robot{robot}"] = {
            "position_mae_mm": float(position_mm.mean()), "position_p95_mm": float(np.percentile(position_mm, 95)),
            "rotation_mae_deg": float(rotation_deg.mean()), "rotation_p95_deg": float(np.percentile(rotation_deg, 95)),
            "gripper_width_mae_mm": float(gripper_mm.mean()), "gripper_width_p95_mm": float(np.percentile(gripper_mm, 95)),
        }

    report = {
        "test_type": "memorization_overfit_smoke_test_not_generalization",
        "dataset": str(args.dataset.resolve()), "frames": int(len(all_states)),
        "transitions": int(count), "episodes": int(len(arrays["episode_ends"])),
        "episode_ends": arrays["episode_ends"].tolist(), "cross_episode_transitions": 0,
        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "steps_run": len(losses), "final_batch_mse_normalized": losses[-1],
        "full_dataset_mse_normalized": float(np.mean((predicted_norm - ((truth - target_mean) / target_std)) ** 2)),
        "metrics": metrics,
    }
    (args.output_dir / "overfit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    torch.save({"model": model.state_dict(), "state_mean": state_mean, "state_std": state_std,
                "target_mean": target_mean, "target_std": target_std, "report": report}, args.output_dir / "overfit_policy.pt")

    with (args.output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("episode", "source_frame", "target_frame", *[f"truth_{i}" for i in range(14)], *[f"pred_{i}" for i in range(14)]))
        for episode, source_index, target_index, actual, estimate in zip(episodes, current, target, truth, predicted):
            writer.writerow((episode, source_index, target_index, *actual.tolist(), *estimate.tolist()))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(np.maximum(losses, 1e-12)); axes[0].set_yscale("log"); axes[0].set_title("Overfit loss (not generalization)")
    axes[0].set_xlabel("optimizer step"); axes[0].set_ylabel("normalized MSE")
    for robot, color in ((0, "#38bdf8"), (1, "#34d399")):
        base = robot * 7
        axes[1].plot(truth[:, base], color=color, linewidth=2, label=f"robot{robot} truth X")
        axes[1].plot(predicted[:, base], color=color, linestyle="--", linewidth=1, label=f"robot{robot} predicted X")
    for end in np.cumsum(np.diff(np.r_[0, arrays["episode_ends"]]) - 1)[:-1]:
        axes[1].axvline(end, color="gray", alpha=.4)
    axes[1].set_title("Next-frame prediction vs truth"); axes[1].set_xlabel("within-episode transition"); axes[1].set_ylabel("relative X (m)")
    axes[1].legend(ncol=2)
    fig.savefig(args.output_dir / "overfit_training_audit.png", dpi=160)
    plt.close(fig)
    render_prediction_video(args.output_dir / "overfit_prediction_audit.mp4", arrays, current, episodes, truth, predicted)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
