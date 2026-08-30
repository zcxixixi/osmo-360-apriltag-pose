#!/usr/bin/env python3
"""Train and visualize a small real VLA policy on one audited UMI scene.

This is a single-instruction memorization demo, not language generalization or a
robot-deployable policy. The model consumes two real camera observations, the
UTF-8 task instruction, and current robot state, then predicts the next action.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from train_zarr_overfit_smoke import (
    CameraEncoder,
    images_to_tensor,
    load_dataset,
    rotation_errors_deg,
    set_seed,
    state_matrix,
    transition_indices,
)


FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def encode_instruction(text: str, max_bytes: int = 128) -> np.ndarray:
    encoded = text.encode("utf-8")[:max_bytes]
    tokens = np.zeros(max_bytes, dtype=np.int64)
    if encoded:
        tokens[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8).astype(np.int64) + 1
    return tokens


class SmallVlaPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.vision = CameraEncoder()
        self.token_embedding = nn.Embedding(257, 32, padding_idx=0)
        self.language = nn.GRU(32, 64, batch_first=True)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 96), nn.SiLU())
        self.action_head = nn.Sequential(
            nn.Linear(192 * 2 + 64 + 96, 384), nn.SiLU(),
            nn.Linear(384, 384), nn.SiLU(),
            nn.Linear(384, action_dim),
        )

    def forward(
        self,
        camera0: torch.Tensor,
        camera1: torch.Tensor,
        instruction: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.token_embedding(instruction)
        _, language = self.language(embedded)
        fused = torch.cat(
            (
                self.vision(camera0),
                self.vision(camera1),
                language[-1],
                self.state_encoder(state),
            ),
            dim=1,
        )
        return self.action_head(fused)


def action_matrix(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    actions = np.zeros_like(current, dtype=np.float32)
    for base in (0, 7):
        actions[:, base : base + 3] = target[:, base : base + 3] - current[:, base : base + 3]
        current_rotation = Rotation.from_rotvec(current[:, base + 3 : base + 6])
        target_rotation = Rotation.from_rotvec(target[:, base + 3 : base + 6])
        actions[:, base + 3 : base + 6] = (current_rotation.inv() * target_rotation).as_rotvec()
        actions[:, base + 6] = target[:, base + 6] - current[:, base + 6]
    return actions


def apply_actions(current: np.ndarray, actions: np.ndarray) -> np.ndarray:
    target = current.copy()
    for base in (0, 7):
        target[:, base : base + 3] += actions[:, base : base + 3]
        current_rotation = Rotation.from_rotvec(current[:, base + 3 : base + 6])
        delta_rotation = Rotation.from_rotvec(actions[:, base + 3 : base + 6])
        target[:, base + 3 : base + 6] = (current_rotation * delta_rotation).as_rotvec()
        target[:, base + 6] += actions[:, base + 6]
    return target


def metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    for robot, base in ((0, 0), (1, 7)):
        position = np.linalg.norm(predicted[:, base : base + 3] - truth[:, base : base + 3], axis=1) * 1000
        rotation = rotation_errors_deg(predicted, truth, base + 3)
        width = np.abs(predicted[:, base + 6] - truth[:, base + 6]) * 1000
        result[f"robot{robot}"] = {
            "position_mae_mm": float(position.mean()),
            "position_p95_mm": float(np.percentile(position, 95)),
            "rotation_mae_deg": float(rotation.mean()),
            "rotation_p95_deg": float(np.percentile(rotation, 95)),
            "gripper_width_mae_mm": float(width.mean()),
            "gripper_width_p95_mm": float(np.percentile(width, 95)),
        }
    return result


def render_demo(
    output: Path,
    arrays: dict[str, np.ndarray],
    current_indices: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    instruction: str,
    parameter_count: int,
    fps: float,
) -> None:
    width, height = 1280, 720
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not create {output}")
    xy = np.concatenate((truth[:, 0:2], truth[:, 7:9]), axis=0)
    low, high = xy.min(0), xy.max(0)
    span = np.maximum(high - low, 1e-4)
    colors = ((255, 180, 55), (90, 225, 85))
    chinese_font = ImageFont.truetype(str(FONT), 22)

    def point(value: np.ndarray) -> tuple[int, int]:
        normalized = (value[:2] - low) / span
        return int(80 + normalized[0] * 1120), int(700 - normalized[1] * 230)

    for row, source_index in enumerate(current_indices):
        canvas = np.full((height, width, 3), (12, 19, 27), dtype=np.uint8)
        cv2.putText(canvas, "SMALL VLA - REAL UMI SINGLE-SCENE DEMO", (24, 29), cv2.FONT_HERSHEY_SIMPLEX, .72, (238, 242, 246), 2)
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        draw.text((24, 42), f"语言指令：{instruction}", font=chinese_font, fill=(112, 220, 247))
        canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.putText(canvas, f"VISION x2 + LANGUAGE + STATE -> ACTION(t+1)   {parameter_count / 1e6:.2f}M params", (690, 29), cv2.FONT_HERSHEY_SIMPLEX, .48, (125, 220, 150), 1)
        for camera, x in ((0, 24), (1, 654)):
            rgb = arrays[f"camera{camera}_rgb"][source_index]
            bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (602, 300), interpolation=cv2.INTER_AREA)
            canvas[86:386, x : x + 602] = bgr
            cv2.putText(canvas, f"CAMERA {camera} / REAL OBSERVATION", (x, 82), cv2.FONT_HERSHEY_SIMPLEX, .52, colors[camera], 1)
        cv2.rectangle(canvas, (24, 410), (1256, 708), (47, 62, 75), 1)
        for robot, base in ((0, 0), (1, 7)):
            truth_points = np.asarray([point(value) for value in truth[: row + 1, base : base + 3]], np.int32)
            predicted_points = np.asarray([point(value) for value in predicted[: row + 1, base : base + 3]], np.int32)
            if len(truth_points) > 1:
                cv2.polylines(canvas, [truth_points], False, colors[robot], 3, cv2.LINE_AA)
                cv2.polylines(canvas, [predicted_points], False, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(truth_points[-1]), 6, colors[robot], -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(predicted_points[-1]), 5, (250, 250, 250), 1, cv2.LINE_AA)
        frame_truth = truth[row : row + 1]
        frame_prediction = predicted[row : row + 1]
        frame_metrics = metrics(frame_prediction, frame_truth)
        cv2.putText(canvas, f"TRANSITION {row + 1}/{len(current_indices)}   thick=color: truth   thin=white: VLA prediction", (40, 440), cv2.FONT_HERSHEY_SIMPLEX, .56, (220, 226, 232), 1)
        cv2.putText(canvas, f"R0 error {frame_metrics['robot0']['position_mae_mm']:.2f} mm / {frame_metrics['robot0']['rotation_mae_deg']:.2f} deg", (40, 470), cv2.FONT_HERSHEY_SIMPLEX, .52, colors[0], 1)
        cv2.putText(canvas, f"R1 error {frame_metrics['robot1']['position_mae_mm']:.2f} mm / {frame_metrics['robot1']['rotation_mae_deg']:.2f} deg", (650, 470), cv2.FONT_HERSHEY_SIMPLEX, .52, colors[1], 1)
        cv2.putText(canvas, "ONE INSTRUCTION - LANGUAGE EFFECT / GENERALIZATION NOT IDENTIFIABLE", (760, 438), cv2.FONT_HERSHEY_SIMPLEX, .40, (105, 120, 132), 1)
        writer.write(canvas)
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = (args.metadata or args.dataset.parent / "episode_metadata.json").resolve(strict=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    instruction = str(metadata["task"]["instruction"])
    frequency_hz = float(metadata["frequency_hz"])
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VLA demo")
    device = torch.device("cuda")

    arrays = load_dataset(args.dataset)
    all_states = state_matrix(arrays)
    current_indices, target_indices, episodes = transition_indices(arrays["episode_ends"])
    if len(np.unique(episodes)) != 1:
        raise RuntimeError("this demo requires exactly one scene/episode")
    current = all_states[current_indices]
    truth = all_states[target_indices]
    actions = action_matrix(current, truth)
    state_mean, state_std = current.mean(0), current.std(0).clip(1e-5)
    action_mean, action_std = actions.mean(0), actions.std(0).clip(1e-6)

    camera0 = images_to_tensor(arrays["camera0_rgb"][current_indices], device)
    camera1 = images_to_tensor(arrays["camera1_rgb"][current_indices], device)
    state_tensor = torch.from_numpy((current - state_mean) / state_std).to(device)
    action_tensor = torch.from_numpy((actions - action_mean) / action_std).to(device)
    instruction_tokens = torch.from_numpy(encode_instruction(instruction)).to(device)
    language_tensor = instruction_tokens.unsqueeze(0).expand(len(current_indices), -1)

    model = SmallVlaPolicy(current.shape[1], actions.shape[1]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    losses = []
    model.train()
    for _ in range(args.steps):
        indices = torch.randint(len(current_indices), (min(args.batch_size, len(current_indices)),), device=device)
        prediction = model(camera0[indices], camera1[indices], language_tensor[indices], state_tensor[indices])
        loss = F.mse_loss(prediction, action_tensor[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    model.eval()
    with torch.inference_mode():
        predicted_normalized = model(camera0, camera1, language_tensor, state_tensor).cpu().numpy()
    predicted_actions = predicted_normalized * action_std + action_mean
    predicted_next = apply_actions(current, predicted_actions)
    result_metrics = metrics(predicted_next, truth)

    checkpoint_path = args.output_dir / "small_vla_policy.pt"
    checkpoint = {
        "model": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "instruction": instruction,
        "parameter_count": parameter_count,
    }
    torch.save(checkpoint, checkpoint_path)
    restored_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    restored = SmallVlaPolicy(current.shape[1], actions.shape[1]).to(device)
    restored.load_state_dict(restored_checkpoint["model"], strict=True)
    restored.eval()
    with torch.inference_mode():
        restored_prediction = restored(camera0, camera1, language_tensor, state_tensor)
    reload_delta = float(torch.max(torch.abs(restored_prediction - torch.from_numpy(predicted_normalized).to(device))).cpu())
    if reload_delta > 1e-7:
        raise RuntimeError(f"checkpoint reload changed predictions by {reload_delta}")

    empty_language = torch.from_numpy(encode_instruction("")).to(device).unsqueeze(0).expand(len(current_indices), -1)
    with torch.inference_mode():
        empty_prediction = restored(camera0, camera1, empty_language, state_tensor).cpu().numpy()
    language_ablation_delta = float(np.mean(np.abs(empty_prediction - predicted_normalized)))

    report = {
        "demo_type": "small_vla_architecture_real_umi_single_scene_memorization",
        "dataset": str(args.dataset.resolve()),
        "metadata": str(metadata_path),
        "instruction": instruction,
        "inputs": ["camera0_rgb", "camera1_rgb", "utf8_language_instruction", "current_robot_state"],
        "output": "next_frame_dual_robot_action_delta",
        "frames": int(len(all_states)),
        "transitions": int(len(current_indices)),
        "episodes": 1,
        "frequency_hz": frequency_hz,
        "model_parameters": parameter_count,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "device": torch.cuda.get_device_name(0),
        "steps": len(losses),
        "final_batch_mse_normalized": losses[-1],
        "full_dataset_mse_normalized": float(np.mean((predicted_normalized - ((actions - action_mean) / action_std)) ** 2)),
        "metrics": result_metrics,
        "checkpoint_reload": {"status": "PASS", "prediction_max_abs_delta": reload_delta},
        "language_ablation_mean_normalized_action_delta": language_ablation_delta,
        "language_effect_status": (
            "NOT_OBSERVED_ONE_CONSTANT_INSTRUCTION"
            if language_ablation_delta < 1e-5
            else "OBSERVED_NOT_SEMANTICALLY_VALIDATED"
        ),
        "limitations": [
            "one real scene and one constant instruction; language effect and generalization are not identifiable",
            "training-set next-action prediction only; not a closed-loop robot evaluation",
            "metrics are memorization errors, not physical ground-truth accuracy",
        ],
    }
    restored_checkpoint["report"] = report
    torch.save(restored_checkpoint, checkpoint_path)
    (args.output_dir / "small_vla_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (args.output_dir / "small_vla_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source_frame", "target_frame", *[f"truth_{index}" for index in range(14)], *[f"pred_{index}" for index in range(14)]))
        for source, target, actual, estimate in zip(current_indices, target_indices, truth, predicted_next):
            writer.writerow((int(source), int(target), *actual.tolist(), *estimate.tolist()))

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(np.maximum(losses, 1e-12));axes[0].set_yscale("log");axes[0].set_title("Small VLA training loss - single scene")
    axes[0].set_xlabel("optimizer step");axes[0].set_ylabel("normalized action MSE")
    for robot, color in ((0, "#38bdf8"), (1, "#34d399")):
        base = robot * 7
        axes[1].plot(truth[:, base], color=color, linewidth=2, label=f"robot{robot} truth X")
        axes[1].plot(predicted_next[:, base], color=color, linestyle="--", linewidth=1, label=f"robot{robot} VLA predicted X")
    axes[1].set_title("Next-action prediction vs truth - one constant instruction")
    axes[1].set_xlabel("transition");axes[1].set_ylabel("tag_map X (m)");axes[1].legend(ncol=2)
    figure.savefig(args.output_dir / "small_vla_training.png", dpi=160);plt.close(figure)
    render_demo(args.output_dir / "small_vla_demo.mp4", arrays, current_indices, truth, predicted_next, instruction, parameter_count, frequency_hz)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
