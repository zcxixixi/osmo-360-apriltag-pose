from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int
    tag_size: float
    spacing_ratio: float
    first_id: int = 0
    id_order: str = "column-major"

    @property
    def pitch(self) -> float:
        return self.tag_size * (1.0 + self.spacing_ratio)

    @property
    def width(self) -> float:
        return (self.cols - 1) * self.pitch + self.tag_size

    @property
    def height(self) -> float:
        return (self.rows - 1) * self.pitch + self.tag_size

    def corners(self, tag_id: int) -> np.ndarray | None:
        index = tag_id - self.first_id
        if index < 0 or index >= self.rows * self.cols:
            return None
        if self.id_order == "column-major":
            col, row = divmod(index, self.rows)
        elif self.id_order == "row-major":
            row, col = divmod(index, self.cols)
        else:
            raise ValueError(f"unsupported AprilGrid ID order: {self.id_order}")
        x0 = col * self.pitch - self.width / 2.0
        y0 = self.height / 2.0 - row * self.pitch
        size = self.tag_size
        return np.asarray([
            [x0, y0, 0.0],
            [x0 + size, y0, 0.0],
            [x0 + size, y0 - size, 0.0],
            [x0, y0 - size, 0.0],
        ], dtype=np.float32)

    def center(self, tag_id: int) -> np.ndarray | None:
        corners = self.corners(tag_id)
        return None if corners is None else corners.mean(axis=0)


def rotation_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    if sy >= 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))
