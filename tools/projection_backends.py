"""CPU and optional CUDA projection backends for equirectangular frames."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import py360convert
from py360convert.utils import EquirecSampler


@dataclass(frozen=True)
class ProjectionRequest:
    yaw: float
    pitch: float
    fov: float
    size: int
    roll: float = 0.0


class ProjectionBackend(Protocol):
    name: str

    def project_many(
        self, pano: np.ndarray, requests: Sequence[ProjectionRequest]
    ) -> list[np.ndarray]: ...


class CpuProjectionBackend:
    name = "cpu"

    def project_many(
        self, pano: np.ndarray, requests: Sequence[ProjectionRequest]
    ) -> list[np.ndarray]:
        return [
            py360convert.e2p(
                pano,
                fov_deg=request.fov,
                u_deg=request.yaw,
                v_deg=request.pitch,
                out_hw=(request.size, request.size),
                in_rot_deg=request.roll,
                mode="bilinear",
            )
            for request in requests
        ]


class CudaProjectionBackend:
    """Batch panorama projections with CuPy, keeping each frame on the GPU once."""

    name = "cuda"

    def __init__(self) -> None:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import map_coordinates
        except ImportError as exc:
            raise RuntimeError(
                "CUDA projection requires cupy-cuda13x; install the gpu extra"
            ) from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy did not find a CUDA device")
        # Compile a tiny kernel now so missing toolkit components fail at startup.
        cp.arange(1, dtype=cp.float32).sum().get()
        self.cp = cp
        self.map_coordinates = map_coordinates
        self._coordinate_cache: OrderedDict[
            tuple[int, int, ProjectionRequest], tuple[object, object]
        ] = OrderedDict()
        # IMU-guided views change slightly every frame. Bounding this cache
        # prevents their 1440x1440 coordinate maps from exhausting VRAM while
        # retaining the recurring global-scan views.
        self._max_cached_coordinates = 48

    def _coordinates(
        self, in_h: int, in_w: int, request: ProjectionRequest
    ) -> tuple[object, object]:
        key = (in_h, in_w, request)
        cached = self._coordinate_cache.get(key)
        if cached is not None:
            self._coordinate_cache.move_to_end(key)
            return cached
        # Order 2 keeps py360convert's floating-point maps instead of converting
        # them to OpenCV's CPU-only fixed-point representation. Coordinates do
        # not depend on interpolation order.
        sampler = EquirecSampler.from_perspective(
            math.radians(request.fov),
            math.radians(request.fov),
            -math.radians(request.yaw),
            math.radians(request.pitch),
            math.radians(request.roll),
            in_h,
            in_w,
            request.size,
            request.size,
            2,
        )
        coord_y = self.cp.asarray(sampler._coor_y, dtype=self.cp.float32).squeeze()
        coord_x = self.cp.asarray(sampler._coor_x, dtype=self.cp.float32).squeeze()
        self._coordinate_cache[key] = (coord_y, coord_x)
        self._coordinate_cache.move_to_end(key)
        while len(self._coordinate_cache) > self._max_cached_coordinates:
            self._coordinate_cache.popitem(last=False)
        return coord_y, coord_x

    def _pad(self, pano: object) -> object:
        cp = self.cp
        height, width, channels = pano.shape
        padded = cp.empty((height + 2, width + 2, channels), dtype=pano.dtype)
        padded[1:-1, 1:-1] = pano
        padded[0, 1:-1] = cp.roll(pano[0], width // 2, axis=0)
        padded[-1, 1:-1] = cp.roll(pano[-1], width // 2, axis=0)
        padded[:, 0] = padded[:, -2]
        padded[:, -1] = padded[:, 1]
        return padded

    def project_many(
        self, pano: np.ndarray, requests: Sequence[ProjectionRequest]
    ) -> list[np.ndarray]:
        if not requests:
            return []
        cp = self.cp
        in_h, in_w = pano.shape[:2]
        pano_gpu = cp.asarray(pano)
        padded = self._pad(pano_gpu)
        outputs: list[np.ndarray] = []

        # Keep the panorama resident but remap one view at a time. This bounds
        # peak host/device memory even at 1440px and avoids a large temporary
        # coordinate tensor for full-scan runs.
        for request in requests:
            coord_y, coord_x = self._coordinates(in_h, in_w, request)
            coordinate_grid = cp.stack((coord_y, coord_x), axis=0)
            channels = [
                self.map_coordinates(
                    padded[..., channel],
                    coordinate_grid,
                    order=1,
                    mode="nearest",
                    prefilter=False,
                )
                for channel in range(pano.shape[2])
            ]
            output = cp.stack(channels, axis=-1).get()
            if output.ndim != 3 or output.shape[2] != pano.shape[2]:
                raise RuntimeError(f"unexpected CUDA projection shape: {output.shape}")
            outputs.append(output)
        return outputs


def make_projection_backend(name: str) -> ProjectionBackend:
    if name == "cpu":
        return CpuProjectionBackend()
    if name == "cuda":
        return CudaProjectionBackend()
    raise ValueError(f"unknown projection backend: {name}")
