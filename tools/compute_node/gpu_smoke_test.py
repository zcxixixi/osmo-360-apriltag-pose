#!/usr/bin/env python3
import importlib.util
import json
import subprocess


def command(*args: str) -> dict[str, object]:
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=False)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError:
        return {"returncode": 127, "stdout": "", "stderr": f"{args[0]} not found"}


report: dict[str, object] = {
    "nvidia_smi": command(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader",
    ),
    "packages": {
        name: bool(importlib.util.find_spec(name))
        for name in ("numpy", "cv2", "cupy", "torch", "av", "PyNvVideoCodec")
    },
}
try:
    import cupy as cp

    report["cupy"] = {
        "device_count": cp.cuda.runtime.getDeviceCount(),
        "device_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "test_sum": int(cp.arange(1_000_000).sum().get()),
    }
except Exception as exc:
    report["cupy_error"] = repr(exc)
try:
    import PyNvVideoCodec as nvc

    report["pynvvideocodec"] = {"version": nvc.__version__}
except Exception as exc:
    report["pynvvideocodec_error"] = repr(exc)
print(json.dumps(report, indent=2))
