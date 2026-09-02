#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s /absolute/path/to/instaumi-dataset\n' "$0" >&2
    exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -d "$1" ]]; then
    printf 'Dataset directory does not exist: %s\n' "$1" >&2
    exit 2
fi
DATASET_ROOT="$(readlink -f -- "$1")"
PROCESS_START_NS="$(date +%s%N)"

elapsed_seconds() {
    local current_ns elapsed_ms
    current_ns="$(date +%s%N)"
    elapsed_ms="$(( (current_ns - PROCESS_START_NS) / 1000000 ))"
    printf '%d.%03d' "$(( elapsed_ms / 1000 ))" "$(( elapsed_ms % 1000 ))"
}

for REQUIRED in \
    dataset.h5 \
    video/Left_back.mp4 \
    video/Left_forward.mp4 \
    video/Right_back.mp4 \
    video/Right_forward.mp4
do
    if [[ ! -f "$DATASET_ROOT/$REQUIRED" ]]; then
        printf 'Missing required InstaUMI input: %s\n' "$DATASET_ROOT/$REQUIRED" >&2
        exit 2
    fi
done

PROGRESS_PID=""
stop_progress() {
    if [[ -n "$PROGRESS_PID" ]] && kill -0 "$PROGRESS_PID" 2>/dev/null; then
        kill "$PROGRESS_PID" 2>/dev/null || true
    fi
    [[ -z "$PROGRESS_PID" ]] || wait "$PROGRESS_PID" 2>/dev/null || true
    PROGRESS_PID=""
}
trap stop_progress EXIT

printf '[开始] 数据集：%s\n' "$DATASET_ROOT" >&2
printf '[轨迹 1/2] 启动四路视频处理与联合轨迹求解\n' >&2
PIPELINE_PROGRESS_START_NS="$(date +%s%N)"
"$REPO_ROOT/.venv/bin/python" -m osmo360.datasets.instaumi_progress \
    "$DATASET_ROOT" \
    --started-at-ns "$PIPELINE_PROGRESS_START_NS" &
PROGRESS_PID=$!
if "$REPO_ROOT/run_pipeline.sh" "$DATASET_ROOT" >/dev/null; then
    stop_progress
else
    PIPELINE_EXIT=$?
    stop_progress
    printf '[失败] 轨迹流水线退出，状态码：%d；已耗时：%ss\n' \
        "$PIPELINE_EXIT" "$(elapsed_seconds)" >&2
    exit "$PIPELINE_EXIT"
fi
printf '[轨迹 1/2] 完成；累计耗时：%ss\n' "$(elapsed_seconds)" >&2

printf '[导出 2/2] 正在原子发布轨迹与夹爪 CSV\n' >&2
if "$REPO_ROOT/.venv/bin/python" -m osmo360.datasets.instaumi_processed_export \
    --remove-pipeline-final "$DATASET_ROOT" >/dev/null
then
    :
else
    EXPORT_EXIT=$?
    printf '[失败] CSV 导出退出，状态码：%d；已耗时：%ss\n' \
        "$EXPORT_EXIT" "$(elapsed_seconds)" >&2
    exit "$EXPORT_EXIT"
fi

printf '[完成] 整条处理耗时：%ss\n' "$(elapsed_seconds)" >&2
printf '[输出] %s/{trajectory,gripper,processed,metadata}.csv\n' \
    "$DATASET_ROOT/processed" >&2
