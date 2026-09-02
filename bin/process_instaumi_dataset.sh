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

"$REPO_ROOT/run_pipeline.sh" "$DATASET_ROOT"
"$REPO_ROOT/.venv/bin/python" -m osmo360.datasets.instaumi_processed_export \
    --remove-pipeline-final \
    "$DATASET_ROOT"

printf 'Processed CSV files: %s/{trajectory,gripper,processed,metadata}.csv\n' \
    "$DATASET_ROOT/processed"
