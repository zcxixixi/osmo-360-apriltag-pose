#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s /absolute/path/to/dataset-root\n' "$0" >&2
    exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$(readlink -f -- "$1")"
INSTAUMI_OK=1
for REQUIRED in dataset.h5 video/Left_back.mp4 video/Left_forward.mp4 video/Right_back.mp4 video/Right_forward.mp4; do
    if [[ ! -f "$DATASET_ROOT/$REQUIRED" ]]; then
        INSTAUMI_OK=0
        break
    fi
done
if [[ "$INSTAUMI_OK" != "1" && ( ! -d "$DATASET_ROOT/raw/left" || ! -d "$DATASET_ROOT/raw/right" ) ]]; then
    printf 'Dataset must contain dataset.h5 + video/*.mp4, or raw/left + raw/right: %s\n' "$DATASET_ROOT" >&2
    exit 2
fi

ARGS=(dataset "$DATASET_ROOT")
if [[ "${OSMO_PIPELINE_DRY_RUN:-0}" == "1" ]]; then
    ARGS+=(--dry-run)
fi
exec "$ROOT/umi" "${ARGS[@]}"
