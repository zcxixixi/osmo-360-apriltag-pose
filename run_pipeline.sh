#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s /absolute/path/to/dataset-root\n' "$0" >&2
    exit 2
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="$(readlink -f -- "$1")"
if [[ ! -d "$DATASET_ROOT/raw/left" || ! -d "$DATASET_ROOT/raw/right" ]]; then
    printf 'Dataset must contain raw/left/ and raw/right/: %s\n' "$DATASET_ROOT" >&2
    exit 2
fi

ARGS=(dataset "$DATASET_ROOT")
if [[ "${INSTAUMI_PIPELINE_DRY_RUN:-0}" == "1" ]]; then
    ARGS+=(--dry-run)
fi
exec "$ROOT/umi" "${ARGS[@]}"
