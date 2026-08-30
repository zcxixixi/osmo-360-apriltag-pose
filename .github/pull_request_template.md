## Problem

<!-- One concrete problem. Do not mix unrelated cleanup or experiments. -->

## Decision

<!-- What changed and why this is the smallest durable solution. -->

## Non-goals

<!-- Explicitly list adjacent behavior that remains unchanged. -->

## Verification

- [ ] Focused behavioral test
- [ ] `uv run pytest -q`
- [ ] `./umi verify` when localization, calibration, force, rendering, or datasets change
- [ ] Real `umi inspect/process/review` path when a capture contract changes
- [ ] Browser or terminal smoke test for the changed user-facing surface

## Compatibility and artifacts

- [ ] No accepted artifact was overwritten
- [ ] New algorithm/calibration output uses a new revision and output directory
- [ ] README changed only when the user-facing interface changed
- [ ] AGENTS.md changed only when a durable invariant or current accepted pointer changed

## Risk

<!-- Failure modes, rollback, and any low-confidence or diagnostic-only result. -->
