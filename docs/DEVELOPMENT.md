# Development and release policy

## Stable main

`main` is releasable and must stay green. All code changes use a short-lived branch and a pull request. Do not develop on `main` or reuse a merged branch.

Branch names:

- `feat/<outcome>` for user-visible capability;
- `fix/<failure>` for a behavioral correction;
- `refactor/<boundary>` for behavior-preserving structure;
- `chore/<maintenance>` for repository or tooling changes.

## Pull requests

One PR solves one independently testable and revertible problem. It must state the problem, decision, non-goals, verification, compatibility, artifacts, and risk.

Required before merge:

1. focused behavioral test;
2. `uv run pytest -q`;
3. `./umi verify` for localization, calibration, force, rendering, or dataset changes;
4. real `umi inspect/process/review` when a capture contract changes;
5. browser or terminal smoke test for a changed user-facing surface.

Use **Squash and merge** by default so experimental commits do not pollute `main`. A merge commit is allowed only when retained commits are durable milestones or a successor baseline intentionally references an earlier commit. Delete the branch after merge.

## Versions and releases

The package follows semantic versioning while pre-1.0:

- `v0.MINOR.0` for a coherent new capability or contract;
- `v0.MINOR.PATCH` for compatible fixes;
- `v1.0.0` only after capture, world-pose, force, and dataset contracts are declared stable.

Do not tag every PR. Merge several compatible PRs, verify `main`, then create an annotated release tag. Release notes list user-visible behavior, migrations, accepted baselines, and known diagnostic-only limitations.

Capture manifests, calibration revisions, and published artifacts have their own immutable revision IDs. A package tag never permits overwriting an accepted artifact directory.

## Repository boundaries

- product code: `src/osmo360/`;
- offline and historical tools: `tools/`;
- compatibility launchers: `bin/`;
- tests: `tests/`;
- focused documentation: `docs/`;
- immutable machine configuration: `config/` and `manifests/`;
- sole root executable: `umi`.

Do not add Python modules, one-off scripts, generated media, or focused design documents to the repository root.

`README.md` is a user quick start, not an experiment log. `AGENTS.md` contains only durable invariants, current accepted pointers, and mandatory verification commands. Historical investigation belongs in Git history, PRs, or focused documents under `docs/`.

## Experiments and baselines

Experiments use new output directories and stay outside Git except for compact manifests, configuration, hashes, and acceptance metrics. Never overwrite an accepted output.

Old baseline manifests remain immutable and bind to historical Git commits. A source-layout or algorithm replacement creates a successor baseline and verifier; it does not rewrite the predecessor.
