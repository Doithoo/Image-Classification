# ADR 0001: Reproducible Classification Contracts

[简体中文](0001-reproducible-classification-contracts.zh-CN.md)

- Status: Accepted
- Scope: prepared data, model construction, checkpoints, runs and evaluation evidence

## Decisions

Prepared data has a schema-v2 identity derived from ordered classes, split
policy, every CSV checksum, per-class counts and source image bytes. Training
and evaluation verify that identity before making a metric claim.

Checkpoints are tensor-only, schema-versioned and atomically written. Resume
validates model specification, ordered classes, preprocessing and dataset
identity. Legacy checkpoints are not silently upgraded into resumable runs.

Runs and evaluation directories are immutable by default. A new run needs an
empty destination; evaluation uses split/TTA-specific paths; replacement
requires an explicit narrow `--overwrite` flag.

Model registration separates static `ModelSpec` metadata from construction.
Discovery never downloads weights. External factories are explicit trusted
imports and are recorded as provenance.

## Consequences

Preparation and verification cost I/O, but results remain inspectable when a
local dataset changes. Checkpoints and evidence include more metadata, but
prediction, export and comparison do not depend on unrecorded YAML state.
These contracts improve reproducibility without promising numerical identity
across hardware or library versions.
