# Security Policy

Security fixes target the current `main` branch until versioned release support
is published.

## Reporting A Vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's
private security advisory form and include the affected revision, a minimal
reproduction and impact. Remove private datasets, credentials and local paths.

Dataset archives and model checkpoints are external inputs. The downloader
verifies its declared archive checksum and rejects traversal, links and special
tar members; prepared data verifies manifest and source identities; schema-v2
checkpoints load in PyTorch tensor-only mode. External model factories are not
treated as checkpoint data: commands require an explicit matching reviewed
config before importing them. Report any path that bypasses these boundaries.
