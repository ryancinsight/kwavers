# ADR 140: Share the published typed providers across Kwavers

Status: Accepted

## Context

The earlier published cohort exposes Aequitas 0.4 and Leto 0.51 types. The
Aequitas 0.5 and Leto 0.52 types have different Cargo identities. Kwavers also
needs the typed Hephaestus FFT preflight before it allocates its PSTD state.

## Decision

Use the matching registry cohort for Aequitas, Leto, RITK, Athena, Apollo,
Coeus, and the Hephaestus 0.32 FFT backend. Pin the private Python quantity
reader to the revision that imports published Aequitas 0.5. Move all affected
public package versions together. The approved 0.4-based release continues
from its checked source commit. The 0.5 cohort uses a separate registry
release decision and the main-branch publisher remains unarmed for it.

The lossless pipelines use 23 storage bindings. The absorbing pipelines
add six through the paired spectral-symbol and material-coefficient
buffers from Kwavers #1123. Each pair keeps the eight-byte WGSL vector
stride. One host staging allocation supports successive medium updates.

## Verification

Run the real Metal FFT capacity and CPU–GPU parity cases and the locked
workspace gates. Resolve a standalone registry lock without workspace path
substitutions. Verify the lock has one Aequitas, Eunomia, and Leto identity.
