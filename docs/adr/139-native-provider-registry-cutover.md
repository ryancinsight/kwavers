# ADR 139: Native providers and the Kwavers registry graph

Status: Accepted

## Context

The clinical APIs use a system clock and report timestamp conversion errors.
The external DICOM dataset and browser plotting packages brought Chrono into
the dependency graph. Atlas providers also moved their branches to Aequitas
0.5 while Kwavers still uses the 0.4 quantity types. Rust treats types from the
two versions as distinct even when their dimensions and representations agree.

## Decision

Use Horae for civil-time conversion, Iris for standalone plotting, and RITK
for DICOM data and codecs. Use the Aequitas 0.4 cohort on crates.io for
materials, transport, time stepping, and sampling. Hephaestus 0.31
provides the typed FFT capacity preflight on this cohort. The Python extension
retains an exact source revision for its private quantity protocol binding.

Release the 22 public packages in dependency order. Advance
`kwavers-analysis` and `kwavers-diagnostics` to 4.0.0 for the clinical API
changes. The other packages use their manifest versions. ADR 138 governs
the separate stable application milestone.

## Alternatives

A Chrono adapter would preserve the redundant runtime dependency. Keeping
branch dependencies for typed providers allows a fresh lock to select
incompatible quantities. Registry package requirements express the shared
type identity used at the application boundary.

## Verification

Generate the workspace lockfile from scratch outside the Atlas overlay.
Verify the candidate and published consumer using all features, all targets,
and all dependency kinds. Reject Chrono in the supply-chain gate. Verify the
native dataset, plotting, and clinical reports with their existing value tests
and check archive checksums against the registry index.

## Aequitas 0.5 workspace migration

The workspace uses the Aequitas 0.5 and Leto 0.52 registry cohort.
Hephaestus 0.32 supplies the typed GPU FFT capacity check for that
cohort. Kwavers pins the merged RITK revision while the current RITK
packages still select Aequitas 0.4. The next Kwavers registry release
requires compatible RITK packages and a new consumer check.
