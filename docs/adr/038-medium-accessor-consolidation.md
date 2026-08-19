# ADR 038: Consolidate continuous medium access

- **Status:** Accepted
- **Date:** 2026-07-17
- **Change class:** [major]

## Context

`kwavers-medium` exposed each continuous-coordinate property accessor twice:
once with a `dyn Medium` parameter and once as a `*_at_core` generic
forwarder. The implementations were identical. The duplicate symbols enlarged
the public surface and made the obsolete wrapper shape part of the API even
though `Medium` is a blanket refinement of `CoreMedium`.

## Decision

Make the canonical `density_at`, `sound_speed_at`, `absorption_at`, and
`nonlinearity_at` functions generic over `CoreMedium + ?Sized`. Update point-wise
interface detection to call those functions and delete all four `*_at_core`
functions and re-exports. Bump `kwavers-medium` from 3.0.0 to 4.0.0 because
the public transitional symbols are removed; no compatibility re-export is
retained.

## Theorem

`Medium` has a blanket implementation for every type satisfying `CoreMedium`
and its additional property traits. Therefore every former `&dyn Medium`
caller satisfies the new `CoreMedium` bound, while concrete `CoreMedium`
callers gain the same entry point. Since both APIs apply the unchanged
`continuous_to_discrete` map and invoke the same indexed property method, the
refactor is value-preserving and removes only duplicate symbols.

## Verification

- Source audit: no `*_at_core` references remain.
- `cargo check --locked -p kwavers-medium` passes.
- `cargo clippy --locked -p kwavers-medium --all-targets -- -D warnings` passes.
- Locked nightly Nextest: 187/187 `kwavers-medium` tests pass in 5.276 seconds.
