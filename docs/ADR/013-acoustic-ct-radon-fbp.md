# ADR 013 — Acoustic Computed Tomography: Radon Transform + Filtered Backprojection

**Status:** Accepted
**Change class:** [major] (new reconstruction kernel)
**Date:** 2026-06-06

## Context

The Inverse Problems chapter (§6, "Acoustic Computed Tomography") documents transmission CT:
the travel-time of a ray through the medium is the line integral of slowness
`s(r) = 1/c(r)`, i.e. the **Radon transform** of the slowness field, inverted by
**filtered backprojection (FBP)**. The book audit found this presented as theory with **no
implementation** (`RadonTransform` / FBP NOT FOUND). The codebase has iterative SIRT
(`real_time_sirt`) and Born/CBS FWI, but no direct Radon/FBP analytic inversion.

## Decision

Implement a 2-D **parallel-beam** Radon transform and its filtered-backprojection inverse in
`kwavers_diagnostics::reconstruction::radon`:

- `radon_transform(image, n_angles) -> sinogram` — forward line-integral projection over
  `n_angles` uniformly spanning `[0, π)`, nearest-ray accumulation along rotated rows.
- `filtered_backprojection(sinogram, output_size) -> image` — Ram-Lak (ramp) filtering of each
  projection in the Fourier domain, then backprojection with bilinear sampling.

Straight-ray (first-order/Born) geometry only — matches the chapter's stated linear-ray
approximation. Bent-ray correction (SIRT/ART with ray tracing) is an explicit follow-on
(`real_time_sirt` already provides an iterative path).

## Alternatives considered

- **Iterative ART/SIRT** — already present for limited-angle/bent-ray; slower, not the
  analytic FBP the chapter documents. Kept as the complement, not the primary.
- **Fourier-slice (gridding) reconstruction** — `O(N² log N)` but needs non-uniform FFT
  gridding; FBP is simpler to verify and the canonical CT reference. Chosen.

## Verification

- **Round-trip**: forward-project a known phantom (centred + off-centred disks), FBP recovers
  it with high correlation and the reconstructed disk centroid at the true location.
- **Localization**: an off-centre disk reconstructs with its peak in the correct quadrant.
- Value-semantic assertions (correlation/centroid), not `is_ok()`.

## Consequences

- Enables analytic quantitative sound-speed CT from travel-time data.
- Straight-ray bias grows with `|∇c|/c`; documented, with SIRT/bent-ray as the upgrade path.
