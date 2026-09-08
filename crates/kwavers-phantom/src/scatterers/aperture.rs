//! Finite-aperture (diffraction) refinement of point-scatterer RF synthesis.
//!
//! The point-element model in the parent module treats each element as a point:
//! exact for point elements and far-field scatterers, but it carries no
//! diffraction. Field II refines it by convolving in the Tupholme–Stepanishen
//! spatial impulse response (SIR) of the extended aperture.
//!
//! # The seam
//!
//! This crate does not depend on `kwavers-physics`, which owns the SIR closed
//! forms, and deliberately does not gain that edge (ADR 113). Instead the
//! kernel is injected: a caller supplies a [`RoundTripKernel`] — any
//! `Fn(f64, f64, f64) -> Vec<f64>` implements it — and the physics closed form
//! supplies it through a one-line adapter,
//! `|r, z, dt| piston.round_trip_response(r, z, dt).samples`, the onset index
//! discarded because synthesis applies the round-trip delay itself.
//!
//! # How the kernel enters
//!
//! The kernel is applied as a **unit-area** filter: the echo is convolved with
//! `k / Σk·dt`, and the existing amplitude law (`1/r²` spreading, power-law
//! attenuation) is untouched.
//!
//! Convolution distributes over the sum of echoes, so synthesis accumulates
//! every scatterer's scaled, delayed kernel into one trace per element and
//! convolves the pulse into that trace once — the per-pair work is the kernel
//! alone, never the pulse or the trace length. That is the structure of
//! Rivera, Demené & Tanter (2026), whose sparse-delta kernels this seam can
//! carry once a far-field patch provider exists: the provider samples its
//! own support, and what it returns is placed, not re-scanned.
//!
//! That normalization is forced by the requirement that this refinement reduce
//! to what it refines. The raw two-way kernel integrates to
//! `(√(z²+a²) − z)²` on axis, which tends to `0` as the aperture radius `a → 0`
//! — so convolving the raw kernel converges on silence, not on the point-element
//! model. Normalized, the kernel tends to a delta and the output converges on
//! `synthesize_rf` exactly.
//!
//! So this models the finite aperture's **temporal** response — the near-field
//! smearing that a point element cannot express — while leaving the echo's
//! amplitude to the established law. It is not the full Field II amplitude
//! model, which derives amplitude from the SIR itself.
//!
//! # References
//! - Rivera, D. E., Demené, C., & Tanter, M. (2026). "Sparse Delta Integration
//!   method for the calculation of spatiotemporal pressure fields of arbitrary
//!   ultrasound transducer geometries." arXiv:2608.26891 — the per-pair cost
//!   model (constant per patch, integration once per trace).
//! - Tupholme, G. E. (1969). "Generation of acoustic pulses by baffled plane
//!   pistons." *Mathematika* 16(2), 209–224.
//! - Stepanishen, P. R. (1971). "Transient radiation from pistons in an infinite
//!   planar baffle." *J. Acoust. Soc. Am.* 49(5B), 1629–1638.

use kwavers_core::error::{KwaversError, KwaversResult};
use std::ops::RangeInclusive;

/// A transducer element with an aperture frame.
///
/// A bare position cannot express an aperture: the spatial impulse response is
/// a function of the field point in the element's own frame, so the element
/// must carry the outward normal that defines it. `ConvexArrayGeometry`
/// produces exactly this pair (ADR 112).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApertureElement {
    /// Element centre \[m].
    pub position: [f64; 3],
    /// Outward unit normal of the element face.
    pub normal: [f64; 3],
}

impl ApertureElement {
    /// Construct from a centre and an outward normal.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` if either vector is non-finite or
    /// the normal has zero length.
    pub fn new(position: [f64; 3], normal: [f64; 3]) -> KwaversResult<Self> {
        for (name, v) in [("position", position), ("normal", normal)] {
            if !v.iter().all(|c| c.is_finite()) {
                return Err(KwaversError::InvalidInput(format!(
                    "ApertureElement {name} must be finite, got {v:?}"
                )));
            }
        }
        let norm = dot(normal, normal).sqrt();
        if norm <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "ApertureElement normal must have non-zero length".to_owned(),
            ));
        }
        Ok(Self {
            position,
            normal: [normal[0] / norm, normal[1] / norm, normal[2] / norm],
        })
    }

    /// Field-point coordinates `(r, z)` of `target` in this element's frame:
    /// `z` along the outward normal, `r` the lateral offset from that axis.
    ///
    /// Returns `None` when the target is at or behind the face (`z <= 0`),
    /// where a baffled-piston SIR is not defined.
    #[must_use]
    pub fn field_point(&self, target: [f64; 3]) -> Option<(f64, f64)> {
        let d = [
            target[0] - self.position[0],
            target[1] - self.position[1],
            target[2] - self.position[2],
        ];
        let z = dot(d, self.normal);
        // `target` is not validated on the way in, so a non-finite coordinate
        // must be rejected here rather than propagating into the kernel.
        if !z.is_finite() || z <= 0.0 {
            return None;
        }
        let lateral = [
            d[0] - z * self.normal[0],
            d[1] - z * self.normal[1],
            d[2] - z * self.normal[2],
        ];
        Some((dot(lateral, lateral).sqrt(), z))
    }
}

/// Supplies the round-trip (two-way) spatial impulse response of an aperture.
///
/// A provider samples the kernel over **its own support only** — from the
/// round-trip onset `2·d_min/c` to `2·d_max/c` — so the cost of one call is the
/// kernel's width in samples and nothing else. The `samples` of
/// `CircularPistonSir::round_trip_response` are exactly that; its onset index
/// is dropped at the seam, because synthesis applies the round-trip delay
/// itself. Implementations are free to cache: the kernel depends only on
/// `(r, z)`, while synthesis evaluates it once per element–scatterer pair.
pub trait RoundTripKernel {
    /// Two-way kernel samples for a field point at lateral offset `r_m` and
    /// axial distance `z_m`, on a `dt_s` grid, starting at the kernel's onset.
    ///
    /// An empty return means the support is narrower than one sample; synthesis
    /// then treats the aperture as a point element. Leading or trailing zeros
    /// are tolerated and stripped, but they are wasted work, not delay: the
    /// delay is synthesis's own.
    fn round_trip(&self, r_m: f64, z_m: f64, dt_s: f64) -> Vec<f64>;
}

impl<F> RoundTripKernel for F
where
    F: Fn(f64, f64, f64) -> Vec<f64>,
{
    fn round_trip(&self, r_m: f64, z_m: f64, dt_s: f64) -> Vec<f64> {
        self(r_m, z_m, dt_s)
    }
}

/// The non-zero span of a kernel and its area `Σk·dt` over that span.
///
/// Dividing the span by the area makes the kernel a unit-area filter *shape*,
/// which is what lets it refine the amplitude law without changing it.
///
/// Returns `None` when the kernel carries no energy — the field point is
/// outside the aperture's support, or the support is narrower than one sample.
pub(super) fn support_and_area(kernel: &[f64], dt: f64) -> Option<(RangeInclusive<usize>, f64)> {
    let first = kernel.iter().position(|k| *k != 0.0)?;
    let last = kernel.iter().rposition(|k| *k != 0.0)?;
    let area: f64 = kernel[first..=last].iter().sum::<f64>() * dt;
    if !area.is_finite() || area <= 0.0 {
        return None;
    }
    Some((first..=last, area))
}

/// Accumulate the discrete convolution `(trace ⊛ pulse)·dt` into `out`,
/// truncated to `out.len()`; trace samples past `out.len()` cannot reach the
/// output and are ignored.
///
/// Runs once per element over the non-zero trace samples only — most of a
/// sparse scatterer field is zero — so it costs `O(N_nz·L)` per element for
/// `N_nz` occupied samples and an `L`-tap pulse. Whether an FFT convolution
/// beats that at some pulse length is a measurement, not taken here.
pub(super) fn convolve_into(out: &mut [f64], trace: &[f64], pulse: &[f64], dt: f64) {
    for (start, &value) in trace.iter().take(out.len()).enumerate() {
        if value == 0.0 {
            continue;
        }
        let scaled = value * dt;
        let span = out.len().saturating_sub(start).min(pulse.len());
        for (slot, &tap) in out[start..start + span].iter_mut().zip(pulse) {
            *slot += scaled * tap;
        }
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

#[cfg(test)]
#[path = "aperture_tests.rs"]
mod tests;
