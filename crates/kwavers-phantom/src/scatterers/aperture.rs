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
//! kernel is injected: a caller supplies [`RoundTripKernel`], whose signature
//! matches `CircularPistonSir::round_trip_response` so the physics type
//! satisfies it directly.
//!
//! # How the kernel enters
//!
//! The kernel is applied as a **unit-area** filter: the pulse is convolved with
//! `k / Σk·dt`, and the existing amplitude law (`1/r²` spreading, power-law
//! attenuation) is untouched.
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
//! - Tupholme, G. E. (1969). "Generation of acoustic pulses by baffled plane
//!   pistons." *Mathematika* 16(2), 209–224.
//! - Stepanishen, P. R. (1971). "Transient radiation from pistons in an infinite
//!   planar baffle." *J. Acoust. Soc. Am.* 49(5B), 1629–1638.

use kwavers_core::error::{KwaversError, KwaversResult};

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
/// The signature matches `CircularPistonSir::round_trip_response`, so that type
/// satisfies this seam without an adapter. Implementations are free to cache:
/// the kernel depends only on `(r, z)`, while synthesis evaluates it once per
/// element–scatterer pair.
pub trait RoundTripKernel {
    /// Two-way kernel for a field point at lateral offset `r_m` and axial
    /// distance `z_m`, sampled on a `dt_s` grid **from `t = 0`**.
    ///
    /// The support begins near `2·d_min/c`, so `n_samples` must reach past
    /// `2·d_max/c` or the returned kernel is all zeros and the scatterer
    /// contributes nothing. Synthesis strips the leading zeros and applies the
    /// delay itself.
    fn round_trip(&self, r_m: f64, z_m: f64, dt_s: f64, n_samples: usize) -> Vec<f64>;
}

impl<F> RoundTripKernel for F
where
    F: Fn(f64, f64, f64, usize) -> Vec<f64>,
{
    fn round_trip(&self, r_m: f64, z_m: f64, dt_s: f64, n_samples: usize) -> Vec<f64> {
        self(r_m, z_m, dt_s, n_samples)
    }
}

/// Reduce a round-trip kernel to a unit-area filter *shape*, dropping the
/// leading zeros that encode its propagation delay.
///
/// A provider samples the kernel from `t = 0`, so its support begins near
/// `2·d_min/c` and everything before that is zero. Synthesis applies the
/// round-trip delay itself, so those leading zeros must come off or the echo is
/// delayed twice. What remains is the aperture's temporal shape, normalized to
/// unit area so it filters without changing the amplitude law.
///
/// Returns `None` when the kernel carries no energy — the field point is
/// outside the aperture's support, or `n_samples` was too small to reach it.
pub(super) fn unit_area_shape(kernel: &[f64], dt: f64) -> Option<Vec<f64>> {
    let first = kernel.iter().position(|k| *k != 0.0)?;
    let last = kernel.iter().rposition(|k| *k != 0.0)?;
    let support = &kernel[first..=last];
    let area: f64 = support.iter().sum::<f64>() * dt;
    if !area.is_finite() || area <= 0.0 {
        return None;
    }
    Some(support.iter().map(|k| k / area).collect())
}

/// Discrete convolution `a ⊛ b` scaled by `dt`, truncated to `a.len() + b.len() - 1`.
pub(super) fn convolve(a: &[f64], b: &[f64], dt: f64) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0_f64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj * dt;
        }
    }
    out
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

#[cfg(test)]
#[path = "aperture_tests.rs"]
mod tests;
