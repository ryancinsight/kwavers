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
//! kernel is injected: a caller supplies an [`ApertureKernel`] — any
//! `Fn(f64, f64, f64, f64) -> SupportSamples` implements it — and a physics
//! closed form satisfies it directly: for a circular piston
//! `|x, y, z, dt| piston.response(x.hypot(y), z, dt)` and for a far-field
//! rectangle `|x, y, z, dt| rect.response(x, y, z, dt)`. The one-way response
//! is the required method; the round trip defaults to its auto-convolution.
//! The field point reaches the provider in the element's own frame — `x`
//! along the element's width axis, `y` along its height, `z` along its
//! outward normal — so an anisotropic aperture sees its orientation.
//!
//! # How the kernel enters
//!
//! On the monostatic path the kernel is applied as a **unit-area** filter: the
//! echo is convolved with `k / Σk·dt`, and the existing amplitude law (`1/r²`
//! spreading, power-law attenuation) is untouched. The array-transmit path
//! keeps the responses' own areas instead — there they are the amplitude
//! (see `synthesize_rf_with_array_transmit`).
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
use kwavers_math::numerics::convolution::SupportSamples;
use std::ops::RangeInclusive;

/// A transducer element with an aperture frame.
///
/// A bare position cannot express an aperture: the spatial impulse response is
/// a function of the field point in the element's own frame, so the element
/// carries the outward normal that defines its axis and the in-plane width
/// axis that orients a rectangular face. `ConvexArrayGeometry` produces the
/// centre and normal (ADR 112); the width axis is the array's lateral tangent
/// there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApertureElement {
    /// Element centre \[m].
    pub position: [f64; 3],
    /// Outward unit normal of the element face.
    pub normal: [f64; 3],
    /// Unit vector along the element's width, in the face plane.
    pub width_axis: [f64; 3],
}

impl ApertureElement {
    /// Construct from a centre, an outward normal, and a width direction.
    ///
    /// The normal is normalized; the width direction has its component along
    /// the normal removed and is then normalized, so a slightly skewed input
    /// still yields an orthonormal frame. The height axis is `normal × width`.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` if any vector is non-finite, the
    /// normal has zero length, or the width direction is parallel to it.
    pub fn new(position: [f64; 3], normal: [f64; 3], width: [f64; 3]) -> KwaversResult<Self> {
        for (name, v) in [("position", position), ("normal", normal), ("width", width)] {
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
        let normal = [normal[0] / norm, normal[1] / norm, normal[2] / norm];
        let along = dot(width, normal);
        let in_plane = [
            width[0] - along * normal[0],
            width[1] - along * normal[1],
            width[2] - along * normal[2],
        ];
        let width_norm = dot(in_plane, in_plane).sqrt();
        if width_norm <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "ApertureElement width axis must not be parallel to the normal".to_owned(),
            ));
        }
        Ok(Self {
            position,
            normal,
            width_axis: [
                in_plane[0] / width_norm,
                in_plane[1] / width_norm,
                in_plane[2] / width_norm,
            ],
        })
    }

    /// Unit vector along the element's height, `normal × width_axis`.
    #[must_use]
    pub fn height_axis(&self) -> [f64; 3] {
        cross(self.normal, self.width_axis)
    }

    /// Field-point coordinates `[x, y, z]` of `target` in this element's
    /// frame: `x` along the width axis, `y` along the height axis, `z` along
    /// the outward normal.
    ///
    /// Returns `None` when the target is at or behind the face (`z <= 0`),
    /// where a baffled-piston SIR is not defined.
    #[must_use]
    pub fn field_point(&self, target: [f64; 3]) -> Option<[f64; 3]> {
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
        Some([dot(d, self.width_axis), dot(d, self.height_axis()), z])
    }
}

/// Supplies an aperture's spatial impulse response at a field point.
///
/// A provider samples the response over **its own support only** — from the
/// onset `d_min/c` to `d_max/c` — with the onset's grid index carried in
/// [`SupportSamples::first_sample`], so the cost of one call is the kernel's
/// width in samples and nothing else, and an assembly of several elements can
/// place each response at its own time. `CircularPistonSir::response` and
/// `FarFieldRectangleSir::response` return exactly that. Implementations are
/// free to cache: the response depends only on the field point in the
/// element frame, while synthesis evaluates it once per element–scatterer
/// pair.
pub trait ApertureKernel {
    /// One-way response for a field point at `[x_m, y_m, z_m]` in the
    /// element's frame (width, height, outward normal — see
    /// [`ApertureElement::field_point`]), on a `dt_s` grid, from its onset.
    ///
    /// The response of a finite aperture is never empty: a support narrower
    /// than one sample is returned as its impulse equivalent, one sample of
    /// `A/dt` carrying the response's area `A`, as the physics providers do.
    /// An empty or zero-area return and a non-finite sample are provider
    /// defects that synthesis reports rather than absorbs — on the array path
    /// the area is the amplitude, so dropping it would be silence.
    fn response(&self, x_m: f64, y_m: f64, z_m: f64, dt_s: f64) -> SupportSamples;

    /// Two-way (monostatic pulse-echo) kernel `(h ⊛ h)·dt`, the
    /// auto-convolution of [`Self::response`]; its onset is twice the one-way
    /// onset. Override only when a provider has a cheaper closed form.
    fn round_trip(&self, x_m: f64, y_m: f64, z_m: f64, dt_s: f64) -> SupportSamples {
        self.response(x_m, y_m, z_m, dt_s).auto_convolve(dt_s)
    }
}

impl<F> ApertureKernel for F
where
    F: Fn(f64, f64, f64, f64) -> SupportSamples,
{
    fn response(&self, x_m: f64, y_m: f64, z_m: f64, dt_s: f64) -> SupportSamples {
        self(x_m, y_m, z_m, dt_s)
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

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
#[path = "aperture_tests.rs"]
mod tests;
