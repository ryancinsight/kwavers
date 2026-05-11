//! Surface-area and arc-length formulae for element geometries.
//!
//! # Mathematical Foundation
//!
//! ## Spherical cap area (Bowl element)
//!
//! For a spherical cap of radius of curvature `R` and aperture diameter `D`:
//! ```text
//! h = R − √(R² − (D/2)²)       (sagittal depth)
//! A_bowl = 2πRh
//! ```
//! Reference: Treeby, B.E. & Cox, B.T. (2010). "k-Wave: MATLAB toolbox for
//! the simulation and reconstruction of photoacoustic wave fields." J. Biomed.
//! Opt. 15(2):021314.
//!
//! ## Annular cap area
//!
//! Ring between polar angles `φ_inner` and `φ_outer` (both measured from the
//! bowl axis):
//! ```text
//! φ = arcsin(d/(2R))   (for each aperture diameter d)
//! A_annulus = 2πR²(cos(φ_inner) − cos(φ_outer))
//! ```
//!
//! ## Arc line length
//!
//! ```text
//! L = R · |Δθ|    (Δθ in radians)
//! ```

use super::KWaveArray;

impl KWaveArray {
    /// Arc length subtended by `[start_angle, end_angle]` (degrees) at radius `radius`.
    #[inline]
    pub(super) fn arc_line_length(radius: f64, start_angle: f64, end_angle: f64) -> f64 {
        radius * (end_angle - start_angle).abs().to_radians()
    }

    /// Spherical cap surface area for a bowl with radius of curvature `radius`
    /// and aperture diameter `diameter`.
    ///
    /// # Proof
    ///
    /// The sagittal depth `h = R − √(R² − (D/2)²)` is the distance from the
    /// aperture plane to the cap vertex. The integral of the surface element
    /// `dA = 2πR · dh` from 0 to h gives `A = 2πRh`. When `R ≤ D/2` (hemisphere
    /// or greater), h is clamped to R (half-sphere area `2πR²`).
    #[inline]
    pub(super) fn bowl_surface_area(radius: f64, diameter: f64) -> f64 {
        let half_aperture = diameter / 2.0;
        let h = if radius > half_aperture {
            radius - radius.mul_add(radius, -(half_aperture * half_aperture)).sqrt()
        } else {
            radius
        };
        2.0 * std::f64::consts::PI * radius * h
    }

    /// Annular spherical-cap surface area between `inner_diameter` and
    /// `outer_diameter` on a bowl of radius of curvature `radius`.
    ///
    /// # Proof
    ///
    /// The annulus spans polar angles `φ_inner = arcsin(d_inner/(2R))` to
    /// `φ_outer = arcsin(d_outer/(2R))`. Integration of `2πR sin(φ) dφ` over
    /// this range gives `2πR²(cos(φ_inner) − cos(φ_outer))`.
    #[inline]
    pub(super) fn annulus_surface_area(
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
    ) -> f64 {
        let phi_outer = (outer_diameter / 2.0 / radius).clamp(0.0, 1.0).asin();
        let phi_inner = (inner_diameter / 2.0 / radius).clamp(0.0, 1.0).asin();
        2.0 * std::f64::consts::PI * radius * radius * (phi_inner.cos() - phi_outer.cos())
    }
}
