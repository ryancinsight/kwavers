//! KWaveArray — Custom transducer geometry builder for k-Wave compatibility.
//!
//! Builder-pattern API for creating custom transducer arrays with mixed
//! element geometries (arcs, discs, rectangles, bowls, annuli), matching
//! k-wave-python's `KWaveArray`.
//!
//! # Module layout
//!
//! - `math`: module-level constants and rotation/linear-algebra helpers.
//! - `construction`: constructors and element-addition methods.
//! - `transform`: global array-position transform helpers.
//! - `geometry`: surface-area and arc-length formulae.
//! - `bli_kernel`: BLI stencil, disc basis, and nearest-index helpers.
//! - `rasterizer_curved`: arc, bowl, and annulus element rasterizers.
//! - `rasterizer_planar`: rect, disc, and per-element rasterizers.
//! - `accessors`: public query methods (masks, delays, apodization, etc.).

mod accessors;
mod bli_kernel;
mod construction;
mod geometry;
pub(crate) mod math;
mod rasterizer_curved;
mod rasterizer_planar;
mod transform;

#[cfg(test)]
mod tests;

/// Apodization window for per-element amplitude weighting.
///
/// # Theorem — Window Functions (Harris 1978)
///
/// Window functions reduce sidelobes in the array beam pattern at the cost of
/// broadening the main lobe. For an N-element array with uniform spacing:
/// - Rectangular: unity weights → minimum main-lobe width, maximum sidelobes (−13 dB)
/// - Hann: `wᵢ = 0.5(1 − cos(2πi/(N−1)))` → −31 dB sidelobes, 1.5× wider main lobe
/// - Hamming: `wᵢ = 0.54 − 0.46·cos(2πi/(N−1))` → −43 dB sidelobes
///
/// Reference: Harris, F.J. (1978). "On the use of windows for harmonic analysis
/// with the discrete Fourier transform." Proc. IEEE 66(1):51–83.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KwaveApodizationWindow {
    /// All weights = 1.0 (no apodization)
    Rectangular,
    /// Hann window — −31 dB sidelobes
    Hann,
    /// Hamming window — −43 dB sidelobes
    Hamming,
    /// Blackman window — −57 dB sidelobes
    Blackman,
    /// Tukey (tapered-cosine) window with cosine fraction `r ∈ [0, 1]`:
    /// rectangular at `r = 0`, Hann at `r = 1`; `r` is clamped to `[0, 1]`.
    Tukey(f64),
}

/// Normalized source-pressure profile over a finite disc element.
///
/// The profile scales each disc surface sample before BLI rasterization. This
/// models mode-dependent finite-source shapes while preserving the original
/// uniform-disc API. The radial-power profile uses
/// `w(r) = (p + 2) r^p / 2`, whose continuous area average over the unit disc is
/// one for `p >= 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscSourceProfile {
    radial_power_exponent: f64,
}

impl DiscSourceProfile {
    /// Uniform finite-disc source pressure.
    #[must_use]
    pub const fn uniform() -> Self {
        Self {
            radial_power_exponent: 0.0,
        }
    }

    /// Edge-weighted radial-power source profile.
    ///
    /// # Errors
    /// Returns an error when `exponent` is non-finite or negative.
    pub fn radial_power(exponent: f64) -> Result<Self, String> {
        if exponent.is_finite() && exponent >= 0.0 {
            Ok(Self {
                radial_power_exponent: exponent,
            })
        } else {
            Err(format!(
                "disc source radial-power exponent must be finite and nonnegative, got {exponent}"
            ))
        }
    }

    /// Weight at normalized radius `r/R`, clamped to `[0, 1]`.
    #[must_use]
    pub fn weight_at_normalized_radius(self, radius_fraction: f64) -> f64 {
        let r = radius_fraction.clamp(0.0, 1.0);
        if self.radial_power_exponent == 0.0 {
            1.0
        } else {
            0.5 * (self.radial_power_exponent + 2.0) * r.powf(self.radial_power_exponent)
        }
    }
}

impl Default for DiscSourceProfile {
    fn default() -> Self {
        Self::uniform()
    }
}

/// Element shape types for custom transducer arrays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementShape {
    /// Arc-shaped element (curved line in 3D)
    Arc {
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
        start_angle: f64,
        end_angle: f64,
    },
    /// Rectangular element
    Rect {
        position: (f64, f64, f64),
        width: f64,
        height: f64,
        /// Retained for API compatibility.
        length: f64,
        /// Intrinsic X-Y-Z Euler rotation in degrees applied about the element
        /// center. `(0.0, 0.0, 0.0)` keeps the rectangle axis-aligned.
        euler_xyz_deg: (f64, f64, f64),
    },
    /// Disc/circular element
    Disc {
        position: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    },
    /// Disc/circular element with finite-source surface profile.
    ProfiledDisc {
        position: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
        profile: DiscSourceProfile,
    },
    /// Bowl/spherical-cap element
    Bowl {
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
    },
    /// Annular spherical-cap element — bowl surface bounded by two aperture
    /// diameters. Facing convention matches `Bowl` (center-of-curvature at
    /// `position`, cap opening along −X).
    Annulus {
        position: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
    },
}

/// Custom transducer array with mixed element geometries.
///
/// Allows building arbitrary transducer arrays by adding elements of different
/// shapes, matching k-wave-python's `KWaveArray` functionality.
#[derive(Debug, Clone)]
pub struct KWaveArray {
    pub(super) elements: Vec<ElementShape>,
    pub(super) frequency: f64,
    pub(super) sound_speed: f64,
    pub(super) _element_width: f64,
    /// Optional global affine transform applied on top of every element.
    /// `None` ⟹ identity (element poses used as-is). Mirrors
    /// k-wave-python's `kWaveArray.set_array_position`.
    pub(super) array_transform: Option<ArrayTransform>,
}

/// Global translation + intrinsic X-Y-Z Euler rotation (degrees) applied to
/// every element before rasterization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ArrayTransform {
    pub(super) translation: (f64, f64, f64),
    pub(super) euler_xyz_deg: (f64, f64, f64),
}

impl Default for KWaveArray {
    fn default() -> Self {
        Self::new()
    }
}
