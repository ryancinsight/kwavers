//! Shepp–Logan numerical head phantom.
//!
//! The Shepp–Logan phantom (Shepp & Logan 1974) is the standard synthetic test
//! image for tomographic/seismic reconstruction: a sum of ten overlapping
//! ellipses on the square `[-1, 1] × [-1, 1]`, each with an additive intensity.
//! The phantom value at a point is the sum of the intensities of the ellipses
//! that contain it.
//!
//! Two intensity sets are provided: the [`SheppLoganVariant::Original`] (Shepp &
//! Logan 1974) and the higher-contrast [`SheppLoganVariant::Modified`]
//! (Toft 1996) used by most imaging toolboxes. Both share the same ten-ellipse
//! geometry.
//!
//! # Coordinate convention
//! `x` rightward, `y` upward, both in `[-1, 1]`; ellipse rotation `φ` is measured
//! counter-clockwise from the `+x` axis. A point `(x, y)` is inside ellipse
//! `(A, a, b, x₀, y₀, φ)` iff `(u/a)² + (v/b)² ≤ 1` with
//! `u = (x−x₀)cosφ + (y−y₀)sinφ`, `v = −(x−x₀)sinφ + (y−y₀)cosφ`.
//!
//! # References
//! - Shepp, L. A., & Logan, B. F. (1974). "The Fourier reconstruction of a head
//!   section." *IEEE Trans. Nucl. Sci.* 21(3), 21–43.
//! - Toft, P. (1996). *The Radon Transform — Theory and Implementation*, PhD
//!   thesis (modified Shepp–Logan intensities).

use leto::Array2;

/// One additive ellipse of the phantom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    /// Additive intensity `A`.
    pub intensity: f64,
    /// `x` semi-axis `a`.
    pub a: f64,
    /// `y` semi-axis `b`.
    pub b: f64,
    /// Centre `x₀`.
    pub x0: f64,
    /// Centre `y₀`.
    pub y0: f64,
    /// Rotation `φ` in radians, counter-clockwise from `+x`.
    pub phi: f64,
}

impl Ellipse {
    /// Whether the point `(x, y)` lies inside this ellipse.
    #[must_use]
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let dx = x - self.x0;
        let dy = y - self.y0;
        let (s, c) = self.phi.sin_cos();
        let u = dx * c + dy * s;
        let v = -dx * s + dy * c;
        (u / self.a).powi(2) + (v / self.b).powi(2) <= 1.0
    }
}

/// Which intensity set the phantom uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SheppLoganVariant {
    /// Original Shepp & Logan (1974) intensities (low soft-tissue contrast).
    Original,
    /// Modified Shepp–Logan (Toft 1996) intensities (higher contrast).
    Modified,
}

/// The ten-ellipse Shepp–Logan phantom.
#[derive(Debug, Clone, PartialEq)]
pub struct SheppLogan {
    ellipses: Vec<Ellipse>,
}

/// Shared geometry `(a, b, x₀, y₀, φ_deg)` of the ten ellipses.
const GEOMETRY: [(f64, f64, f64, f64, f64); 10] = [
    (0.6900, 0.9200, 0.0, 0.0000, 0.0),
    (0.6624, 0.8740, 0.0, -0.0184, 0.0),
    (0.1100, 0.3100, 0.22, 0.0000, -18.0),
    (0.1600, 0.4100, -0.22, 0.0000, 18.0),
    (0.2100, 0.2500, 0.0, 0.3500, 0.0),
    (0.0460, 0.0460, 0.0, 0.1000, 0.0),
    (0.0460, 0.0460, 0.0, -0.1000, 0.0),
    (0.0460, 0.0230, -0.08, -0.6050, 0.0),
    (0.0230, 0.0230, 0.0, -0.6060, 0.0),
    (0.0230, 0.0460, 0.06, -0.6050, 0.0),
];

/// Original Shepp–Logan (1974) intensities.
const INTENSITY_ORIGINAL: [f64; 10] =
    [2.0, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];

/// Modified Shepp–Logan (Toft 1996) intensities.
const INTENSITY_MODIFIED: [f64; 10] = [1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];

impl SheppLogan {
    /// Build a phantom with the given intensity variant.
    #[must_use]
    pub fn new(variant: SheppLoganVariant) -> Self {
        let intensities = match variant {
            SheppLoganVariant::Original => &INTENSITY_ORIGINAL,
            SheppLoganVariant::Modified => &INTENSITY_MODIFIED,
        };
        let ellipses = GEOMETRY
            .iter()
            .zip(intensities)
            .map(|(&(a, b, x0, y0, phi_deg), &intensity)| Ellipse {
                intensity,
                a,
                b,
                x0,
                y0,
                phi: phi_deg.to_radians(),
            })
            .collect();
        Self { ellipses }
    }

    /// Original Shepp & Logan (1974) phantom.
    #[must_use]
    pub fn original() -> Self {
        Self::new(SheppLoganVariant::Original)
    }

    /// Modified Shepp–Logan (Toft 1996) phantom.
    #[must_use]
    pub fn modified() -> Self {
        Self::new(SheppLoganVariant::Modified)
    }

    /// The ten ellipses.
    #[must_use]
    pub fn ellipses(&self) -> &[Ellipse] {
        &self.ellipses
    }

    /// Phantom value at `(x, y)`: sum of the intensities of the containing ellipses.
    #[must_use]
    pub fn value_at(&self, x: f64, y: f64) -> f64 {
        self.ellipses
            .iter()
            .filter(|e| e.contains(x, y))
            .map(|e| e.intensity)
            .sum()
    }

    /// Rasterize to an `n × n` image over `[-1, 1] × [-1, 1]`.
    ///
    /// Row index increases with `y` from `−1` (row 0) to `+1` (row `n−1`); column
    /// index increases with `x`. Pixel centres are sampled.
    #[must_use]
    pub fn rasterize(&self, n: usize) -> Array2<f64> {
        Array2::from_shape_fn([n, n], |[row, col]| {
            // Pixel-centre coordinates in [-1, 1].
            let x = 2.0 * (col as f64 + 0.5) / n as f64 - 1.0;
            let y = 2.0 * (row as f64 + 0.5) / n as f64 - 1.0;
            self.value_at(x, y)
        })
    }
}

#[cfg(test)]
mod tests;
