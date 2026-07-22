//! Scan conversion: polar beam space → Cartesian display.
//!
//! Sector (phased) and convex-array probes acquire data along beams that fan out
//! at different angles. Each beam is a column of samples in `(range, angle)`
//! space; the display, however, is a rectangular Cartesian image. Scan
//! conversion resamples the polar beam grid onto Cartesian pixels by bilinear
//! interpolation.
//!
//! # Geometry
//!
//! Beams emanate from an apex. A beam at angle `θ` (measured from the axial `z`
//! axis) samples range `r = r₀ + i·Δr`, where `r₀` is the apex-to-aperture
//! radius (`0` for a sector phased array, `> 0` for a convex array). A Cartesian
//! pixel at `(x, z)` maps back to `r = √(x²+z²)`, `θ = atan2(x, z)`; if `(r, θ)`
//! falls inside the acquired fan it is bilinearly interpolated from the four
//! surrounding beam samples, otherwise it is background (`0`).
//!
//! # Reference
//! - Szabo, T. L. (2014). *Diagnostic Ultrasound Imaging: Inside Out* (2nd ed.),
//!   §10.4 (scan conversion). Academic Press.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array2, ArrayView2};

/// Polar acquisition geometry: uniformly-spaced beams, uniform range sampling.
#[derive(Debug, Clone, Copy)]
pub struct ScanGeometry {
    /// First beam angle `rad` (from the axial axis; negative = left of center).
    pub angle_min: f64,
    /// Angular spacing between beams `rad`.
    pub angle_step: f64,
    /// Apex-to-first-sample radius `r₀` `m` (0 for a sector phased array).
    pub radius_offset: f64,
    /// Range sampling step `Δr` `m`.
    pub range_step: f64,
}

/// Output Cartesian raster specification.
#[derive(Debug, Clone, Copy)]
pub struct CartesianGrid {
    /// Image width `pixels` (lateral, `x`).
    pub width: usize,
    /// Image height `pixels` (axial, `z`).
    pub height: usize,
    /// Lateral extent `[x_min, x_max]` `m`.
    pub x_range: (f64, f64),
    /// Axial extent `[z_min, z_max]` `m`.
    pub z_range: (f64, f64),
}

/// Sector/convex scan converter.
#[derive(Debug, Clone, Copy)]
pub struct ScanConverter {
    geometry: ScanGeometry,
    grid: CartesianGrid,
}

impl ScanConverter {
    /// Create a scan converter.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` for a non-positive range/angle
    /// step or a degenerate output grid.
    pub fn new(geometry: ScanGeometry, grid: CartesianGrid) -> KwaversResult<Self> {
        if geometry.range_step <= 0.0 || geometry.angle_step <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "scan geometry needs positive range and angle steps".to_owned(),
            ));
        }
        if grid.width < 2 || grid.height < 2 {
            return Err(KwaversError::InvalidInput(
                "Cartesian grid must be at least 2×2".to_owned(),
            ));
        }
        Ok(Self { geometry, grid })
    }

    /// Bilinear sample of the beam grid at fractional `(line, sample)`; returns
    /// `None` when outside the acquired fan.
    fn sample(beam: ArrayView2<f64>, line: f64, sample: f64) -> Option<f64> {
        let [n_lines, n_samples] = beam.shape();
        if line < 0.0 || sample < 0.0 {
            return None;
        }
        let l0 = line.floor() as usize;
        let s0 = sample.floor() as usize;
        if l0 + 1 >= n_lines || s0 + 1 >= n_samples {
            return None;
        }
        let fl = line - l0 as f64;
        let fs = sample - s0 as f64;
        let v = beam[[l0, s0]] * (1.0 - fl) * (1.0 - fs)
            + beam[[l0 + 1, s0]] * fl * (1.0 - fs)
            + beam[[l0, s0 + 1]] * (1.0 - fl) * fs
            + beam[[l0 + 1, s0 + 1]] * fl * fs;
        Some(v)
    }

    /// Convert polar `beam_data` `[n_lines, n_samples]` to a Cartesian image
    /// `[height, width]` (row-major, row = axial `z`, column = lateral `x`).
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` when `beam_data` has fewer than two
    /// beams or two samples (interpolation needs a 2×2 neighbourhood).
    pub fn convert(&self, beam_data: ArrayView2<f64>) -> KwaversResult<Array2<f64>> {
        let [n_lines, n_samples] = beam_data.shape();
        if n_lines < 2 || n_samples < 2 {
            return Err(KwaversError::InvalidInput(
                "scan conversion needs at least 2 beams and 2 samples".to_owned(),
            ));
        }
        let g = self.grid;
        let geo = self.geometry;
        let dx = (g.x_range.1 - g.x_range.0) / (g.width - 1) as f64;
        let dz = (g.z_range.1 - g.z_range.0) / (g.height - 1) as f64;
        let mut image = Array2::zeros((g.height, g.width));
        for row in 0..g.height {
            let z = g.z_range.0 + row as f64 * dz;
            for col in 0..g.width {
                let x = g.x_range.0 + col as f64 * dx;
                let r = z.hypot(x);
                let theta = x.atan2(z);
                let line = (theta - geo.angle_min) / geo.angle_step;
                let sample = (r - geo.radius_offset) / geo.range_step;
                if let Some(v) = Self::sample(beam_data, line, sample) {
                    image[[row, col]] = v;
                }
            }
        }
        Ok(image)
    }
}
