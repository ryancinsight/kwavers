//! Correction validation: field simulation and performance metrics
//!
//! ## Theory: Transcranial Ultrasound Focusing Metrics
//!
//! ### Focal Intensity
//! For an acoustic pressure field p(r), the time-averaged acoustic intensity is
//! (O'Neil 1949):
//! ```text
//!   I(r) = |p(r)|² / (2 ρ₀ c₀)   [W/m²]
//! ```
//! The field array stores `|p|²` (norm_sqr of complex pressure). Target point values
//! are recovered by trilinear interpolation to avoid discretization artifacts when
//! the physical target does not coincide with a grid node.
//!
//! ### Sidelobe Level (Peak Sidelobe Level, PSL)
//! The main lobe is defined as the −6 dB region (cells where `I ≥ I_peak / 4`).
//! The bounding box of this region delineates the main lobe extent.
//! The peak sidelobe level is:
//! ```text
//!   PSL = 10 · log10(I_sidelobe_peak / I_main_peak)   [dB]
//! ```
//! Reference: Zhu & Steinberg (1993), IEEE Trans. UFFC 40(6):726–737.
//!
//! ### Focal Spot Size — FWHM per Axis
//! The 3D Full Width at Half Maximum (FWHM) characterizes spatial resolution.
//! For each axis α ∈ {x, y, z}:
//! 1. Extract the 1D profile through the peak voxel along α.
//! 2. Find the leftmost and rightmost indices where `I ≥ 0.5 · I_peak`.
//! 3. `FWHM_α = (right_index − left_index) · Δα`.
//!
//! The scalar focal spot size stored in `CorrectionValidation` is the geometric
//! mean of the three FWHM values: `(FWHM_x · FWHM_y · FWHM_z)^(1/3)`, providing
//! a single isotropic focal extent measure.
//!
//! ## References
//! - O'Neil (1949). J. Acoust. Soc. Am. 21(5):516–526. (acoustic intensity)
//! - Zhu & Steinberg (1993). IEEE Trans. UFFC 40(6):726–737. (PSL definition)
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2):021314. (FWHM focal metrics)
//! - Press et al. (2007). Numerical Recipes, §3.6. (trilinear interpolation)

use super::phase_correction::{PhaseCorrection, TranscranialAberrationCorrection};
use crate::core::error::KwaversResult;
use ndarray::Array3;
use num_complex::Complex;

/// Validation results for aberration correction
#[derive(Debug)]
pub struct CorrectionValidation {
    /// Focal intensity (W/m²) via trilinear interpolation at target
    pub focal_intensity: f64,
    /// Peak sidelobe level (dB below main lobe, negative = below main lobe)
    pub sidelobe_level_db: f64,
    /// Geometric-mean FWHM focal spot size: `(FWHM_x · FWHM_y · FWHM_z)^(1/3)` (m)
    pub focal_spot_size: f64,
}

impl TranscranialAberrationCorrection {
    /// Validate correction performance against three metrics.
    pub fn validate_correction(
        &self,
        correction: &PhaseCorrection,
        skull_model: &ndarray::Array3<f64>,
        transducer_positions: &[[f64; 3]],
        target_point: &[f64; 3],
    ) -> KwaversResult<CorrectionValidation> {
        let corrected_field = self.simulate_corrected_field(
            correction,
            skull_model,
            transducer_positions,
            target_point,
        )?;

        let focal_intensity = self.calculate_focal_intensity(&corrected_field, target_point);
        let sidelobe_ratio = self.calculate_sidelobe_level(&corrected_field, target_point);
        let sidelobe_level_db = if sidelobe_ratio > 0.0 {
            10.0 * sidelobe_ratio.log10()
        } else {
            f64::NEG_INFINITY
        };

        Ok(CorrectionValidation {
            focal_intensity,
            sidelobe_level_db,
            focal_spot_size: self.calculate_focal_spot_size(&corrected_field, target_point),
        })
    }

    /// Simulate acoustic field with phase correction applied.
    fn simulate_corrected_field(
        &self,
        correction: &PhaseCorrection,
        _skull_model: &ndarray::Array3<f64>,
        transducer_positions: &[[f64; 3]],
        _target_point: &[f64; 3],
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut field = Array3::zeros((nx, ny, nz));

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * self.grid.dx;
                    let y = j as f64 * self.grid.dy;
                    let z = k as f64 * self.grid.dz;

                    let mut total_field = Complex::new(0.0, 0.0);

                    for (elem_idx, &elem_pos) in transducer_positions.iter().enumerate() {
                        let dx = x - elem_pos[0];
                        let dy = y - elem_pos[1];
                        let dz = z - elem_pos[2];
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                        if distance > 0.0 {
                            let phase = correction.phases.get(elem_idx).unwrap_or(&0.0);
                            let amplitude = correction.amplitudes.get(elem_idx).unwrap_or(&1.0);
                            let k_wave =
                                2.0 * std::f64::consts::PI * self.frequency / self.reference_speed;
                            let corrected_phase = k_wave * distance + phase;
                            let contribution = Complex::from_polar(*amplitude, corrected_phase);
                            total_field += contribution / distance;
                        }
                    }

                    field[[i, j, k]] = total_field.norm_sqr(); // store |p|²
                }
            }
        }

        Ok(field)
    }

    /// Calculate focal acoustic intensity at the target using trilinear interpolation.
    ///
    /// ## Theorem: Trilinear Interpolation of Acoustic Intensity
    /// For a pressure-squared field `field[i,j,k] = |p(i,j,k)|²`, the intensity
    /// at a sub-voxel point r = (x, y, z) is obtained by trilinear interpolation:
    /// ```text
    ///   I(r) = TRILINEAR(field, x/Δx, y/Δy, z/Δz) / (2 ρ₀ c₀)
    /// ```
    ///
    /// ## Algorithm
    /// 1. Convert physical coordinates to fractional grid indices (xi, yj, zk).
    /// 2. Clamp to [0, N−2] so the upper corner is always valid.
    /// 3. Perform 3D trilinear interpolation over the enclosing unit cell.
    /// 4. Divide by 2ρ₀c₀ (ρ₀ = 1000 kg/m³, c₀ = reference_speed).
    ///
    /// ## References
    /// - O'Neil (1949). J. Acoust. Soc. Am. 21(5):516–526.
    /// - Press et al. (2007). Numerical Recipes §3.6.
    fn calculate_focal_intensity(&self, field: &Array3<f64>, target_point: &[f64; 3]) -> f64 {
        let (nx, ny, nz) = field.dim();
        let xi = (target_point[0] / self.grid.dx).clamp(0.0, nx.saturating_sub(2) as f64);
        let yj = (target_point[1] / self.grid.dy).clamp(0.0, ny.saturating_sub(2) as f64);
        let zk = (target_point[2] / self.grid.dz).clamp(0.0, nz.saturating_sub(2) as f64);

        let p2_interp = Self::trilinear_interpolate(field, xi, yj, zk);
        let rho0 = 1000.0_f64; // kg/m³ water reference density
        p2_interp / (2.0 * rho0 * self.reference_speed)
    }

    /// Trilinear interpolation at fractional grid position (xi, yj, zk).
    ///
    /// ## Precondition
    /// `0 ≤ xi ≤ nx−2`, `0 ≤ yj ≤ ny−2`, `0 ≤ zk ≤ nz−2` (caller must clamp).
    pub(crate) fn trilinear_interpolate(field: &Array3<f64>, xi: f64, yj: f64, zk: f64) -> f64 {
        let i0 = xi.floor() as usize;
        let j0 = yj.floor() as usize;
        let k0 = zk.floor() as usize;
        let tx = xi - xi.floor();
        let ty = yj - yj.floor();
        let tz = zk - zk.floor();

        let f000 = field[[i0, j0, k0]];
        let f100 = field[[i0 + 1, j0, k0]];
        let f010 = field[[i0, j0 + 1, k0]];
        let f001 = field[[i0, j0, k0 + 1]];
        let f110 = field[[i0 + 1, j0 + 1, k0]];
        let f101 = field[[i0 + 1, j0, k0 + 1]];
        let f011 = field[[i0, j0 + 1, k0 + 1]];
        let f111 = field[[i0 + 1, j0 + 1, k0 + 1]];

        let fx00 = f000 * (1.0 - tx) + f100 * tx;
        let fx10 = f010 * (1.0 - tx) + f110 * tx;
        let fx01 = f001 * (1.0 - tx) + f101 * tx;
        let fx11 = f011 * (1.0 - tx) + f111 * tx;

        let fxy0 = fx00 * (1.0 - ty) + fx10 * ty;
        let fxy1 = fx01 * (1.0 - ty) + fx11 * ty;

        fxy0 * (1.0 - tz) + fxy1 * tz
    }

    /// Calculate peak sidelobe level relative to the main lobe (linear ratio).
    ///
    /// ## Theorem: 6 dB Main Lobe Exclusion
    /// The main lobe is defined as the −6 dB region: all cells where
    /// `I ≥ I_peak / 4`. The bounding box of this region is excluded from the
    /// sidelobe search. The returned value is the linear ratio `I_side / I_peak`;
    /// caller converts to dB via `10·log10(ratio)`.
    ///
    /// ## Algorithm
    /// 1. `I_peak = max(field)`.
    /// 2. `T = I_peak / 4` (−6 dB threshold).
    /// 3. Compute bounding box of cells ≥ T.
    /// 4. Return `max(field outside bounding box) / I_peak`.
    ///
    /// ## References
    /// - Zhu & Steinberg (1993). IEEE Trans. UFFC 40(6):726–737.
    fn calculate_sidelobe_level(&self, field: &Array3<f64>, _target_point: &[f64; 3]) -> f64 {
        let (nx, ny, nz) = field.dim();

        let i_peak = field.iter().cloned().fold(0.0_f64, f64::max);
        if i_peak <= 0.0 {
            return 0.0;
        }

        let threshold_6db = i_peak * 0.25;

        let mut i_min = nx;
        let mut i_max = 0_usize;
        let mut j_min = ny;
        let mut j_max = 0_usize;
        let mut k_min = nz;
        let mut k_max = 0_usize;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if field[[i, j, k]] >= threshold_6db {
                        if i < i_min { i_min = i; }
                        if i > i_max { i_max = i; }
                        if j < j_min { j_min = j; }
                        if j > j_max { j_max = j; }
                        if k < k_min { k_min = k; }
                        if k > k_max { k_max = k; }
                    }
                }
            }
        }

        let mut max_sidelobe = 0.0_f64;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if i < i_min || i > i_max || j < j_min || j > j_max || k < k_min || k > k_max {
                        let v = field[[i, j, k]];
                        if v > max_sidelobe { max_sidelobe = v; }
                    }
                }
            }
        }

        max_sidelobe / i_peak
    }

    /// Calculate the geometric-mean FWHM focal spot size (metres).
    ///
    /// ## Theorem: FWHM Resolution Metric
    /// For each axis α ∈ {x, y, z}, the FWHM of the 1D profile through the peak
    /// voxel is:
    /// ```text
    ///   FWHM_α = (last_half_max_index − first_half_max_index) · Δα   [m]
    /// ```
    /// The scalar focal spot size is the geometric mean of the three FWHM values:
    /// ```text
    ///   FWHM_geom = (FWHM_x · FWHM_y · FWHM_z)^(1/3)   [m]
    /// ```
    ///
    /// ## Algorithm
    /// 1. Find peak voxel (pi, pj, pk) and half-maximum H = I_peak / 2.
    /// 2. For each axis: extract 1D profile, find first/last index ≥ H.
    /// 3. `FWHM_α = (last − first) · Δα`; 0 if the profile never reaches H.
    /// 4. Return geometric mean (or max if any axis is zero).
    ///
    /// ## References
    /// - Treeby & Cox (2010). J. Biomed. Opt. 15(2):021314.
    /// - Goodman (2005). Introduction to Fourier Optics §6.2.
    fn calculate_focal_spot_size(&self, field: &Array3<f64>, _target_point: &[f64; 3]) -> f64 {
        let (nx, ny, nz) = field.dim();

        let i_peak = field.iter().cloned().fold(0.0_f64, f64::max);
        if i_peak <= 0.0 {
            return 0.0;
        }
        let half_max = i_peak * 0.5;

        // Find peak voxel indices.
        let mut pi = 0_usize;
        let mut pj = 0_usize;
        let mut pk = 0_usize;
        'find_peak: for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if field[[i, j, k]] >= i_peak * (1.0 - 1e-12) {
                        pi = i;
                        pj = j;
                        pk = k;
                        break 'find_peak;
                    }
                }
            }
        }

        // FWHM along a 1D profile (scalar slice, delta = grid spacing).
        let fwhm_1d = |profile: &[f64], delta: f64| -> f64 {
            let first = profile.iter().position(|&v| v >= half_max);
            let last = profile.iter().rposition(|&v| v >= half_max);
            match (first, last) {
                (Some(f), Some(l)) if l >= f => (l - f) as f64 * delta,
                _ => 0.0,
            }
        };

        let profile_x: Vec<f64> = (0..nx).map(|i| field[[i, pj, pk]]).collect();
        let profile_y: Vec<f64> = (0..ny).map(|j| field[[pi, j, pk]]).collect();
        let profile_z: Vec<f64> = (0..nz).map(|k| field[[pi, pj, k]]).collect();

        let fwhm_x = fwhm_1d(&profile_x, self.grid.dx);
        let fwhm_y = fwhm_1d(&profile_y, self.grid.dy);
        let fwhm_z = fwhm_1d(&profile_z, self.grid.dz);

        if fwhm_x > 0.0 && fwhm_y > 0.0 && fwhm_z > 0.0 {
            (fwhm_x * fwhm_y * fwhm_z).cbrt()
        } else {
            fwhm_x.max(fwhm_y).max(fwhm_z)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use ndarray::Array3;

    fn make_correction() -> TranscranialAberrationCorrection {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        TranscranialAberrationCorrection::new(&grid).unwrap()
    }

    /// 3D Gaussian intensity field centred at fractional grid coordinates (cx, cy, cz).
    fn gaussian_field(
        nx: usize,
        ny: usize,
        nz: usize,
        cx: f64,
        cy: f64,
        cz: f64,
        sigma: f64,
    ) -> Array3<f64> {
        let mut f = Array3::zeros((nx, ny, nz));
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let r2 = (i as f64 - cx).powi(2)
                        + (j as f64 - cy).powi(2)
                        + (k as f64 - cz).powi(2);
                    f[[i, j, k]] = (-r2 / (2.0 * sigma * sigma)).exp();
                }
            }
        }
        f
    }

    #[test]
    fn test_trilinear_at_grid_node() {
        // At an exact integer grid node trilinear equals the grid value.
        let field = gaussian_field(8, 8, 8, 4.0, 4.0, 4.0, 1.5);
        let result =
            TranscranialAberrationCorrection::trilinear_interpolate(&field, 3.0, 3.0, 3.0);
        assert!((result - field[[3, 3, 3]]).abs() < 1e-12);
    }

    #[test]
    fn test_trilinear_midpoint_is_average() {
        // Between two nodes with values 2 and 4, midpoint should be 3.
        let mut field = Array3::zeros((4, 4, 4));
        field[[0, 0, 0]] = 2.0;
        field[[1, 0, 0]] = 4.0;
        let result =
            TranscranialAberrationCorrection::trilinear_interpolate(&field, 0.5, 0.0, 0.0);
        assert!((result - 3.0).abs() < 1e-12, "midpoint should be 3.0, got {result}");
    }

    #[test]
    fn test_focal_intensity_positive() {
        let correction = make_correction();
        let field = gaussian_field(32, 32, 32, 16.0, 16.0, 16.0, 3.0);
        let target = [16e-3_f64, 16e-3, 16e-3];
        let intensity = correction.calculate_focal_intensity(&field, &target);
        assert!(intensity > 0.0, "focal intensity must be positive for non-zero field");
    }

    #[test]
    fn test_focal_intensity_zero_field() {
        let correction = make_correction();
        let field = Array3::<f64>::zeros((32, 32, 32));
        let target = [16e-3_f64, 16e-3, 16e-3];
        assert_eq!(correction.calculate_focal_intensity(&field, &target), 0.0);
    }

    #[test]
    fn test_sidelobe_zero_for_uniform_field() {
        // Uniform field: entire domain is the main lobe bounding box → no sidelobe.
        let correction = make_correction();
        let field = Array3::from_elem((32, 32, 32), 1.0);
        let target = [16e-3_f64, 16e-3, 16e-3];
        let ratio = correction.calculate_sidelobe_level(&field, &target);
        assert_eq!(ratio, 0.0, "uniform field has no sidelobe outside bounding box");
    }

    #[test]
    fn test_sidelobe_less_than_main_lobe_for_gaussian() {
        let correction = make_correction();
        let field = gaussian_field(32, 32, 32, 16.0, 16.0, 16.0, 2.0);
        let target = [16e-3_f64, 16e-3, 16e-3];
        let ratio = correction.calculate_sidelobe_level(&field, &target);
        assert!(ratio < 1.0, "sidelobe ratio must be < 1 for Gaussian field; got {ratio}");
    }

    #[test]
    fn test_focal_spot_size_matches_gaussian_fwhm() {
        // For a Gaussian with σ cells, FWHM = 2√(2·ln2)·σ ≈ 2.355·σ cells.
        let correction = make_correction();
        let sigma_cells = 3.0_f64;
        let dx = 1e-3_f64;
        let field = gaussian_field(32, 32, 32, 16.0, 16.0, 16.0, sigma_cells);
        let target = [16e-3_f64, 16e-3, 16e-3];
        let spot = correction.calculate_focal_spot_size(&field, &target);
        let expected_fwhm = 2.0 * (2.0_f64 * std::f64::consts::LN_2).sqrt() * sigma_cells * dx;
        // Allow ±2 cells of tolerance for discrete sampling
        let tol = 2.0 * dx;
        assert!(
            (spot - expected_fwhm).abs() <= tol,
            "FWHM {spot:.4e} m, expected ≈ {expected_fwhm:.4e} m (±{tol:.1e})"
        );
    }

    #[test]
    fn test_focal_spot_size_zero_field() {
        let correction = make_correction();
        let field = Array3::<f64>::zeros((32, 32, 32));
        let target = [0.0_f64, 0.0, 0.0];
        assert_eq!(correction.calculate_focal_spot_size(&field, &target), 0.0);
    }
}
