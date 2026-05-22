//! Acoustic-pressure-to-membrane-tension conversion for sonogenetics.
//!
//! # Physics
//!
//! A thin spherical cell membrane under isotropic pressure loading obeys Laplace's law
//! for a thin spherical shell (Timoshenko & Woinowsky-Krieger 1959 §1.1):
//!
//!   σ_hoop · h = ΔP · R / 2
//!
//! In terms of membrane tension T [N/m] = σ_hoop · h [Pa·m]:
//!
//!   ΔT_membrane = ΔP · R / 2    [N/m]
//!
//! For sonogenetics the quasi-static pressure acting on the membrane is the acoustic
//! radiation pressure (Duck 1990 §4.2):
//!
//!   P_rad(x) = I(x) / c(x)     (Pa)   (progressive wave; Sarvazyan 2010 Eq. 3)
//!
//! Combining:
//!
//!   ΔT_membrane(x) = I(x) · R / (2 · c(x))    [N/m]
//!
//! # Canonical cell parameters (mammalian neuron soma)
//!
//! - Cell radius    R = 10 μm  (Hochmuth 2000; Duque 2023 supplementary)
//! - Membrane thickness h = 5 nm  (lipid bilayer; Engelman 2005)
//!
//! # References
//!
//! - Timoshenko, S.P. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*.
//!   McGraw-Hill.
//! - Duck, F.A. (1990). *Physical Properties of Tissue*. Academic Press.
//! - Hochmuth, R.M. (2000). Micropipette aspiration of living cells.
//!   *Journal of Biomechanics*, 33(1), 15-22.
//! - Hamill, O.P. & Martinac, B. (2001). Molecular basis of mechanotransduction in living cells.
//!   *Physiological Reviews*, 81(2), 685-740.
//! - Engelman, D.M. (2005). Membranes are more mosaic than fluid. *Nature*, 438, 578-580.
//! - Sarvazyan, A.P. et al. (2010). Acoustic radiation force — a review.
//!   *Curr. Med. Imaging Rev.*, 6(1), 15-25.
//! - Duque, M. et al. (2023). Sonogenetic control via MscL-G22S. *Science*, 380, 1084-1090.

use ndarray::{Array3, Zip};

/// Cell geometry and membrane parameters for the Laplace tension model.
///
/// Units are SI throughout.
#[derive(Debug, Clone)]
pub struct CellMembraneParams {
    /// Cell soma radius R (m).
    pub radius_m: f64,
    /// Lipid bilayer thickness h (m).
    pub thickness_m: f64,
}

impl Default for CellMembraneParams {
    /// Canonical mammalian neuron soma parameters (Hochmuth 2000; Engelman 2005).
    ///
    /// R = 10 μm, h = 5 nm.
    fn default() -> Self {
        Self {
            radius_m: 10.0e-6,   // 10 μm
            thickness_m: 5.0e-9, // 5 nm
        }
    }
}

impl CellMembraneParams {
    /// Returns `true` if radius and thickness are both strictly positive.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.radius_m > 0.0 && self.thickness_m > 0.0
    }
}

/// Compute per-voxel acoustic radiation pressure P_rad(x) = I(x)/c(x) (Pa).
///
/// # Formula
///
/// P_rad = I / c    (progressive wave; Sarvazyan 2010 Eq. 3)
///
/// # Arguments
///
/// - `intensity`   — time-averaged intensity I(x) [W/m²]
/// - `sound_speed` — per-voxel c(x) (m/s); voxels with c ≤ 0 yield 0
///
/// # Returns
///
/// Per-voxel radiation pressure (Pa).
#[must_use]
pub fn compute_radiation_pressure(
    intensity: &Array3<f64>,
    sound_speed: &Array3<f64>,
) -> Array3<f64> {
    let mut out = Array3::<f64>::zeros(intensity.dim());
    Zip::from(&mut out)
        .and(intensity)
        .and(sound_speed)
        .par_for_each(|p_rad, &i, &c| {
            *p_rad = if c > 0.0 { i / c } else { 0.0 };
        });
    out
}

/// Compute per-voxel membrane tension increment ΔT_membrane(x) [N/m].
///
/// # Formula
///
/// ΔT_membrane = I · R / (2 · c)
///
/// Derivation: P_rad = I/c; Laplace thin-shell ΔT = P_rad · R / 2.
///
/// # Arguments
///
/// - `intensity`   — time-averaged intensity I(x) [W/m²]
/// - `sound_speed` — per-voxel c(x) (m/s); voxels with c ≤ 0 yield 0
/// - `params`      — cell geometry (radius R)
///
/// # Panics
///
/// Panics in debug builds if `params.is_valid()` is false.
#[must_use]
pub fn compute_membrane_tension(
    intensity: &Array3<f64>,
    sound_speed: &Array3<f64>,
    params: &CellMembraneParams,
) -> Array3<f64> {
    debug_assert!(
        params.is_valid(),
        "CellMembraneParams must have positive radius and thickness"
    );
    let r = params.radius_m;
    let mut out = Array3::<f64>::zeros(intensity.dim());
    Zip::from(&mut out)
        .and(intensity)
        .and(sound_speed)
        .par_for_each(|t, &i, &c| {
            *t = if c > 0.0 { i * r / (2.0 * c) } else { 0.0 };
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use ndarray::Array3;

    /// Analytical reference:
    ///   I = 1e5 W/m², c = 1500 m/s, R = 10 μm
    ///   P_rad = 1e5 / 1500 = 66.667 Pa
    ///   ΔT = 66.667 * 10e-6 / 2 = 3.333e-4 N/m = 0.3333 mN/m
    #[test]
    fn test_membrane_tension_analytical() {
        let (nx, ny, nz) = (3, 3, 3);
        let intensity = Array3::from_elem((nx, ny, nz), 1.0e5_f64);
        let sound_speed = Array3::from_elem((nx, ny, nz), SOUND_SPEED_WATER_SIM);
        let params = CellMembraneParams {
            radius_m: 10.0e-6,
            thickness_m: 5.0e-9,
        };

        let tension = compute_membrane_tension(&intensity, &sound_speed, &params);

        let expected_p_rad = 1.0e5 / SOUND_SPEED_WATER_SIM;
        let expected_tension = expected_p_rad * 10.0e-6 / 2.0;

        for &v in tension.iter() {
            assert_relative_eq!(v, expected_tension, max_relative = 1e-12);
        }
    }

    /// HIFU-level intensity: I = 1e6 W/m², c = 1500 m/s, R = 10 μm → ΔT ≈ 3.333 mN/m
    /// This is within the half-activation range of MscL-G22S (T_half ≈ 4.7 mN/m).
    /// # Panics
    /// - Panics if assertion fails: `Expected tension {expected:.4e} N/m should be below T_half {mscl_t_half:.4e} N/m`.
    ///
    #[test]
    fn test_hifu_level_tension_in_channel_range() {
        let (nx, ny, nz) = (2, 2, 2);
        let intensity = Array3::from_elem((nx, ny, nz), 1.0e6_f64);
        let sound_speed = Array3::from_elem((nx, ny, nz), SOUND_SPEED_WATER_SIM);
        let params = CellMembraneParams::default();

        let tension = compute_membrane_tension(&intensity, &sound_speed, &params);

        // Expected: 1e6 / 1500 * 10e-6 / 2 = 3.333e-3 N/m = 3.333 mN/m
        let expected = 1.0e6 * 10.0e-6 / (2.0 * SOUND_SPEED_WATER_SIM);
        for &v in tension.iter() {
            assert_relative_eq!(v, expected, max_relative = 1e-12);
        }
        // Must be below MscL-G22S T_half (4.7 mN/m) but in activation range
        let mscl_t_half = 4.7e-3_f64;
        assert!(
            expected < mscl_t_half,
            "Expected tension {expected:.4e} N/m should be below T_half {mscl_t_half:.4e} N/m"
        );
    }

    /// Zero sound speed voxels produce zero output.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_zero_sound_speed_produces_zero() {
        let (nx, ny, nz) = (2, 2, 2);
        let intensity = Array3::from_elem((nx, ny, nz), 1.0e5_f64);
        let mut sound_speed = Array3::from_elem((nx, ny, nz), SOUND_SPEED_WATER_SIM);
        sound_speed[[0, 0, 0]] = 0.0;
        let params = CellMembraneParams::default();

        let tension = compute_membrane_tension(&intensity, &sound_speed, &params);
        assert_eq!(
            tension[[0, 0, 0]],
            0.0,
            "zero sound speed voxel must produce zero tension"
        );
    }

    /// Radiation pressure formula: P_rad = I/c.
    #[test]
    fn test_radiation_pressure_formula() {
        let (nx, ny, nz) = (2, 2, 2);
        let intensity = Array3::from_elem((nx, ny, nz), 3000.0_f64);
        let sound_speed = Array3::from_elem((nx, ny, nz), SOUND_SPEED_WATER_SIM);

        let p_rad = compute_radiation_pressure(&intensity, &sound_speed);

        for &v in p_rad.iter() {
            assert_relative_eq!(v, 2.0, max_relative = 1e-12);
        }
    }
}
