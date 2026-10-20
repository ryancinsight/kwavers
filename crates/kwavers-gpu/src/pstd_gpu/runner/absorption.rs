//! Host construction of the GPU PSTD power-law absorption operators.

use crate::pstd_gpu::medium::{validate_active_power_law_exponent, PstdMediumSnapshot};
use eunomia::FloatElement;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
use kwavers_solver::forward::pstd::config::PowerLawDispersion;
use std::f64::consts::PI;

/// Wavenumber magnitude below which the fractional-Laplacian symbols are set
/// to zero: the DC bin, whose `|k|^(y−2)` is singular.
const SINGULARITY_THRESHOLD_RAD_M: f64 = 1e-8;

/// Flattened power-law absorption operators in the GPU's `f32` layout, one
/// value per voxel in `[ix][iy][iz]` order.
#[derive(Clone, Debug, PartialEq)]
pub struct PowerLawOperators {
    /// `|k|^(y−2)`, zero at the DC bin.
    pub nabla1: Vec<f32>,
    /// `|k|^(y−1)`, zero at the DC bin.
    pub nabla2: Vec<f32>,
    /// Absorbing coefficient `τ = −2 α₀ c₀^(y−1)`.
    pub tau: Vec<f32>,
    /// Dispersive coefficient `η = 2 α₀ c₀^y tan(πy/2)`; zero without
    /// dispersion.
    pub eta: Vec<f32>,
}

impl PowerLawOperators {
    /// Operators of a medium that does not absorb: every value zero.
    #[must_use]
    pub fn lossless(total: usize) -> Self {
        Self {
            nabla1: vec![0.0; total],
            nabla2: vec![0.0; total],
            tau: vec![0.0; total],
            eta: vec![0.0; total],
        }
    }
}

/// GPU PSTD power-law absorption operators for one exponent `y` and a
/// per-voxel coefficient in dB/(MHz^y cm), the one construction the Rust
/// runner and the Python session share.
///
/// `τ = −2 α₀ c₀^(y−1)` and `η = 2 α₀ c₀^y tan(πy/2)` (Treeby & Cox 2010,
/// Eqs. 19–20); without dispersion `η = 0`, which admits `y = 1`.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] when `c0_flat` does not hold one
/// sound speed per voxel, `y` is not finite, or `y = 1` with causal
/// dispersion, where `tan(πy/2)` is singular.
pub fn power_law_operators(
    grid: &Grid,
    exponent: f64,
    dispersion: PowerLawDispersion,
    c0_flat: &[f32],
    coefficient_db_at: impl Fn(usize) -> f64,
) -> KwaversResult<PowerLawOperators> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let total = c0_flat.len();
    if nx.checked_mul(ny).and_then(|xy| xy.checked_mul(nz)) != Some(total) {
        return Err(KwaversError::InvalidInput(format!(
            "GPU PSTD absorption needs one sound speed per voxel of the {nx}x{ny}x{nz} grid; got {total}"
        )));
    }
    validate_active_power_law_exponent(exponent, dispersion)?;

    let mut operators = PowerLawOperators::lossless(total);
    let (lx, ly, lz) = grid.physical_size();
    let (dk_x, dk_y, dk_z) = (TWO_PI / lx, TWO_PI / ly, TWO_PI / lz);
    let y = exponent;
    let dispersion_factor = match dispersion {
        PowerLawDispersion::Causal => (PI * y / 2.0).tan(),
        PowerLawDispersion::Omitted => 0.0,
    };
    // Signed FFT-order wavenumber magnitude along one axis of `n` bins.
    let axis_wavenumber = |index: usize, n: usize, dk: f64| {
        f64::from_count(if index <= n / 2 { index } else { n - index }) * dk
    };

    for (flat, &c0) in c0_flat.iter().enumerate() {
        let ix = flat / (ny * nz);
        let iy = (flat % (ny * nz)) / nz;
        let iz = flat % nz;
        let kix = axis_wavenumber(ix, nx, dk_x);
        let kiy = axis_wavenumber(iy, ny, dk_y);
        let kiz = axis_wavenumber(iz, nz, dk_z);
        let k_mag = (kix * kix + kiy * kiy + kiz * kiz).sqrt();
        if k_mag > SINGULARITY_THRESHOLD_RAD_M {
            operators.nabla1[flat] = f32::from_f64(k_mag.powf(y - 2.0));
            operators.nabla2[flat] = f32::from_f64(k_mag.powf(y - 1.0));
        }

        let alpha_0_si = power_law_db_cm_to_np_omega_m(coefficient_db_at(flat), y);
        let c0_local = f64::from(c0);
        operators.tau[flat] = f32::from_f64(-2.0 * alpha_0_si * c0_local.powf(y - 1.0));
        operators.eta[flat] =
            f32::from_f64(2.0 * alpha_0_si * c0_local.powf(y) * dispersion_factor);
    }
    Ok(operators)
}

/// The snapshot's operators: all zero when the medium does not absorb.
pub(super) fn absorption_operators(
    grid: &Grid,
    medium: &PstdMediumSnapshot,
) -> KwaversResult<PowerLawOperators> {
    let Some(absorption) = &medium.absorption else {
        return Ok(PowerLawOperators::lossless(medium.c0_flat.len()));
    };
    power_law_operators(
        grid,
        absorption.exponent,
        absorption.dispersion,
        &medium.c0_flat,
        |flat| absorption.coefficients.at(flat),
    )
}

#[cfg(test)]
mod tests {
    use super::{absorption_operators, power_law_operators, PowerLawOperators};
    use crate::pstd_gpu::medium::PstdMediumSnapshot;
    use kwavers_core::error::KwaversError;
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;
    use kwavers_solver::forward::pstd::config::PowerLawDispersion::{self, Causal, Omitted};

    fn grid() -> Grid {
        Grid::new(4, 4, 4, 1.0e-4, 1.0e-4, 1.0e-4).expect("grid")
    }

    fn operators(alpha_power: f64, dispersion: PowerLawDispersion) -> PowerLawOperators {
        let grid = grid();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
        let snapshot =
            PstdMediumSnapshot::from_medium(&grid, &medium, 0.5, alpha_power, dispersion)
                .expect("absorbing snapshot");
        absorption_operators(&grid, &snapshot).expect("operators")
    }

    /// Without dispersion the dispersive term is exactly zero, and the
    /// absorbing term and both symbols match the causal construction, so the
    /// mode switches only η.
    #[test]
    fn omitted_dispersion_zeroes_only_the_dispersive_term() {
        let omitted = operators(1.5, Omitted);
        let causal = operators(1.5, Causal);
        assert!(omitted.eta.iter().all(|&value| value == 0.0));
        assert_eq!(omitted.nabla1, causal.nabla1);
        assert_eq!(omitted.nabla2, causal.nabla2);
        assert_eq!(omitted.tau, causal.tau);
        // η = −τ·c₀·tan(πy/2) and tan(3π/4) = −1, so causal η = τ·c₀.
        for (&eta, &tau) in causal.eta.iter().zip(&causal.tau) {
            assert!(tau < 0.0);
            let expected = tau * 1500.0;
            // Two f32 roundings of the stored values and the f64 tan's
            // residual at 3π/4, far below one f32 ulp.
            assert!(
                (eta - expected).abs() <= 2.0 * f32::EPSILON * expected.abs(),
                "eta {eta} against {expected}"
            );
        }
    }

    /// A unit exponent without dispersion yields finite absorption with
    /// η = 0, where the causal form's `tan(π/2)` is singular.
    #[test]
    fn unit_exponent_without_dispersion_absorbs_finitely() {
        let unit = operators(1.0, Omitted);
        assert!(unit.eta.iter().all(|&value| value == 0.0));
        assert!(unit
            .tau
            .iter()
            .all(|&value| value.is_finite() && value < 0.0));
        // |k|^(y−1) = |k|^0 = 1 away from the DC bin.
        assert_eq!(unit.nabla2[0], 0.0);
        assert!(unit.nabla2[1..].iter().all(|&value| value == 1.0));
    }

    /// Each voxel's τ follows its own coefficient: α₀ is linear in the
    /// coefficient, so doubling it doubles τ exactly in f64, and the f32
    /// store of a doubled value is the doubled f32.
    #[test]
    fn spatial_coefficients_reach_their_own_voxels() {
        let c0 = vec![1500.0_f32; 64];
        let uniform = power_law_operators(&grid(), 1.5, Causal, &c0, |_| 0.5).expect("uniform");
        let spatial = power_law_operators(&grid(), 1.5, Causal, &c0, |flat| {
            if flat % 2 == 0 {
                0.5
            } else {
                1.0
            }
        })
        .expect("spatial");
        for (flat, (&single, &varied)) in uniform.tau.iter().zip(&spatial.tau).enumerate() {
            let scale = if flat % 2 == 0 { 1.0 } else { 2.0 };
            assert_eq!(varied, single * scale, "voxel {flat}");
        }
    }

    #[test]
    fn construction_rejects_a_causal_unit_exponent_and_a_short_field() {
        let c0 = vec![1500.0_f32; 64];
        let singular = power_law_operators(&grid(), 1.0, Causal, &c0, |_| 0.5).unwrap_err();
        assert!(
            matches!(&singular, KwaversError::InvalidInput(message) if message.contains("PowerLawDispersion::Omitted")),
            "{singular:?}"
        );
        let short = power_law_operators(&grid(), 1.5, Causal, &c0[..63], |_| 0.5).unwrap_err();
        assert!(
            matches!(&short, KwaversError::InvalidInput(message) if message.contains("got 63")),
            "{short:?}"
        );
    }
}
