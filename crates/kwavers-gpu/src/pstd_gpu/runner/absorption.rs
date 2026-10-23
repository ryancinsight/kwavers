//! Host construction of the GPU PSTD power-law absorption operators.

use crate::pstd_gpu::medium::PstdMediumSnapshot;
use eunomia::FloatElement;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_grid::Grid;
use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
use kwavers_solver::forward::pstd::config::PowerLawDispersion;
use std::f64::consts::PI;

/// Wavenumber magnitude below which the fractional-Laplacian symbols are set
/// to zero: the DC bin, whose `|k|^(y−2)` is singular.
const SINGULARITY_THRESHOLD_RAD_M: f64 = 1e-8;

/// Flattened `(|k|^(y−2), |k|^(y−1), τ, η)` in the GPU's `f32` layout, all
/// zero when the medium does not absorb.
///
/// `τ = −2 α₀ c₀^(y−1)` and `η = 2 α₀ c₀^y tan(πy/2)` (Treeby & Cox 2010,
/// Eqs. 19–20); without dispersion `η = 0`, which admits `y = 1`.
pub(super) fn absorption_operators(
    grid: &Grid,
    medium: &PstdMediumSnapshot,
    total: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut nabla1 = vec![0.0f32; total];
    let mut nabla2 = vec![0.0f32; total];
    let mut tau = vec![0.0f32; total];
    let mut eta = vec![0.0f32; total];
    let Some(absorption) = &medium.absorption else {
        return (nabla1, nabla2, tau, eta);
    };

    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let (lx, ly, lz) = grid.physical_size();
    let (dk_x, dk_y, dk_z) = (TWO_PI / lx, TWO_PI / ly, TWO_PI / lz);
    let y = absorption.exponent;
    let dispersion_factor = match absorption.dispersion {
        PowerLawDispersion::Causal => (PI * y / 2.0).tan(),
        PowerLawDispersion::Omitted => 0.0,
    };
    // Signed FFT-order wavenumber magnitude along one axis of `n` bins.
    let axis_wavenumber = |index: usize, n: usize, dk: f64| {
        f64::from_count(if index <= n / 2 { index } else { n - index }) * dk
    };

    for flat in 0..total {
        let ix = flat / (ny * nz);
        let iy = (flat % (ny * nz)) / nz;
        let iz = flat % nz;
        let kix = axis_wavenumber(ix, nx, dk_x);
        let kiy = axis_wavenumber(iy, ny, dk_y);
        let kiz = axis_wavenumber(iz, nz, dk_z);
        let k_mag = (kix * kix + kiy * kiy + kiz * kiz).sqrt();
        if k_mag > SINGULARITY_THRESHOLD_RAD_M {
            nabla1[flat] = f32::from_f64(k_mag.powf(y - 2.0));
            nabla2[flat] = f32::from_f64(k_mag.powf(y - 1.0));
        }

        let alpha_0_si = power_law_db_cm_to_np_omega_m(absorption.coefficients.at(flat), y);
        let c0_local = f64::from(medium.c0_flat[flat]);
        tau[flat] = f32::from_f64(-2.0 * alpha_0_si * c0_local.powf(y - 1.0));
        eta[flat] = f32::from_f64(2.0 * alpha_0_si * c0_local.powf(y) * dispersion_factor);
    }
    (nabla1, nabla2, tau, eta)
}

#[cfg(test)]
mod tests {
    use super::absorption_operators;
    use crate::pstd_gpu::medium::PstdMediumSnapshot;
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;
    use kwavers_solver::forward::pstd::config::PowerLawDispersion::{self, Causal, Omitted};

    fn operators(
        alpha_power: f64,
        dispersion: PowerLawDispersion,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let grid = Grid::new(4, 4, 4, 1.0e-4, 1.0e-4, 1.0e-4).expect("grid");
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
        let snapshot =
            PstdMediumSnapshot::from_medium(&grid, &medium, 0.5, alpha_power, dispersion)
                .expect("absorbing snapshot");
        absorption_operators(&grid, &snapshot, 64)
    }

    /// Without dispersion the dispersive term is exactly zero, and the
    /// absorbing term and both symbols match the causal construction, so the
    /// mode switches only η.
    #[test]
    fn omitted_dispersion_zeroes_only_the_dispersive_term() {
        let (nabla1, nabla2, tau, eta) = operators(1.5, Omitted);
        let (causal_nabla1, causal_nabla2, causal_tau, causal_eta) = operators(1.5, Causal);
        assert!(eta.iter().all(|&value| value == 0.0));
        assert_eq!(nabla1, causal_nabla1);
        assert_eq!(nabla2, causal_nabla2);
        assert_eq!(tau, causal_tau);
        // η = −τ·c₀·tan(πy/2) and tan(3π/4) = −1, so causal η = τ·c₀.
        for (&eta, &tau) in causal_eta.iter().zip(&causal_tau) {
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
        let (_, nabla2, tau, eta) = operators(1.0, Omitted);
        assert!(eta.iter().all(|&value| value == 0.0));
        assert!(tau.iter().all(|&value| value.is_finite() && value < 0.0));
        // |k|^(y−1) = |k|^0 = 1 away from the DC bin.
        assert_eq!(nabla2[0], 0.0);
        assert!(nabla2[1..].iter().all(|&value| value == 1.0));
    }
}
