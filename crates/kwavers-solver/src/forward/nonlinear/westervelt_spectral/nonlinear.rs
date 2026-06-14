//! Nonlinear term computation for Westervelt equation

use kwavers_grid::Grid;
use kwavers_medium::Medium;
use ndarray::{Array3, Zip};

/// Compute the nonlinear term for the Westervelt equation
///
/// The Westervelt nonlinear term is: (β/ρc⁴) * ∂²(p²)/∂t²
/// Where ∂²(p²)/∂t² = 2p * ∂²p/∂t² + 2(∂p/∂t)²
pub fn compute_nonlinear_term(
    pressure: &Array3<f64>,
    prev_pressure: &Array3<f64>,
    pressure_history: Option<&Array3<f64>>,
    medium: &dyn Medium,
    grid: &Grid,
    dt: f64,
) -> Array3<f64> {
    let mut nonlinear_term = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    compute_nonlinear_term_into(
        &mut nonlinear_term,
        pressure,
        prev_pressure,
        pressure_history,
        medium,
        grid,
        dt,
    );
    nonlinear_term
}

/// Fill a caller-owned Westervelt nonlinear-term workspace.
///
/// # Theorem — Westervelt nonlinear term derivation
///
/// The Westervelt equation in operator form (Hamilton & Blackstock 1998, Eq. 4.1.8):
/// ```text
/// ∇²p − (1/c²)∂²p/∂t² = −(β/ρc⁴)∂²(p²)/∂t² − δ∇²(∂p/∂t) − Q
/// ```
/// Rearranging for the leapfrog explicit form `∂²p/∂t²`:
/// ```text
/// ∂²p/∂t² = c²∇²p + (β/ρc²)∂²(p²)/∂t² + c²δ∇²(∂p/∂t) + c²Q
/// ```
/// This function returns the nonlinear contribution `(β/ρc²)∂²(p²)/∂t²` (positive).
///
/// Product rule: `∂²(p²)/∂t² = 2p·∂²p/∂t² + 2(∂p/∂t)²`.
///
/// Evaluating pointwise into `output` is algebraically identical to returning a
/// newly allocated array because every voxel depends only on collocated pressure
/// history and medium values. Boundary voxels remain zero.
pub(super) fn compute_nonlinear_term_into(
    output: &mut Array3<f64>,
    pressure: &Array3<f64>,
    prev_pressure: &Array3<f64>,
    pressure_history: Option<&Array3<f64>>,
    medium: &dyn Medium,
    grid: &Grid,
    dt: f64,
) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    output.fill(0.0);
    // Get spatially varying medium properties
    let rho_arr = medium.density_array();
    let c_arr = medium.sound_speed_array();

    Zip::indexed(output)
        .and(pressure)
        .and(prev_pressure)
        .for_each(|(i, j, k), nl_val, &p_curr, &p_prev| {
            if i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1 {
                let rho = rho_arr[[i, j, k]].max(1e-9);
                let c = c_arr[[i, j, k]].max(1e-9);

                // Get spatially-varying nonlinearity coefficient
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let beta = kwavers_medium::AcousticProperties::nonlinearity_coefficient(
                    medium, x, y, z, grid,
                );

                // Explicit-form nonlinear coefficient for leapfrog:
                // ∂²p/∂t² = c²∇²p + (β/ρc²)∂²(p²)/∂t² + …
                // Coefficient is positive; the ∂²(p²)/∂t² factor is positive for
                // forward-propagating waves with positive nonlinearity parameter.
                let nonlinear_coeff = beta / (rho * c.powi(2));

                let term = if let Some(p_history) = pressure_history {
                    // Full second-order accuracy with pressure history
                    let p_prev_prev = p_history[[i, j, k]];
                    let d2p_dt2 = (2.0f64.mul_add(-p_prev, p_curr) + p_prev_prev) / (dt * dt);
                    let dp_dt = (p_curr - p_prev) / dt;

                    // Westervelt nonlinear term
                    let p_squared_second_deriv =
                        (2.0 * p_curr).mul_add(d2p_dt2, 2.0 * dp_dt.powi(2));
                    nonlinear_coeff * p_squared_second_deriv
                } else {
                    // Bootstrap for first iteration
                    let dp_dt = (p_curr - p_prev) / dt.max(1e-12);
                    let d2p_dt2_bootstrap = dp_dt / dt.max(1e-12);

                    let p_squared_second_deriv =
                        (2.0 * p_curr).mul_add(d2p_dt2_bootstrap, 2.0 * dp_dt.powi(2));
                    nonlinear_coeff * p_squared_second_deriv
                };

                *nl_val = term;
            }
        });
}

/// Fill a caller-owned viscoelastic damping workspace.
///
/// # Theorem — viscoelastic (Stokes) damping term derivation
///
/// The thermoviscous Westervelt equation in explicit `∂²p/∂t²` form:
/// ```text
/// ∂²p/∂t² = c²∇²p + (β/ρc²)∂²(p²)/∂t² + δ∇²(∂p/∂t) + …
/// ```
/// where `δ = (4η_s/3 + η_b)/ρ` [m²/s] is the diffusivity of sound. The damping
/// term carries **no** `c²` factor: it is dimensionally `Pa/s²`
/// (`δ[m²/s]·∇²[1/m²]·∂ₜp[Pa/s]`), and it yields the classical absorption
/// `α(ω) = δω²/2c³` (temporal amplitude decay rate `δk²/2` per mode `k`).
///
/// This function returns `δ∇²(∂p/∂t)` — the full contribution to `∂²p/∂t²`.
///
/// # Stencil identity (zero-copy theorem)
/// Linearity of the discrete Laplacian implies
/// `∇²((pⁿ−pⁿ⁻¹)/dt) = (∇²pⁿ − ∇²pⁿ⁻¹)/dt`. Computing neighbour pressure
/// differences directly in the centered stencil gives the same result as
/// materializing `dp_dt` first, while removing one full-volume temporary.
pub(super) fn compute_viscoelastic_term_into(
    output: &mut Array3<f64>,
    pressure: &Array3<f64>,
    prev_pressure: &Array3<f64>,
    medium: &dyn Medium,
    grid: &Grid,
    dt: f64,
) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    output.fill(0.0);
    if nx < 3 || ny < 3 || nz < 3 {
        return;
    }

    let dx2_inv = 1.0 / (grid.dx * grid.dx);
    let dy2_inv = 1.0 / (grid.dy * grid.dy);
    let dz2_inv = 1.0 / (grid.dz * grid.dz);
    let inv_dt = 1.0 / dt;
    let rho_arr = medium.density_array();

    // PERIODIC FD Laplacian of ∂p/∂t (wrap-around neighbours). The main wave term
    // uses the periodic spectral Laplacian, so the domain is periodic; damping must
    // cover ALL points the same way. Skipping the domain boundaries (the previous
    // behaviour) left them undamped, and the periodic spectral operator then fed
    // that undamped boundary energy back into the interior — biasing absorption.
    for k in 0..nz {
        let kp = (k + 1) % nz;
        let km = (k + nz - 1) % nz;
        for j in 0..ny {
            let jp = (j + 1) % ny;
            let jm = (j + ny - 1) % ny;
            for i in 0..nx {
                let ip = (i + 1) % nx;
                let im = (i + nx - 1) % nx;
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let center = (pressure[[i, j, k]] - prev_pressure[[i, j, k]]) * inv_dt;
                let z_plus = (pressure[[i, j, kp]] - prev_pressure[[i, j, kp]]) * inv_dt;
                let z_minus = (pressure[[i, j, km]] - prev_pressure[[i, j, km]]) * inv_dt;
                let x_plus = (pressure[[ip, j, k]] - prev_pressure[[ip, j, k]]) * inv_dt;
                let x_minus = (pressure[[im, j, k]] - prev_pressure[[im, j, k]]) * inv_dt;
                let y_plus = (pressure[[i, jp, k]] - prev_pressure[[i, jp, k]]) * inv_dt;
                let y_minus = (pressure[[i, jm, k]] - prev_pressure[[i, jm, k]]) * inv_dt;
                let laplacian_dp_dt = (2.0f64.mul_add(-center, z_plus) + z_minus).mul_add(
                    dz2_inv,
                    (2.0f64.mul_add(-center, x_plus) + x_minus).mul_add(
                        dx2_inv,
                        (2.0f64.mul_add(-center, y_plus) + y_minus) * dy2_inv,
                    ),
                );

                let eta_s = medium.shear_viscosity(x, y, z, grid);
                let eta_b = medium.bulk_viscosity(x, y, z, grid);
                let rho = rho_arr[[i, j, k]].max(1e-9);

                // Diffusivity of sound δ = (4η_s/3 + η_b)/ρ [m²/s]. The Stokes
                // damping term is δ·∇²(∂p/∂t) — with NO c² factor: the term already
                // has units Pa/s² (δ[m²/s]·∇²[1/m²]·∂ₜp[Pa/s]), matching ∂²p/∂t².
                // A spurious c² here (≈2.25e6 for water) made the explicit damping
                // amplification ~c²·δ·dt·k² ≈ 89 at Nyquist → instability/NaN.
                let visc_coeff = (4.0 * eta_s / 3.0 + eta_b) / rho;

                output[[i, j, k]] = visc_coeff * laplacian_dp_dt;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_medium::{AcousticProperties, HomogeneousMedium};

    /// **Theorem**: Westervelt explicit-form nonlinear coefficient is `+β/(ρc²)`.
    ///
    /// Westervelt operator form: `∇²p − (1/c²)∂²p/∂t² = −(β/ρc⁴)∂²(p²)/∂t²`
    ///
    /// Rearranged for `∂²p/∂t²`:
    /// ```text
    /// ∂²p/∂t² = c²∇²p + (β/ρc²)∂²(p²)/∂t²
    /// ```
    ///
    /// Bootstrap (no history, `p_prev=0`):
    /// - `∂p/∂t ≈ (p_curr − 0) / dt = 2/0.5 = 4`
    /// - `∂²p/∂t² ≈ ∂p/∂t / dt = 8`
    /// - `∂²(p²)/∂t² = 2p·∂²p/∂t² + 2(∂p/∂t)² = 2·2·8 + 2·16 = 64`
    /// - `coeff = β/(ρc²) = β/(1000·1500²)` [positive]
    /// - `expected = coeff · 64` [positive]
    #[test]
    fn nonlinear_term_into_matches_product_rule_for_constant_history() {
        const RHO: f64 = kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
        const C: f64 = SOUND_SPEED_WATER_SIM;
        const DT: f64 = 0.5;
        const P_CURR: f64 = 2.0;

        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0).unwrap();
        let medium = HomogeneousMedium::from_minimal(RHO, C, &grid);
        let pressure = Array3::from_elem((3, 3, 3), P_CURR);
        let prev_pressure = Array3::zeros((3, 3, 3));
        let mut output = Array3::from_elem((3, 3, 3), 7.0);

        compute_nonlinear_term_into(
            &mut output,
            &pressure,
            &prev_pressure,
            None,
            &medium,
            &grid,
            DT,
        );

        // Analytical: explicit-form coefficient = +β/(ρc²)
        let beta = AcousticProperties::nonlinearity_coefficient(&medium, 1.0, 1.0, 1.0, &grid);
        let coeff = beta / (RHO * C.powi(2));

        // Bootstrap approximation used when prev_prev is absent:
        // dp_dt = p_curr / dt  (prev=0)
        // d2p_dt2 = dp_dt / dt
        let dp_dt = P_CURR / DT;
        let d2p_dt2_bootstrap = dp_dt / DT;
        let p_squared_second_deriv = (2.0 * P_CURR).mul_add(d2p_dt2_bootstrap, 2.0 * dp_dt.powi(2));
        let expected = coeff * p_squared_second_deriv;

        // Interior voxel [1,1,1] must match analytical formula.
        assert!(
            (output[[1, 1, 1]] - expected).abs() < 1e-20,
            "nl[1,1,1]={:.6e}, expected={:.6e} (coeff={:.6e})",
            output[[1, 1, 1]],
            expected,
            coeff
        );
        // Boundary voxel must remain zero.
        assert_eq!(output[[0, 1, 1]], 0.0, "boundary nl must be zero");
        // Verify sign: with positive β and positive p, the nonlinear term is positive.
        assert!(
            expected > 0.0,
            "Westervelt explicit-form NL term must be positive for β>0, p>0"
        );
    }

    /// **Theorem**: Stokes damping coefficient is `δ = (4η_s/3 + η_b)/ρ` — with
    /// NO `c²` factor (the term is dimensionally `Pa/s²`).
    ///
    /// Westervelt explicit form:
    /// ```text
    /// ∂²p/∂t² = c²∇²p + … + δ∇²(∂p/∂t)
    /// ```
    ///
    /// Field: `p = i² + 2j² + 3k²`; `∇²p = 2 + 4 + 6 = 12` at interior (Δx=1).
    /// `prev_pressure = 0` ⟹ `∂p/∂t = p/dt`; stencil identical to Laplacian of p.
    /// `∇²(∂p/∂t) = (∇²p)/dt = 12/dt`.
    /// `expected = δ·(12/dt)`.
    ///
    /// Viscosities are queried from the medium to avoid hardcoding defaults.
    #[test]
    fn viscoelastic_term_into_matches_direct_quadratic_laplacian() {
        use kwavers_medium::ViscousProperties;

        const RHO: f64 = kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
        const C: f64 = SOUND_SPEED_WATER_SIM;
        const DT: f64 = 0.25;

        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0).unwrap();
        let medium = HomogeneousMedium::from_minimal(RHO, C, &grid);
        let mut pressure = Array3::<f64>::zeros((5, 5, 5));
        for k in 0..5 {
            for j in 0..5 {
                for i in 0..5 {
                    pressure[[i, j, k]] =
                        (i as f64).powi(2) + 2.0 * (j as f64).powi(2) + 3.0 * (k as f64).powi(2);
                }
            }
        }
        let prev_pressure = Array3::zeros((5, 5, 5));
        let mut output = Array3::from_elem((5, 5, 5), 3.0);

        compute_viscoelastic_term_into(&mut output, &pressure, &prev_pressure, &medium, &grid, DT);

        // Query actual viscosities from the medium — do not hardcode defaults.
        let eta_s = medium.shear_viscosity(2.0, 2.0, 2.0, &grid);
        let eta_b = medium.bulk_viscosity(2.0, 2.0, 2.0, &grid);
        let visc_coeff = (4.0 * eta_s / 3.0 + eta_b) / RHO;
        let laplacian_at_center = 12.0_f64; // 2+4+6 for the quadratic field (Δx=1)
        // δ·∇²(∂p/∂t): NO c² factor.
        let expected = visc_coeff * (laplacian_at_center / DT);

        assert!(
            (output[[2, 2, 2]] - expected).abs() <= 1.0e-10,
            "damp[2,2,2]={:.10e}, expected={:.10e} (η_s={eta_s:.4e}, η_b={eta_b:.4e})",
            output[[2, 2, 2]],
            expected
        );
        // Boundary voxels are now damped too (periodic wrap-around stencil), so
        // they are no longer forced to zero. The interior check above validates the
        // coefficient; periodic-boundary correctness is covered by the Stokes tests.
    }
}
