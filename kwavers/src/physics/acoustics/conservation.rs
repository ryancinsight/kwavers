//! Conservation law validation and entropy production for acoustic simulations.
//!
//! # Conservation Laws in Linear Acoustics
//!
//! The linearised equations of compressible fluid mechanics conserve energy,
//! mass, and momentum.  Verifying these conservation laws provides a
//! necessary (though not sufficient) check of numerical solver correctness.
//!
//! ## Acoustic Energy Density (Morse & Ingard 1968, §6.2)
//!
//! ```text
//!   e(x,t) = ½ρ₀|v|² + p²/(2ρ₀c₀²)
//! ```
//!
//! - First term: kinetic energy per unit volume.
//! - Second term: potential (compressional) energy per unit volume.
//!
//! Total energy: `E = ∫_V e dV`.
//!
//! ## Acoustic Intensity (Poynting Vector Analogue)
//!
//! ```text
//!   I(x,t) = p(x,t) · v(x,t)          [W/m²]
//! ```
//!
//! The acoustic Poynting theorem:
//! ```text
//!   ∂e/∂t + ∇·I = −2α c₀ e           [volumetric source = absorbed power]
//! ```
//!
//! ## Entropy Production (Second Law Compliance)
//!
//! ### Theorem (Acoustic Entropy Production — Hamilton & Blackstock 1998, §5.3)
//!
//! For a viscothermal acoustic medium with absorption coefficient α [Np/m]
//! at frequency f, the volumetric irreversible entropy production rate is:
//!
//! ```text
//!   σ_s(x,t) = 2α(x) c₀(x) e(x,t) / T₀
//! ```
//!
//! **Derivation.**
//! The power absorbed per unit volume from a progressive plane wave is
//! `ẇ_abs = 2α c₀ e` (Blackstock 2000, §12.4).  By the Second Law,
//! absorbed power in an irreversible process produces entropy at rate
//! `σ_s = ẇ_abs / T₀` (Landau & Lifshitz *Fluid Mechanics* §49).
//! Integrating over the domain:
//! ```text
//!   dS_irr/dt = ∫_V σ_s dV = ∫_V (2α c₀ e / T₀) dV ≥ 0
//! ```
//! Equality holds only when α = 0 (lossless medium), consistent with the
//! Second Law.  QED.
//!
//! **Corollary (Second Law Check).** The sign of `dS_irr/dt` provides a
//! necessary condition for physical validity: a simulation that produces
//! `dS_irr/dt < 0` is violating the Second Law and contains a numerical error.
//!
//! ### Acoustic-Thermal Energy Coupling
//!
//! The absorbed acoustic power `ẇ_abs = 2α c₀ e` appears as a volumetric heat
//! source in the bio-heat transfer equation:
//! ```text
//!   ρ c_p ∂T/∂t = ∇·(κ ∇T) + ẇ_abs = ∇·(κ ∇T) + 2αc₀ e
//! ```
//! This is the HIFU thermal dose model (Pennes 1948; Sapareto & Dewey 1984).
//!
//! # References
//!
//! - Morse PM, Ingard KU (1968). *Theoretical Acoustics*. McGraw-Hill. §6.2.
//! - Hamilton MF, Blackstock DT (1998). *Nonlinear Acoustics*. Academic Press. §5.3.
//! - Blackstock DT (2000). *Fundamentals of Physical Acoustics*. Wiley. §12.4.
//! - Landau LD, Lifshitz EM (1987). *Fluid Mechanics*, 2nd ed. Pergamon. §49.
//! - Pennes HH (1948). J. Appl. Physiol. 1(2), 93–122.
//! - Sapareto SA, Dewey WC (1984). Int. J. Radiat. Oncol. Biol. Phys. 10(6), 787–800.

use crate::domain::grid::Grid;
use ndarray::{Array3, Zip};

/// Conservation validation metrics for a single timestep.
#[derive(Debug, Clone)]
pub struct ConservationMetrics {
    /// Relative energy error: `|E(t) − E(0)| / E(0)`.
    pub energy_error: f64,
    /// Maximum pointwise mass continuity residual [kg m⁻³ s⁻¹].
    pub mass_error: f64,
    /// Maximum pointwise linearised momentum residual per axis [N m⁻³].
    pub momentum_error: (f64, f64, f64),
    /// Volumetric irreversible entropy production rate [W/K] (≥ 0 if physical).
    pub entropy_production_rate: f64,
    /// `true` if all errors are below the supplied tolerance.
    pub is_conserved: bool,
}

// ── Energy ────────────────────────────────────────────────────────────────────

/// Compute total acoustic energy and its relative error against `initial_energy`.
///
/// ## Formula
///
/// ```text
///   E = ∫_V [ ½ρ₀|v|² + p²/(2ρ₀c₀²) ] dV
/// ```
///
/// ## Arguments
///
/// * `pressure`       — pressure field [Pa]
/// * `velocity_{x,y,z}` — velocity components [m/s]
/// * `density`        — density field [kg/m³]
/// * `sound_speed`    — sound speed field [m/s]  (may be uniform)
/// * `initial_energy` — reference energy at t=0 [J]
/// * `grid`           — spatial grid
///
/// ## Returns
///
/// `|E(t) − E(0)| / E(0)` (dimensionless relative error).
///
/// ## Reference
///
/// Morse & Ingard (1968) §6.2, Eq. (6.2.4).
pub fn validate_energy_conservation(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    initial_energy: f64,
    grid: &Grid,
) -> f64 {
    let mut total_energy = 0.0_f64;
    let dv = grid.dx * grid.dy * grid.dz;

    Zip::from(pressure)
        .and(velocity_x)
        .and(velocity_y)
        .and(velocity_z)
        .and(density)
        .and(sound_speed)
        .for_each(|&p, &vx, &vy, &vz, &rho, &c| {
            if rho > 0.0 && c > 0.0 {
                let kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
                let c2 = c * c;
                let potential = p * p / (2.0 * rho * c2);
                total_energy += (kinetic + potential) * dv;
            }
        });

    (total_energy - initial_energy).abs() / initial_energy.max(1e-10)
}

// ── Mass ─────────────────────────────────────────────────────────────────────

/// Compute the maximum pointwise mass continuity residual.
///
/// ## Formula
///
/// ```text
///   residual(i,j,k) = ∂ρ/∂t + ∇·(ρv)   ≈ 0   (continuity equation)
/// ```
///
/// Both derivatives are approximated by second-order central differences
/// on the interior of the grid.  Boundary points are excluded.
pub fn validate_mass_conservation(
    density: &Array3<f64>,
    density_previous: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    dt: f64,
    grid: &Grid,
) -> f64 {
    let mut max_error = 0.0_f64;
    let dx_inv = 1.0 / grid.dx;
    let dy_inv = 1.0 / grid.dy;
    let dz_inv = 1.0 / grid.dz;
    let dt_inv = 1.0 / dt;

    for i in 1..grid.nx - 1 {
        for j in 1..grid.ny - 1 {
            for k in 1..grid.nz - 1 {
                let drho_dt = (density[[i, j, k]] - density_previous[[i, j, k]]) * dt_inv;

                let div_flux = (density[[i + 1, j, k]] * velocity_x[[i + 1, j, k]]
                    - density[[i - 1, j, k]] * velocity_x[[i - 1, j, k]])
                    * 0.5
                    * dx_inv
                    + (density[[i, j + 1, k]] * velocity_y[[i, j + 1, k]]
                        - density[[i, j - 1, k]] * velocity_y[[i, j - 1, k]])
                        * 0.5
                        * dy_inv
                    + (density[[i, j, k + 1]] * velocity_z[[i, j, k + 1]]
                        - density[[i, j, k - 1]] * velocity_z[[i, j, k - 1]])
                        * 0.5
                        * dz_inv;

                let error = (drho_dt + div_flux).abs();
                max_error = max_error.max(error);
            }
        }
    }

    max_error
}

// ── Momentum ──────────────────────────────────────────────────────────────────

/// Compute the maximum pointwise linearised momentum residual per axis.
///
/// ## Formula
///
/// ```text
///   ρ₀ ∂v/∂t + ∇p = 0        (linearised Euler equation, no viscosity)
/// ```
///
/// Each component is checked independently; central differences are used
/// for both the pressure gradient and the temporal derivative.
pub fn validate_momentum_conservation(
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    velocity_x_previous: &Array3<f64>,
    velocity_y_previous: &Array3<f64>,
    velocity_z_previous: &Array3<f64>,
    pressure: &Array3<f64>,
    density: &Array3<f64>,
    dt: f64,
    grid: &Grid,
) -> (f64, f64, f64) {
    let mut max_err_x = 0.0_f64;
    let mut max_err_y = 0.0_f64;
    let mut max_err_z = 0.0_f64;

    let dx_inv = 1.0 / grid.dx;
    let dy_inv = 1.0 / grid.dy;
    let dz_inv = 1.0 / grid.dz;
    let dt_inv = 1.0 / dt;

    for i in 1..grid.nx - 1 {
        for j in 1..grid.ny - 1 {
            for k in 1..grid.nz - 1 {
                let rho = density[[i, j, k]];

                let dvx_dt = (velocity_x[[i, j, k]] - velocity_x_previous[[i, j, k]]) * dt_inv;
                let dpx_dx = (pressure[[i + 1, j, k]] - pressure[[i - 1, j, k]]) * 0.5 * dx_inv;
                max_err_x = max_err_x.max((rho * dvx_dt + dpx_dx).abs());

                let dvy_dt = (velocity_y[[i, j, k]] - velocity_y_previous[[i, j, k]]) * dt_inv;
                let dpy_dy = (pressure[[i, j + 1, k]] - pressure[[i, j - 1, k]]) * 0.5 * dy_inv;
                max_err_y = max_err_y.max((rho * dvy_dt + dpy_dy).abs());

                let dvz_dt = (velocity_z[[i, j, k]] - velocity_z_previous[[i, j, k]]) * dt_inv;
                let dpz_dz = (pressure[[i, j, k + 1]] - pressure[[i, j, k - 1]]) * 0.5 * dz_inv;
                max_err_z = max_err_z.max((rho * dvz_dt + dpz_dz).abs());
            }
        }
    }

    (max_err_x, max_err_y, max_err_z)
}

// ── Entropy production ────────────────────────────────────────────────────────

/// Compute the volumetric irreversible entropy production rate [W/K].
///
/// ## Algorithm
///
/// For each grid voxel compute the local acoustic energy density
/// `e = ½ρ₀|v|² + p²/(2ρ₀c₀²)`, then accumulate:
///
/// ```text
///   dS_irr/dt = ∫_V σ_s dV
///             = ∫_V (2 α(x) c₀(x) e(x)) / T₀  dV
/// ```
///
/// ## Arguments
///
/// * `pressure`    — pressure field [Pa]
/// * `velocity_{x,y,z}` — velocity components [m/s]
/// * `density`     — density field [kg/m³]
/// * `sound_speed` — sound speed field [m/s]
/// * `absorption`  — absorption coefficient field [Np/m]
/// * `temperature` — ambient temperature [K]  (scalar T₀)
/// * `grid`        — spatial grid
///
/// ## Returns
///
/// `dS_irr/dt ≥ 0` [W/K].  A negative return value indicates a numerical
/// violation of the Second Law.
///
/// ## Reference
///
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics* §5.3;
/// Landau & Lifshitz (1987) *Fluid Mechanics* §49.
pub fn entropy_production_rate(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    absorption: &Array3<f64>,
    temperature: f64,
    grid: &Grid,
) -> f64 {
    debug_assert!(temperature > 0.0, "Temperature must be positive (Kelvin)");
    let dv = grid.dx * grid.dy * grid.dz;
    let t0_inv = 1.0 / temperature.max(1.0);

    let mut total = 0.0_f64;
    let (nx, ny, nz) = pressure.dim();

    // Use index loops: ndarray Zip supports at most 6 simultaneous arrays.
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let rho = density[[i, j, k]];
                let c = sound_speed[[i, j, k]];
                let alpha = absorption[[i, j, k]];
                if rho > 0.0 && c > 0.0 {
                    let p = pressure[[i, j, k]];
                    let vx = velocity_x[[i, j, k]];
                    let vy = velocity_y[[i, j, k]];
                    let vz = velocity_z[[i, j, k]];
                    let c2 = c * c;
                    let e =
                        0.5 * rho * (vx * vx + vy * vy + vz * vz) + p * p / (2.0 * rho * c2);
                    // σ_s = 2α c e / T₀   [W m⁻³ K⁻¹]
                    // Negative α (unphysical) produces negative total, flagging a Second Law violation.
                    total += 2.0 * alpha * c * e * t0_inv * dv;
                }
            }
        }
    }

    total
}

// ── Acoustic intensity (Poynting flux) ───────────────────────────────────────

/// Compute the acoustic intensity vector field `I = p · v` [W/m²].
///
/// Returns three `Array3<f64>` arrays: `(I_x, I_y, I_z)`.
///
/// ## Reference
///
/// Morse & Ingard (1968) §6.2, Eq. (6.2.9): the acoustic Poynting vector
/// represents the instantaneous acoustic power per unit area.
pub fn acoustic_intensity(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let ix = Zip::from(pressure)
        .and(velocity_x)
        .map_collect(|&p, &vx| p * vx);
    let iy = Zip::from(pressure)
        .and(velocity_y)
        .map_collect(|&p, &vy| p * vy);
    let iz = Zip::from(pressure)
        .and(velocity_z)
        .map_collect(|&p, &vz| p * vz);
    (ix, iy, iz)
}

/// Compute the total acoustic power through a z-plane at index `k_plane`.
///
/// `P_z = ∫∫ I_z(x,y,k_plane) dx dy`  [W]
pub fn acoustic_power_through_z_plane(
    pressure: &Array3<f64>,
    velocity_z: &Array3<f64>,
    k_plane: usize,
    grid: &Grid,
) -> f64 {
    let da = grid.dx * grid.dy;
    let mut power = 0.0_f64;
    let (nx, ny, _) = pressure.dim();
    for i in 0..nx {
        for j in 0..ny {
            power += pressure[[i, j, k_plane]] * velocity_z[[i, j, k_plane]] * da;
        }
    }
    power
}

// ── Acoustic-thermal coupling term ───────────────────────────────────────────

/// Compute the volumetric heat source from acoustic absorption [W/m³].
///
/// This is the acoustic-to-thermal coupling term in the bio-heat equation:
///
/// ```text
///   Q_abs(x,t) = 2 α(x) c₀(x) e(x,t)          [W/m³]
/// ```
///
/// ## Reference
///
/// - Pennes HH (1948). Analysis of tissue and arterial blood temperature
///   in the resting human forearm. J. Appl. Physiol. 1(2), 93–122.
/// - Sapareto SA, Dewey WC (1984). Thermal dose determination in cancer
///   therapy. Int. J. Radiat. Oncol. Biol. Phys. 10(6), 787–800.
pub fn acoustic_heat_source(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    absorption: &Array3<f64>,
) -> Array3<f64> {
    let (nx, ny, nz) = pressure.dim();
    let mut q = Array3::zeros((nx, ny, nz));
    // Use index loops: ndarray Zip supports at most 6 simultaneous arrays.
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let rho = density[[i, j, k]];
                let c = sound_speed[[i, j, k]];
                if rho > 0.0 && c > 0.0 {
                    let p = pressure[[i, j, k]];
                    let vx = velocity_x[[i, j, k]];
                    let vy = velocity_y[[i, j, k]];
                    let vz = velocity_z[[i, j, k]];
                    let alpha = absorption[[i, j, k]];
                    let c2 = c * c;
                    let e =
                        0.5 * rho * (vx * vx + vy * vy + vz * vz) + p * p / (2.0 * rho * c2);
                    q[[i, j, k]] = 2.0 * alpha * c * e;
                }
            }
        }
    }
    q
}

// ── Comprehensive validation ──────────────────────────────────────────────────

/// Run all conservation checks and return a consolidated `ConservationMetrics`.
///
/// `sound_speed` and `absorption` fields must have the same shape as the
/// pressure and velocity arrays.  Pass uniform arrays when the medium is
/// homogeneous.
#[allow(clippy::too_many_arguments)]
pub fn validate_conservation(
    pressure: &Array3<f64>,
    velocity_x: &Array3<f64>,
    velocity_y: &Array3<f64>,
    velocity_z: &Array3<f64>,
    density: &Array3<f64>,
    sound_speed: &Array3<f64>,
    absorption: &Array3<f64>,
    temperature: f64,
    _pressure_previous: Option<&Array3<f64>>,
    velocity_x_previous: Option<&Array3<f64>>,
    velocity_y_previous: Option<&Array3<f64>>,
    velocity_z_previous: Option<&Array3<f64>>,
    density_previous: Option<&Array3<f64>>,
    initial_energy: f64,
    dt: f64,
    grid: &Grid,
    tolerance: f64,
) -> ConservationMetrics {
    let energy_error = validate_energy_conservation(
        pressure,
        velocity_x,
        velocity_y,
        velocity_z,
        density,
        sound_speed,
        initial_energy,
        grid,
    );

    let mass_error = if let Some(rho_prev) = density_previous {
        validate_mass_conservation(
            density, rho_prev, velocity_x, velocity_y, velocity_z, dt, grid,
        )
    } else {
        0.0
    };

    let momentum_error = if let (Some(vx_prev), Some(vy_prev), Some(vz_prev)) = (
        velocity_x_previous,
        velocity_y_previous,
        velocity_z_previous,
    ) {
        validate_momentum_conservation(
            velocity_x, velocity_y, velocity_z, vx_prev, vy_prev, vz_prev, pressure, density, dt,
            grid,
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    let ds_dt = entropy_production_rate(
        pressure,
        velocity_x,
        velocity_y,
        velocity_z,
        density,
        sound_speed,
        absorption,
        temperature,
        grid,
    );

    let is_conserved = energy_error < tolerance
        && mass_error < tolerance
        && momentum_error.0 < tolerance
        && momentum_error.1 < tolerance
        && momentum_error.2 < tolerance
        && ds_dt >= 0.0; // Second Law compliance

    ConservationMetrics {
        energy_error,
        mass_error,
        momentum_error,
        entropy_production_rate: ds_dt,
        is_conserved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use ndarray::Array3;

    fn make_grid() -> Grid {
        Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap()
    }

    fn uniform_array(shape: (usize, usize, usize), val: f64) -> Array3<f64> {
        Array3::from_elem(shape, val)
    }

    // ── Energy ──────────────────────────────────────────────────────────────

    /// Zero velocity and zero pressure → zero energy → relative error = 0.
    ///
    /// With `initial_energy = 1e-12` (sentinel), the error is `|0 − 1e-12| / 1e-12 = 1.0`.
    /// But if `initial_energy = 0` the function uses the guard `max(e, 1e-10)`,
    /// so we set the initial energy to the computed value to get 0 error.
    #[test]
    fn test_zero_energy_field_has_zero_relative_error() {
        let grid = make_grid();
        let shape = (grid.nx, grid.ny, grid.nz);
        let p = uniform_array(shape, 0.0);
        let v = uniform_array(shape, 0.0);
        let rho = uniform_array(shape, 1000.0);
        let c = uniform_array(shape, 1500.0);

        // Initial energy = 0 → function uses max(0, 1e-10) = 1e-10 guard
        let err = validate_energy_conservation(&p, &v, &v, &v, &rho, &c, 0.0, &grid);
        assert!(err < 1e-8, "Zero field must give near-zero energy error: {err:.3e}");
    }

    /// A uniform pressure field with known analytical energy must match within 1e-10.
    ///
    /// ## Analytical result
    ///
    /// With `p = p₀` (uniform), `v = 0`, `ρ = ρ₀`, `c = c₀`:
    /// ```text
    ///   E = p₀²/(2ρ₀c₀²) × V
    /// ```
    /// For 8×8×8 grid with `dx = dy = dz = 1 mm`:
    /// `V = (0.008)³ = 5.12e-7 m³`
    #[test]
    fn test_energy_analytical_uniform_pressure() {
        let grid = make_grid();
        let shape = (grid.nx, grid.ny, grid.nz);
        let p0 = 1000.0_f64; // 1 kPa
        let rho0 = 1000.0_f64;
        let c0 = 1500.0_f64;

        let p = uniform_array(shape, p0);
        let v = uniform_array(shape, 0.0);
        let rho = uniform_array(shape, rho0);
        let c = uniform_array(shape, c0);

        let volume = (grid.nx as f64 * grid.dx)
            * (grid.ny as f64 * grid.dy)
            * (grid.nz as f64 * grid.dz);
        let e_analytical = p0 * p0 / (2.0 * rho0 * c0 * c0) * volume;

        // Pass analytical energy as initial → relative error must be zero
        let err = validate_energy_conservation(&p, &v, &v, &v, &rho, &c, e_analytical, &grid);
        assert!(
            err < 1e-10,
            "Energy relative error must be <1e-10 for exact initial value, got {err:.3e}"
        );
    }

    // ── Entropy production ───────────────────────────────────────────────────

    /// Entropy production with zero absorption must be zero.
    ///
    /// If α = 0 everywhere, `σ_s = 2 × 0 × c × e / T₀ = 0`.
    #[test]
    fn test_entropy_production_zero_absorption() {
        let grid = make_grid();
        let shape = (grid.nx, grid.ny, grid.nz);
        let p = uniform_array(shape, 1000.0);
        let v = uniform_array(shape, 0.0);
        let rho = uniform_array(shape, 1000.0);
        let c = uniform_array(shape, 1500.0);
        let alpha = uniform_array(shape, 0.0); // lossless

        let ds = entropy_production_rate(&p, &v, &v, &v, &rho, &c, &alpha, 310.0, &grid);
        assert!(
            ds.abs() < 1e-20,
            "Entropy production in lossless medium must be zero: {ds:.3e}"
        );
    }

    /// Entropy production must be non-negative (Second Law compliance).
    ///
    /// For any non-negative absorption coefficient α ≥ 0 and positive energy
    /// density e ≥ 0, `σ_s = 2α c e / T₀ ≥ 0`.
    #[test]
    fn test_entropy_production_non_negative() {
        let grid = make_grid();
        let shape = (grid.nx, grid.ny, grid.nz);
        let p = uniform_array(shape, 5000.0); // 5 kPa
        let v = uniform_array(shape, 0.1);
        let rho = uniform_array(shape, 1000.0);
        let c = uniform_array(shape, 1500.0);
        let alpha = uniform_array(shape, 2.0); // 2 Np/m

        let ds = entropy_production_rate(&p, &v, &v, &v, &rho, &c, &alpha, 310.0, &grid);
        assert!(ds >= 0.0, "Entropy production rate must be ≥ 0: {ds:.3e}");
    }

    /// Entropy production must scale linearly with absorption coefficient.
    ///
    /// Doubling α must exactly double `dS/dt`.
    #[test]
    fn test_entropy_production_scales_linearly_with_absorption() {
        let grid = make_grid();
        let shape = (grid.nx, grid.ny, grid.nz);
        let p = uniform_array(shape, 2000.0);
        let v = uniform_array(shape, 0.05);
        let rho = uniform_array(shape, 1000.0);
        let c = uniform_array(shape, 1500.0);
        let alpha1 = uniform_array(shape, 1.0);
        let alpha2 = uniform_array(shape, 2.0);

        let ds1 = entropy_production_rate(&p, &v, &v, &v, &rho, &c, &alpha1, 300.0, &grid);
        let ds2 = entropy_production_rate(&p, &v, &v, &v, &rho, &c, &alpha2, 300.0, &grid);

        let ratio = ds2 / ds1;
        assert!(
            (ratio - 2.0).abs() < 1e-10,
            "Entropy production must scale linearly with α: ratio={ratio:.6}, expected 2.0"
        );
    }

    /// Analytical entropy production for a uniform field with absorption.
    ///
    /// ## Analytical result
    ///
    /// With `p = p₀`, `v = 0`, `ρ = ρ₀`, `c = c₀`, `α = α₀`:
    /// ```text
    ///   e = p₀²/(2ρ₀c₀²)
    ///   dS/dt = 2 α₀ c₀ e / T₀ × V
    ///         = α₀ p₀² / (ρ₀ c₀ T₀) × V
    /// ```
    #[test]
    fn test_entropy_production_analytical() {
        let grid = make_grid();
        let shape = (grid.nx, grid.ny, grid.nz);
        let p0 = 2000.0_f64; // 2 kPa
        let rho0 = 1000.0_f64;
        let c0 = 1500.0_f64;
        let alpha0 = 3.0_f64; // Np/m
        let t0 = 310.0_f64; // K

        let p = uniform_array(shape, p0);
        let v = uniform_array(shape, 0.0);
        let rho = uniform_array(shape, rho0);
        let c = uniform_array(shape, c0);
        let alpha = uniform_array(shape, alpha0);

        let volume = (grid.nx as f64 * grid.dx)
            * (grid.ny as f64 * grid.dy)
            * (grid.nz as f64 * grid.dz);
        let expected = alpha0 * p0 * p0 / (rho0 * c0 * t0) * volume;

        let ds = entropy_production_rate(&p, &v, &v, &v, &rho, &c, &alpha, t0, &grid);

        let rel = (ds - expected).abs() / expected;
        assert!(
            rel < 1e-10,
            "Analytical entropy production: expected {expected:.6e} W/K, \
             got {ds:.6e} W/K, rel={rel:.3e}"
        );
    }

    // ── Acoustic intensity ───────────────────────────────────────────────────

    /// `acoustic_intensity` must compute `I = p × v` component-wise.
    #[test]
    fn test_acoustic_intensity_pointwise() {
        let grid = make_grid();
        let shape = (grid.nx, grid.ny, grid.nz);
        let p = uniform_array(shape, 3.0);
        let vx = uniform_array(shape, 2.0);
        let vy = uniform_array(shape, 0.0);
        let vz = uniform_array(shape, -1.0);

        let (ix, iy, iz) = acoustic_intensity(&p, &vx, &vy, &vz);

        for v in ix.iter() {
            assert!((v - 6.0).abs() < 1e-12, "I_x must equal p×vx = 6: {v}");
        }
        for v in iy.iter() {
            assert!(v.abs() < 1e-12, "I_y must equal 0: {v}");
        }
        for v in iz.iter() {
            assert!((v + 3.0).abs() < 1e-12, "I_z must equal p×vz = -3: {v}");
        }
    }

    // ── Acoustic heat source ─────────────────────────────────────────────────

    /// `acoustic_heat_source` with zero absorption must return zero everywhere.
    #[test]
    fn test_heat_source_zero_absorption() {
        let grid = make_grid();
        let shape = (grid.nx, grid.ny, grid.nz);
        let p = uniform_array(shape, 5000.0);
        let v = uniform_array(shape, 1.0);
        let rho = uniform_array(shape, 1000.0);
        let c = uniform_array(shape, 1500.0);
        let alpha = uniform_array(shape, 0.0);

        let q = acoustic_heat_source(&p, &v, &v, &v, &rho, &c, &alpha);
        for val in q.iter() {
            assert!(val.abs() < 1e-20, "Heat source with α=0 must be zero: {val:.3e}");
        }
    }

    // ── Second Law compliance ────────────────────────────────────────────────

    /// `validate_conservation` must report `is_conserved = false` when
    /// entropy production is negative (unphysical state).
    ///
    /// We force a negative entropy rate by passing negative absorption, which
    /// would be unphysical (energy creation).
    #[test]
    fn test_second_law_violation_detected() {
        let grid = make_grid();
        let shape = (grid.nx, grid.ny, grid.nz);
        let p = uniform_array(shape, 1000.0);
        let v = uniform_array(shape, 0.0);
        let rho = uniform_array(shape, 1000.0);
        let c = uniform_array(shape, 1500.0);
        // Negative absorption — unphysical but useful for testing the check
        let alpha = uniform_array(shape, -1.0);

        let volume = (grid.nx as f64 * grid.dx)
            * (grid.ny as f64 * grid.dy)
            * (grid.nz as f64 * grid.dz);
        let e_init = 1000.0 * 1000.0 / (2.0 * 1000.0 * 1500.0 * 1500.0) * volume;

        let metrics = validate_conservation(
            &p, &v, &v, &v, &rho, &c, &alpha, 310.0, None, None, None, None, None, e_init, 1e-6,
            &grid, 1.0,
        );

        assert!(
            metrics.entropy_production_rate < 0.0,
            "Negative absorption must yield negative entropy production"
        );
        assert!(
            !metrics.is_conserved,
            "Second Law violation must set is_conserved = false"
        );
    }
}
