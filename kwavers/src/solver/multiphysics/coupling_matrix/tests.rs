use super::coupler::{stability_dt, AcousticElasticCoupler};
use ndarray::Array3;

fn make_coupler(nx: usize) -> AcousticElasticCoupler {
    let mask = Array3::from_elem((nx, nx, nx), false);
    AcousticElasticCoupler::new([1.0, 0.0, 0.0], mask, 1000.0, 1500.0, 5960.0, 0.9).unwrap()
}

/// Coupling matrix antisymmetry: Cᵀ = −C
///
/// Both coupling coefficients must equal −n₀² = −1 to machine precision.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_coupling_antisymmetry() {
    let coupler = make_coupler(4);
    let dt = 1e-7_f64;
    let p = 1.0e3_f64;
    let a = [1.0, 0.0, 0.0_f64];

    let terms = coupler.coupling_terms_at_cell(p, a, dt);

    let rho_f = 1000.0_f64;
    let c_stress = terms.delta_solid_stress[0] / p;
    let c_velocity = terms.delta_fluid_velocity[0] * rho_f / (dt * a[0]);

    let antisymmetry_residual = c_stress - c_velocity;
    assert!(
        antisymmetry_residual.abs() < 1e-12,
        "Coupling antisymmetry violated: c_stress={:.6} c_velocity={:.6} residual={:.3e}",
        c_stress,
        c_velocity,
        antisymmetry_residual
    );
    assert!(
        (c_stress + 1.0).abs() < 1e-12,
        "Stress coefficient must equal −n₀² = −1, got {:.6}",
        c_stress
    );
}

/// Stability dt is below individual CFL limits for both sub-domains.
/// # Panics
/// - Panics if assertion fails: `dt = {:.3e} must be < fluid CFL limit {:.3e}`.
/// - Panics if assertion fails: `dt = {:.3e} must be < solid CFL limit {:.3e}`.
///
#[test]
fn test_stability_dt_below_cfl() {
    let dx = 0.1e-3_f64;
    let c_fluid = 1500.0_f64;
    let c_solid = 5960.0_f64;
    let cfl = 0.9_f64;

    let dt = stability_dt(c_fluid, c_solid, dx, cfl);

    let dt_fluid_max = dx / c_fluid;
    let dt_solid_max = dx / c_solid;

    assert!(
        dt < dt_fluid_max,
        "dt = {:.3e} must be < fluid CFL limit {:.3e}",
        dt,
        dt_fluid_max
    );
    assert!(
        dt < dt_solid_max,
        "dt = {:.3e} must be < solid CFL limit {:.3e}",
        dt,
        dt_solid_max
    );
}

/// Traction balance: (σ + Δσ)·n̂ + p·n̂ = 0 at interface.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_traction_balance() {
    let nx = 4usize;
    let mut coupler = make_coupler(nx);

    let i_face = nx / 2;
    for j in 0..nx {
        for k in 0..nx {
            coupler.interface_mask[(i_face, j, k)] = true;
        }
    }

    let p0 = 1.0e4_f64;
    let dt = 1.0e-7_f64;

    let fluid_pressure = Array3::from_elem((nx, nx, nx), p0);
    let solid_accel: [Array3<f64>; 3] = [
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
    ];
    let mut fluid_velocity: [Array3<f64>; 3] = [
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
    ];
    let mut solid_stress: [Array3<f64>; 6] = [
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
    ];

    coupler
        .apply(
            &mut fluid_velocity,
            &mut solid_stress,
            &fluid_pressure,
            &solid_accel,
            dt,
        )
        .unwrap();

    for j in 0..nx {
        for k in 0..nx {
            let sigma_xx = solid_stress[0][(i_face, j, k)];
            let traction_x = sigma_xx + p0;
            assert!(
                traction_x.abs() < 1e-8,
                "Traction balance violated at ({},{},{}): σ_xx + p₀ = {:.3e}",
                i_face,
                j,
                k,
                traction_x
            );
        }
    }
}

/// Energy conservation: zero solid acceleration must not change fluid kinetic energy.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_energy_conservation_cavity() {
    let nx = 4usize;
    let mut coupler = make_coupler(nx);
    let i_face = nx / 2;
    for j in 0..nx {
        for k in 0..nx {
            coupler.interface_mask[(i_face, j, k)] = true;
        }
    }

    let dt = 1.0e-8_f64;
    let p0 = 1.0e3_f64;
    let v0 = p0 / (1000.0 * 1500.0);

    let fluid_pressure = Array3::from_elem((nx, nx, nx), p0);
    let solid_accel: [Array3<f64>; 3] = [
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
    ];
    let mut fluid_velocity: [Array3<f64>; 3] = [
        Array3::from_elem((nx, nx, nx), v0),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
    ];
    let mut solid_stress: [Array3<f64>; 6] =
        std::array::from_fn(|_| Array3::zeros((nx, nx, nx)));

    let e_before: f64 = fluid_velocity[0]
        .iter()
        .map(|&v| 0.5 * 1000.0 * v * v)
        .sum();

    coupler
        .apply(
            &mut fluid_velocity,
            &mut solid_stress,
            &fluid_pressure,
            &solid_accel,
            dt,
        )
        .unwrap();

    let e_after: f64 = fluid_velocity[0]
        .iter()
        .map(|&v| 0.5 * 1000.0 * v * v)
        .sum();

    let rel_change = (e_after - e_before).abs() / e_before;
    assert!(
        rel_change < 1e-12,
        "Energy changed by {:.3e} (must be 0 for zero solid acceleration)",
        rel_change
    );
}
