use super::*;
use crate::cpml::config::CPMLConfig;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_grid::Grid;

/// The k-Wave wall value is `sigma_max = pml_alpha * c0 / dx`.
/// # Panics
/// - Panics if `grid`.
/// - Panics if `CPMLProfiles::new should succeed`.
///
#[test]
fn test_cpml_sigma_max_formula() {
    let c0 = SOUND_SPEED_WATER_SIM;
    let dx = 1e-3_f64;
    let pml_size = 10_usize;
    let dt = 1e-7_f64;
    let pml_alpha = 2.0_f64;

    let grid = Grid::new(32, 32, 32, dx, dx, dx).expect("grid");
    let config = CPMLConfig::with_thickness(pml_size).with_alpha(pml_alpha);

    let profiles =
        CPMLProfiles::new(&config, &grid, c0, dt).expect("CPMLProfiles::new should succeed");

    let expected_sigma_max = pml_alpha * (c0 / dx);
    let actual = profiles.sigma_x[0];
    assert!(
        (actual - expected_sigma_max).abs() / expected_sigma_max < 0.01,
        "sigma_max = {actual:.1} should match k-Wave formula {expected_sigma_max:.1}"
    );
}

/// Singleton axes must remain CPML-neutral for lower-dimensional embeddings.
/// # Panics
/// - Panics if `grid`.
/// - Panics if `CPMLProfiles::new should succeed`.
///
#[test]
fn test_singleton_axis_profiles_are_neutral() {
    let c0 = SOUND_SPEED_WATER_SIM;
    let dx = 1e-3_f64;
    let dt = 1e-7_f64;

    let grid = Grid::new(32, 32, 1, dx, dx, dx).expect("grid");
    let config = CPMLConfig::with_thickness(10).with_alpha(2.0);

    let profiles =
        CPMLProfiles::new(&config, &grid, c0, dt).expect("CPMLProfiles::new should succeed");

    assert!(profiles.sigma_x.iter().any(|&v| v > 0.0));
    assert!(profiles.sigma_y.iter().any(|&v| v > 0.0));
    assert!(profiles.sigma_z.iter().all(|&v| v == 0.0));
    assert!(profiles.sigma_z_sgz.iter().all(|&v| v == 0.0));
    assert!(profiles.kappa_z.iter().all(|&v| v == 1.0));
    assert!(profiles.alpha_z.iter().all(|&v| v == 0.0));
    assert!(profiles.a_z.iter().all(|&v| v == 0.0));
    assert!(profiles.b_z.iter().all(|&v| v == 1.0));
}

/// Precomputed PML factors satisfy `pml_vel_*[i] = exp(-sigma_*_sg*[i] * dt/2)`
/// and `pml_den_*[i] = exp(-sigma_*[i] * dt/2)`.
///
/// Mathematical derivation: Treeby & Cox (2010) Eq. 17 applies the multiplicative
/// PML factor `exp(-σ·Δt/2)` twice per step.  The precomputed arrays materialise
/// each factor once at construction, enabling O(N) multiplications per step
/// instead of O(N) transcendental evaluations.
///
/// Invariants:
/// - PML wall cell: `pml_factor = exp(-sigma_max_at_wall * dt/2)` < 1
/// - Interior cell: `sigma = 0 → exp(0) = 1.0` (no attenuation)
/// - Staggered (velocity) vs collocated (density) factors differ at wall cells.
/// # Panics
/// - Panics if `CPMLProfiles::new` fails.
///
#[test]
fn test_precomputed_pml_exp_factors_match_sigma() {
    let c0 = SOUND_SPEED_WATER_SIM;
    let dx = 1e-3_f64;
    let pml_size = 10_usize;
    let dt = 1e-7_f64;
    let pml_alpha = 2.0_f64;

    let nx = 32;
    let grid = Grid::new(nx, nx, nx, dx, dx, dx).expect("grid");
    let config = CPMLConfig::with_thickness(pml_size).with_alpha(pml_alpha);
    let p = CPMLProfiles::new(&config, &grid, c0, dt).expect("CPMLProfiles::new");

    // ── Velocity factors (staggered sigma) ────────────────────────────────
    for i in 0..nx {
        let expected_vx = (-p.sigma_x_sgx[i] * dt * 0.5).exp();
        assert!(
            (p.pml_vel_x[i] - expected_vx).abs() < 1e-15,
            "pml_vel_x[{i}]: expected {expected_vx:.15e}, got {:.15e}",
            p.pml_vel_x[i]
        );
        let expected_vy = (-p.sigma_y_sgy[i] * dt * 0.5).exp();
        assert!(
            (p.pml_vel_y[i] - expected_vy).abs() < 1e-15,
            "pml_vel_y[{i}]"
        );
        let expected_vz = (-p.sigma_z_sgz[i] * dt * 0.5).exp();
        assert!(
            (p.pml_vel_z[i] - expected_vz).abs() < 1e-15,
            "pml_vel_z[{i}]"
        );

        // ── Density factors (collocated sigma) ────────────────────────────
        let expected_dx = (-p.sigma_x[i] * dt * 0.5).exp();
        assert!(
            (p.pml_den_x[i] - expected_dx).abs() < 1e-15,
            "pml_den_x[{i}]"
        );
        let expected_dy = (-p.sigma_y[i] * dt * 0.5).exp();
        assert!(
            (p.pml_den_y[i] - expected_dy).abs() < 1e-15,
            "pml_den_y[{i}]"
        );
        let expected_dz = (-p.sigma_z[i] * dt * 0.5).exp();
        assert!(
            (p.pml_den_z[i] - expected_dz).abs() < 1e-15,
            "pml_den_z[{i}]"
        );
    }

    // Interior cells must have factor = 1.0 (sigma = 0 → exp(0) = 1).
    let mid = nx / 2;
    assert!(
        (p.pml_vel_x[mid] - 1.0).abs() < 1e-14,
        "pml_vel_x at interior must be 1.0, got {}",
        p.pml_vel_x[mid]
    );
    assert!(
        (p.pml_den_x[mid] - 1.0).abs() < 1e-14,
        "pml_den_x at interior must be 1.0, got {}",
        p.pml_den_x[mid]
    );

    // PML wall cell must attenuate (factor < 1.0 since sigma_wall > 0).
    assert!(
        p.pml_vel_x[0] < 1.0,
        "pml_vel_x at PML wall must attenuate: got {}",
        p.pml_vel_x[0]
    );
    assert!(
        p.pml_den_x[0] < 1.0,
        "pml_den_x at PML wall must attenuate: got {}",
        p.pml_den_x[0]
    );

    // Staggered and collocated factors differ at wall (staggered is less absorbing).
    assert!(
        p.pml_vel_x[0] > p.pml_den_x[0],
        "staggered factor must be less absorbing than collocated at PML wall: \
         pml_vel_x[0]={:.8} pml_den_x[0]={:.8}",
        p.pml_vel_x[0],
        p.pml_den_x[0]
    );
}

/// Precomputed PML factors are consistent with `apply_velocity_pml_directional`
/// and `apply_acoustic_directional` — both must apply the same per-element attenuation.
///
/// This is the differential-equivalence test between the legacy per-step exp() path
/// and the precomputed-factor path.
/// # Panics
/// - Panics if `CPMLBoundary::new_with_time_step` fails or any assertion fails.
///
#[test]
fn test_fused_pml_matches_sequential_pml() {
    use crate::{Boundary, CPMLBoundary};
    use ndarray::Array3;

    let c0 = SOUND_SPEED_WATER_SIM;
    let dx = 1e-3_f64;
    let dt = 1e-7_f64;
    let pml_alpha = 2.0_f64;
    let nx = 16_usize;

    let grid = Grid::new(nx, nx, nx, dx, dx, dx).expect("grid");
    let config = CPMLConfig::with_thickness(4).with_alpha(pml_alpha);
    let mut boundary =
        CPMLBoundary::new_with_time_step(config.clone(), &grid, c0, Some(dt)).expect("boundary");

    let profiles = CPMLProfiles::new(&config, &grid, c0, dt).expect("profiles");

    // Construct a non-trivial initial field: random-ish values using deterministic formula.
    let shape = (nx, nx, nx);
    let u_init: Array3<f64> = Array3::from_shape_fn(shape, |(i, j, k)| {
        ((i + 2 * j + 3 * k) as f64 * 0.001).sin()
    });

    // ── Legacy path: apply pre-PML then post-PML with no gradient term ──────
    // pre: u *= pml; gradient = 0; post: u *= pml ↔ u *= pml^2
    let mut u_sequential = u_init.clone();
    boundary
        .apply_velocity_pml_directional(u_sequential.view_mut(), &grid, 0, 0)
        .expect("pre-PML x");
    boundary
        .apply_velocity_pml_directional(u_sequential.view_mut(), &grid, 0, 0)
        .expect("post-PML x");

    // ── Fused path: u = pml * (pml * u_old - 0) = pml^2 * u_old ────────────
    let mut u_fused = u_init.clone();
    let pml_vx = profiles.pml_vel_x.as_slice().expect("contiguous");
    ndarray::Zip::indexed(u_fused.view_mut()).for_each(|(i, _j, _k), val| {
        let p = pml_vx[i];
        *val = p * (p * *val);
    });

    // Both paths must produce identical results (bitwise, or within f64 rounding).
    for i in 0..nx {
        for j in 0..nx {
            for k in 0..nx {
                let seq = u_sequential[[i, j, k]];
                let fused = u_fused[[i, j, k]];
                assert!(
                    (seq - fused).abs() < 1e-15 * seq.abs().max(1e-20),
                    "mismatch at [{i},{j},{k}]: sequential={seq:.15e} fused={fused:.15e} diff={:.3e}",
                    (seq - fused).abs()
                );
            }
        }
    }
}

/// Roden-Gedney coefficients reduce to `b = exp(-sigma dt)` and `a = b - 1`.
/// # Panics
/// - Panics if `grid`.
/// - Panics if `CPMLProfiles::new should succeed`.
///
#[test]
fn test_cpml_recursive_convolution_coefficients() {
    let c0 = SOUND_SPEED_WATER_SIM;
    let dx = 1e-3_f64;
    let pml_size = 10_usize;
    let dt = 1e-7_f64;
    let pml_alpha = 2.0_f64;

    let nx = 32;
    let grid = Grid::new(nx, nx, nx, dx, dx, dx).expect("grid");
    let config = CPMLConfig::with_thickness(pml_size).with_alpha(pml_alpha);

    let profiles =
        CPMLProfiles::new(&config, &grid, c0, dt).expect("CPMLProfiles::new should succeed");

    let sigma_max = profiles.sigma_x[0];
    let expected_b = (-sigma_max * dt).exp();
    let expected_a = expected_b - 1.0;

    assert!(
        (profiles.b_x[0] - expected_b).abs() < 1e-12,
        "b_x at PML wall: expected {expected_b:.10}, got {:.10}",
        profiles.b_x[0]
    );
    assert!(
        (profiles.a_x[0] - expected_a).abs() < 1e-12,
        "a_x at PML wall: expected {expected_a:.10}, got {:.10}",
        profiles.a_x[0]
    );

    let mid = nx / 2;
    assert!(
        (profiles.b_x[mid] - 1.0).abs() < 1e-14,
        "b_x at interior must be 1.0, got {}",
        profiles.b_x[mid]
    );
    assert!(
        profiles.a_x[mid].abs() < 1e-14,
        "a_x at interior must be 0.0, got {}",
        profiles.a_x[mid]
    );
}
