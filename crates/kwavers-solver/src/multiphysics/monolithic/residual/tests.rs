use super::super::config::NewtonKrylovConfig;
use super::super::coupler::MonolithicCoupler;
use super::super::residual_metric::norm;
use crate::integration::nonlinear::GMRESConfig;
use kwavers_field::UnifiedFieldType;
use leto::Array3;

#[test]
fn test_compute_residual_zero_fields() {
    let coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    let field_order = vec![UnifiedFieldType::Pressure, UnifiedFieldType::Temperature];
    let dims = (4, 3, 2);
    let n = field_order.len() * dims.0;
    let u = Array3::zeros((n, dims.1, dims.2));
    let u_prev = Array3::zeros((n, dims.1, dims.2));

    let res = coupler
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();
    let norm = norm(&res);
    assert!(
        norm < 1e-15,
        "Residual of zero state should be zero, got {norm}"
    );
}

#[test]
fn test_pressure_residual_scales_laplacian_by_sound_speed_squared() {
    let mut coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
    coupler.physics_coefficients.sound_speed = 3.0;
    coupler.grid_spacing = (1.0, 1.0, 1.0);

    let dims = (3, 3, 3);
    let field_order = vec![UnifiedFieldType::Pressure];
    let mut u = Array3::zeros(dims);
    u[[1, 1, 1]] = 1.0;
    let u_prev = Array3::zeros(dims);

    let residual = coupler
        .compute_residual(&u, &u_prev, 1.0, dims, &field_order)
        .unwrap();

    // Centered 3-D Laplacian of a unit impulse is -6. The pressure residual is
    // p - dt * c^2 * laplacian(p), so 1 - 1 * 9 * (-6) = 55.
    assert_eq!(residual[[1, 1, 1]], 55.0);
}

/// Halving the Grüneisen parameter halves the photoacoustic source contribution
/// in the Pressure block residual.
///
/// The photoacoustic source is `R_p += Γ · μ_a · I` (Oraevsky & Karabutov
/// 2003). With zero pressure and no Laplacian contribution, the residual at a
/// lit voxel is proportional to `Γ · μ_a · I`, so halving `Γ` halves the
/// residual contribution.
#[test]
fn test_photoacoustic_source_scales_with_gruneisen() {
    let make_coupler = |gruneisen: f64| {
        let mut c = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
        c.physics_coefficients.gruneisen = gruneisen;
        c.grid_spacing = (1e-3, 1e-3, 1e-3);
        c
    };

    let dims = (4, 4, 4);
    let nx = dims.0;
    let field_order = vec![UnifiedFieldType::Pressure, UnifiedFieldType::LightFluence];
    let n_blocks = field_order.len();

    let mut u = Array3::zeros((n_blocks * nx, dims.1, dims.2));
    u[[nx + 1, 1, 1]] = 1.0;
    let mut u_prev = Array3::zeros((n_blocks * nx, dims.1, dims.2));
    u_prev[[nx + 1, 1, 1]] = 1.0;

    let c1 = make_coupler(0.12);
    let r1 = c1
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    let c2 = make_coupler(0.06);
    let r2 = c2
        .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
        .unwrap();

    let v1 = r1[[1, 1, 1]];
    let v2 = r2[[1, 1, 1]];
    assert!(
        v1.abs() > 1e-20,
        "Pressure residual at lit voxel must be non-zero, got {v1}"
    );
    let ratio = v1 / v2;
    assert!(
        (ratio - 2.0).abs() < 1e-10,
        "Residual ratio (gamma=0.12)/(gamma=0.06) must be exactly 2.0, got {ratio}"
    );
}
