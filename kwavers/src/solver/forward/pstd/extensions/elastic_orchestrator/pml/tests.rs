use super::*;

fn spec(
    shape: (usize, usize, usize),
    thickness_cells: (usize, usize, usize),
    spacing: (f64, f64, f64),
    c_max: f64,
    dt: f64,
    r0: f64,
) -> ElasticPmlSpec {
    ElasticPmlSpec {
        shape,
        thickness_cells,
        spacing,
        c_max,
        dt,
        r0,
    }
}

#[test]
fn no_thickness_means_unit_damping() {
    let pml = ElasticPml::new(spec(
        (16, 16, 16),
        (0, 0, 0),
        (1e-3, 1e-3, 1e-3),
        1500.0,
        1e-7,
        1e-4,
    ));
    let (dx, dy, dz) = pml.damping_axes();
    for v in dx.iter().chain(dy.iter()).chain(dz.iter()) {
        assert_eq!(*v, 1.0, "no-PML damping must be exactly 1.0 everywhere");
    }
}

#[test]
fn damping_is_monotonic_and_in_unit_interval() {
    let pml = ElasticPml::new(spec(
        (32, 32, 32),
        (8, 8, 8),
        (1e-3, 1e-3, 1e-3),
        1500.0,
        1e-7,
        1e-4,
    ));
    let (dx, _, _) = pml.damping_axes();
    for v in dx.iter() {
        assert!(*v > 0.0 && *v <= 1.0, "damping = {v} must be in (0, 1]");
    }
    assert!(dx[0] < dx[8], "damping must increase outward (left side)");
    let n = dx.len();
    assert!(
        dx[n - 1] < dx[n - 9],
        "damping must increase outward (right side)"
    );
}

/// Apply the PML to a unit field across `n_passes` passes and verify
/// that the cumulative attenuation in the absorbing layer matches
/// `(damping[i])^n_passes` exactly — i.e., the per-step multiplier
/// commutes with itself, as required for stable absorption.
#[test]
fn cumulative_attenuation_matches_per_step_multiplier_to_n() {
    let nx = 16usize;
    let ny = 4usize;
    let nz = 4usize;
    let thickness = 4usize;
    let pml = ElasticPml::new(spec(
        (nx, ny, nz),
        (thickness, 0, 0),
        (1e-3, 1e-3, 1e-3),
        1500.0,
        1e-7,
        1e-4,
    ));

    let mut field = ndarray::Array3::<f64>::ones((nx, ny, nz));
    let n_passes = 50usize;
    for _ in 0..n_passes {
        pml.apply_to_field(&mut field);
    }

    let (dx, _, _) = pml.damping_axes();
    for i in 0..nx {
        let expected = dx[i].powi(n_passes as i32);
        let actual = field[[i, 0, 0]];
        let rel_err = (actual - expected).abs() / expected.max(1e-300);
        assert!(
            rel_err < 1e-9,
            "i={i}: actual = {actual:.3e}, expected = {expected:.3e}, \
             rel_err = {rel_err:.3e}"
        );
    }
}

/// Roden-Gedney σ_max calibration check.
#[test]
fn outermost_damping_matches_roden_gedney_calibration() {
    let nx = 64usize;
    let thickness = 10usize;
    let dx = 1e-3_f64;
    let c_max = 1500.0_f64;
    let dt = 1e-7_f64;
    let r0 = 1e-4_f64;
    let pml = ElasticPml::new(spec(
        (nx, 4, 4),
        (thickness, 0, 0),
        (dx, 1e-3, 1e-3),
        c_max,
        dt,
        r0,
    ));

    let (dx_axis, _, _) = pml.damping_axes();

    const P: f64 = 4.0;
    let l = thickness as f64 * dx;
    let sigma_max = -(P + 1.0) * c_max * r0.ln() / (2.0 * l);
    let expected_outermost = (-sigma_max * dt).exp();

    let actual_outermost = dx_axis[0];
    let rel_err = (actual_outermost - expected_outermost).abs() / expected_outermost;
    assert!(
        rel_err < 1e-9,
        "outermost cell damping = {actual_outermost:.6e}, \
         expected exp(−σ_max·dt) = {expected_outermost:.6e}, rel_err = {rel_err:.3e}"
    );

    assert_eq!(
        dx_axis[thickness], 1.0,
        "first interior cell must have unity damping (no absorption)"
    );

    assert!(
        actual_outermost < 0.99,
        "outermost damping {actual_outermost:.6e} ≥ 0.99 — PML σ_max \
         too small to absorb meaningfully at the strongest cell"
    );
}
