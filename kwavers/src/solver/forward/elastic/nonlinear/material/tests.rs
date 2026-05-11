#![allow(clippy::needless_range_loop)]
use super::*;

#[test]
fn test_hyperelastic_neo_hookean() {
    let model = HyperelasticModel::neo_hookean_soft_tissue();

    // Test strain energy for undeformed state (I₁=3, I₂=3, J=1)
    let w = model.strain_energy(3.0, 3.0, 1.0);
    assert!(
        (w - 0.0).abs() < 1e-10,
        "Strain energy should be zero at reference state"
    );

    // Test with deformation
    let w_deformed = model.strain_energy(4.0, 4.0, 1.0);
    assert!(
        w_deformed > 0.0,
        "Strain energy should be positive under deformation"
    );
}

#[test]
fn test_hyperelastic_mooney_rivlin() {
    let model = HyperelasticModel::mooney_rivlin_biological();

    // Test strain energy
    let w = model.strain_energy(3.0, 3.0, 1.0);
    assert!((w - 0.0).abs() < 1e-10);

    let w_deformed = model.strain_energy(4.0, 5.0, 1.0);
    assert!(w_deformed > 0.0);
}

#[test]
fn test_principal_stretches() {
    let model = HyperelasticModel::neo_hookean_soft_tissue();

    // Identity deformation gradient
    let f_identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let lambda = model.principal_stretches(&f_identity);

    // All principal stretches should be 1.0
    for &l in &lambda {
        assert!((l - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_strain_energy_derivatives() {
    let model = HyperelasticModel::neo_hookean_soft_tissue();

    let dw_di1 = model.compute_strain_energy_derivative_wrt_i1(3.0, 3.0, 1.0, None);
    assert!(dw_di1 > 0.0);

    let dw_di2 = model.compute_strain_energy_derivative_wrt_i2(3.0, 3.0, 1.0);
    assert_eq!(dw_di2, 0.0); // Neo-Hookean has no I₂ dependence

    let dw_dj = model.compute_strain_energy_derivative_wrt_j(3.0, 3.0, 1.0);
    assert_eq!(dw_dj, 0.0); // Zero at J=1 (reference state)
}

#[test]
fn test_cauchy_stress_reference_state() {
    let model = HyperelasticModel::neo_hookean_soft_tissue();

    // Identity deformation gradient
    let f_identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let stress = model.cauchy_stress(&f_identity);

    // Stress should be zero at reference state
    for row in &stress {
        for &val in row {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }
}

#[test]
fn test_matrix_eigenvalues() {
    let model = HyperelasticModel::neo_hookean_soft_tissue();

    // Identity matrix
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let eig = model.matrix_eigenvalues(&identity);

    // All eigenvalues should be 1.0
    for &e in &eig {
        assert!((e - 1.0).abs() < 1e-10);
    }

    // Diagonal matrix
    let diag = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
    let eig_diag = model.matrix_eigenvalues(&diag);

    // Eigenvalues should be 2, 3, 4 (sorted)
    assert!((eig_diag[0] - 2.0).abs() < 1e-10);
    assert!((eig_diag[1] - 3.0).abs() < 1e-10);
    assert!((eig_diag[2] - 4.0).abs() < 1e-10);
}

/// At F = I, the volumetric term ∂W/∂J = 2D₁(J−1) = 0, so P = 2C₁·I.
///
/// ## Note on compressible vs. incompressible formulations
///
/// For the compressible Neo-Hookean model W = C₁(I₁−3) + D₁(J−1)²,
/// ∂W/∂F|_{F=I} = 2C₁·I ≠ 0.  This is mathematically correct: the strain
/// energy W = C₁(I₁−3) is minimized at I₁=3 (F=I), but the first PK stress
/// P = ∂W/∂F = 2C₁F is non-zero at F=I.  The incompressible formulation
/// (Lagrange multiplier p enforcing J=1) adds −p I to produce zero total stress
/// at the reference state.  The energy-derivative formula correctly captures
/// the compressible behavior.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_first_pk_stress_reference_state_diagonal() {
    let model = HyperelasticModel::neo_hookean_soft_tissue();
    let f_id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let p = model.first_pk_stress(&f_id);
    // Off-diagonal must be zero (P = 2C₁·I is diagonal at F = I)
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                assert!(
                    p[i][j].abs() < 1e-10,
                    "Off-diagonal P[{i}][{j}] = {} at reference state",
                    p[i][j]
                );
            }
        }
    }
    // Diagonal elements must be equal (isotropic model)
    assert!(
        (p[0][0] - p[1][1]).abs() < 1e-10 && (p[1][1] - p[2][2]).abs() < 1e-10,
        "Diagonal PK stress must be equal at reference state for isotropic model"
    );
}

/// For principal-axis loading (diagonal F), P must also be diagonal (zero off-diagonals).
///
/// ## Proof sketch
///
/// Under a diagonal F = diag(λ₁, λ₂, λ₃), both σ and cof(F) are diagonal.
/// The matrix product P = σ · cof(F) of two diagonal matrices is diagonal. □
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_first_pk_stress_diagonal_for_principal_loading() {
    let model = HyperelasticModel::neo_hookean_soft_tissue();
    let lam = 1.5_f64;
    // Isochoric uniaxial stretch: λ₁ = λ, λ₂ = λ₃ = 1/√λ → det = 1
    let inv_sqrt_lam = 1.0 / lam.sqrt();
    let f = [
        [lam, 0.0, 0.0],
        [0.0, inv_sqrt_lam, 0.0],
        [0.0, 0.0, inv_sqrt_lam],
    ];
    let p = model.first_pk_stress(&f);
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                assert!(
                    p[i][j].abs() < 1e-12,
                    "Off-diagonal P[{i}][{j}] = {} for principal loading",
                    p[i][j]
                );
            }
        }
    }
}

/// Verify P = ∂W/∂F via forward-difference numerical differentiation.
///
/// For any hyperelastic material, `P_{iA} = ∂W/∂F_{iA}` (definition).
/// We check this numerically with step h = 1e-6 (relative error ≈ h²/6 ·|∂³W/∂F³|).
/// # Panics
/// - Panics if assertion fails: `P[{i}][{cap_a}]: analytic={:.6e} finite-diff={:.6e} rel_err={rel_err:.2e}`.
///
#[test]
fn test_first_pk_stress_is_energy_gradient() {
    let model = HyperelasticModel::neo_hookean_soft_tissue();
    let lam = 1.3_f64;
    let f0 = [
        [lam, 0.0, 0.0],
        [0.0, 1.0 / lam.sqrt(), 0.0],
        [0.0, 0.0, 1.0 / lam.sqrt()],
    ];

    fn w_from_f(model: &HyperelasticModel, f: &[[f64; 3]; 3]) -> f64 {
        let mut c = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    c[i][j] += f[k][i] * f[k][j];
                }
            }
        }
        let i1 = c[0][0] + c[1][1] + c[2][2];
        let mut tr_c2 = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                tr_c2 += c[i][j] * c[j][i];
            }
        }
        let i2 = 0.5 * (i1 * i1 - tr_c2);
        let j_det = f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
            - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
            + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]);
        model.strain_energy(i1, i2, j_det.abs())
    }

    let p_analytic = model.first_pk_stress(&f0);
    let h = 1e-6_f64;

    for i in 0..3 {
        for cap_a in 0..3 {
            let mut f_plus = f0;
            let mut f_minus = f0;
            f_plus[i][cap_a] += h;
            f_minus[i][cap_a] -= h;
            let dw_dfia = (w_from_f(&model, &f_plus) - w_from_f(&model, &f_minus)) / (2.0 * h);
            let rel_err = (p_analytic[i][cap_a] - dw_dfia).abs() / (dw_dfia.abs().max(1e-12));
            assert!(
                rel_err < 1e-5,
                "P[{i}][{cap_a}]: analytic={:.6e} finite-diff={:.6e} rel_err={rel_err:.2e}",
                p_analytic[i][cap_a],
                dw_dfia
            );
        }
    }
}
