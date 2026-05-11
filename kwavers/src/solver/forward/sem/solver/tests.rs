use super::config::SemConfig;
use super::sem_solver::SemSolver;
use crate::solver::forward::sem::mesh::MeshBuilder;
use ndarray::Array1;
use num_complex::Complex64;
use std::sync::Arc;

#[test]
fn test_sem_solver_creation() {
    let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 4);
    let config = SemConfig::default();

    let solver = SemSolver::new(config, Arc::new(mesh)).unwrap();

    assert_eq!(solver.config.polynomial_degree, 4);
    assert_eq!(solver.solution.len(), 125); // (4+1)³ = 125 DOFs for degree 4 mesh
}

#[test]
fn test_sem_system_assembly() {
    let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 2);
    let config = SemConfig {
        polynomial_degree: 2,
        ..Default::default()
    };

    let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
    solver.assemble_system().unwrap();

    assert!(solver.mass_matrix.iter().any(|&m| m > 0.0));
}

#[test]
fn test_sem_time_stepping() {
    let mesh = MeshBuilder::create_rectangular_mesh(0.1, 0.1, 0.1, 2);
    let config = SemConfig {
        polynomial_degree: 2,
        n_steps: 5,
        dt: 1e-8,
        ..Default::default()
    };

    let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
    solver.assemble_system().unwrap();

    for _ in 0..3 {
        solver.step().unwrap();
    }

    assert_eq!(solver.current_step(), 3);
    assert!(solver.current_time() > 0.0);
}

/// Constant field u=1: ∇u=0 everywhere, so K·u must be identically zero.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_stiffness_constant_field_is_zero() {
    let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 3);
    let config = SemConfig {
        polynomial_degree: 3,
        sound_speed: 1500.0,
        density: 1000.0,
        ..Default::default()
    };
    let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
    solver.assemble_system().unwrap();

    let n = solver.solution.len();
    let u_const = Array1::<f64>::ones(n);
    let ku = solver.apply_stiffness(&u_const);

    let scale = solver.config.density * solver.config.sound_speed.powi(2);
    let tol = scale * 1e-10;
    for (i, &v) in ku.iter().enumerate() {
        assert!(
            v.abs() < tol,
            "K·1 should be zero at dof {i}, got {v:.3e} (tol {tol:.3e})"
        );
    }
}

/// uᵀKu = ρc² ∫|∇u|² dΩ = ρc² × Lx × Ly × Lz  for u = x on [0,Lx]×[0,Ly]×[0,Lz]
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_stiffness_energy_linear_field() {
    let lx = 2.0;
    let ly = 1.5;
    let lz = 1.0;
    let mesh = MeshBuilder::create_rectangular_mesh(lx, ly, lz, 3);
    let config = SemConfig {
        polynomial_degree: 3,
        sound_speed: 1500.0,
        density: 1000.0,
        ..Default::default()
    };
    let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
    solver.assemble_system().unwrap();

    let n = solver.mesh.basis.n_points();
    let mut u = Array1::<f64>::zeros(solver.solution.len());
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                let xi = solver.mesh.basis.gll_points[a];
                let x = lx * (xi + 1.0) / 2.0;
                let g = solver.element_local_to_global_dof(0, a, b, c, n);
                u[g] = x;
            }
        }
    }

    let ku = solver.apply_stiffness(&u);
    let energy: f64 = u.iter().zip(ku.iter()).map(|(ui, kui)| ui * kui).sum();

    let expected = solver.config.density * solver.config.sound_speed.powi(2) * lx * ly * lz;
    let rel_err = (energy - expected).abs() / expected;
    assert!(
        rel_err < 1e-6,
        "Stiffness energy u^T K u = {energy:.6e}, expected {expected:.6e} (rel err {rel_err:.2e})"
    );
}

/// K is symmetric: (K·u)·v = u·(K·v)
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_stiffness_symmetry() {
    let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 2);
    let config = SemConfig {
        polynomial_degree: 2,
        ..Default::default()
    };
    let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
    solver.assemble_system().unwrap();

    let n = solver.solution.len();
    let u: Array1<f64> = (0..n)
        .map(|i| (std::f64::consts::PI * i as f64 / n as f64).sin())
        .collect();
    let v: Array1<f64> = (0..n)
        .map(|i| (std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect();

    let ku = solver.apply_stiffness(&u);
    let kv = solver.apply_stiffness(&v);

    let ku_dot_v: f64 = ku.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let u_dot_kv: f64 = u.iter().zip(kv.iter()).map(|(a, b)| a * b).sum();

    let scale = ku_dot_v.abs().max(u_dot_kv.abs()).max(1e-30);
    assert!(
        (ku_dot_v - u_dot_kv).abs() / scale < 1e-10,
        "Stiffness asymmetry: (Ku)·v={ku_dot_v:.6e} u·(Kv)={u_dot_kv:.6e}"
    );
}

/// E = ½(vᵀMv + uᵀKu) is conserved within 1% over 20 steps with Newmark.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_free_vibration_energy_conservation() {
    let mesh = MeshBuilder::create_rectangular_mesh(0.1, 0.1, 0.1, 2);
    let config = SemConfig {
        polynomial_degree: 2,
        n_steps: 20,
        dt: 1e-9,
        sound_speed: 1500.0,
        density: 1000.0,
        wavenumber: 1.0,
    };
    let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
    solver.assemble_system().unwrap();

    let n = solver.mesh.basis.n_points();
    let n_dofs = solver.solution.len();
    let mut u0 = Array1::<f64>::zeros(n_dofs);
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                let g = solver.element_local_to_global_dof(0, a, b, c, n);
                u0[g] = (std::f64::consts::PI * a as f64 / (n - 1).max(1) as f64).sin();
            }
        }
    }
    solver.set_initial_conditions(u0).unwrap();

    let compute_energy = |sol: &SemSolver| -> f64 {
        let ku = sol.apply_stiffness(&sol.solution);
        let v = &sol.integrator.velocity;
        let potential: f64 = sol
            .solution
            .iter()
            .zip(ku.iter())
            .map(|(u, ku)| u * ku)
            .sum();
        let kinetic: f64 = v
            .iter()
            .zip(sol.mass_matrix.iter())
            .map(|(vi, mi)| vi * vi * mi)
            .sum();
        0.5 * (kinetic + potential)
    };

    let e0 = compute_energy(&solver);
    if e0 < 1e-30 {
        return;
    }

    for _ in 0..20 {
        solver.step().unwrap();
    }

    let e_final = compute_energy(&solver);
    let relative_drift = (e_final - e0).abs() / e0;
    assert!(
        relative_drift < 0.01,
        "Energy drift {relative_drift:.3e} exceeds 1% over 20 steps (E0={e0:.3e}, Ef={e_final:.3e})"
    );
}

#[test]
fn test_boundary_condition_management() {
    let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 2);
    let config = SemConfig {
        polynomial_degree: 2,
        ..Default::default()
    };

    let mut solver = SemSolver::new(config, Arc::new(mesh)).unwrap();
    solver
        .boundary_manager()
        .add_dirichlet(vec![(0, Complex64::new(1.0, 0.0))]);

    assert_eq!(solver.boundary_manager_ref().len(), 1);
}
