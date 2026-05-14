use ndarray::Array3;

use super::super::interface::{FsiInterface, FsiInterfaceSpec};
use super::struct_impl::FluidStructureSolver;

fn water_steel_spec(normal: [f64; 3], n: usize) -> FsiInterfaceSpec {
    FsiInterfaceSpec {
        fluid_density: 1000.0,
        fluid_sound_speed: 1500.0,
        solid_density: 7850.0,
        solid_c_l: 5960.0,
        solid_c_t: 3240.0,
        normal,
        grid_shape: (n, n, n),
    }
}

/// Test zero normal vector rejection
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_zero_normal_rejected() {
    let interface = FsiInterface::new(water_steel_spec([0.0, 0.0, 0.0], 64));
    assert!(interface.is_err());
}

/// Test ghost cell traction balance at planar water-steel interface.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_ghost_cell_traction_balance() {
    let nx = 8usize;
    let interface = FsiInterface::new(water_steel_spec([1.0, 0.0, 0.0], nx)).unwrap();

    let mut solver = FluidStructureSolver::new(interface);

    let i_face = nx / 2;
    for j in 0..nx {
        for k in 0..nx {
            solver.interface.interface_mask[(i_face, j, k)] = true;
        }
    }

    let p0 = 1.0e5_f64;
    let fluid_pressure = Array3::from_elem((nx, nx, nx), p0);
    let solid_stress: [Array3<f64>; 6] = [
        Array3::from_elem((nx, nx, nx), p0),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
    ];

    solver
        .exchange_ghost_cells(&fluid_pressure, &solid_stress)
        .unwrap();

    let mut traction_jump_sq = 0.0f64;
    for j in 0..nx {
        for k in 0..nx {
            let t_fluid_x = -fluid_pressure[(i_face, j, k)] * 1.0;
            let t_solid_x = solver.p_fluid_ghost[(i_face, j, k)];
            traction_jump_sq += (t_fluid_x + t_solid_x).powi(2);
        }
    }
    assert!(
        traction_jump_sq < 1e-10,
        "Traction balance violated: ||t_fluid + t_solid||² = {:.3e} (must be < 1e-10)",
        traction_jump_sq
    );
    for j in 0..nx {
        for k in 0..nx {
            let t_x = solver.t_solid_ghost[0][(i_face, j, k)];
            assert!(
                (t_x + p0).abs() < 1e-10,
                "Solid ghost traction t_x = {}, expected {}",
                t_x,
                -p0
            );
        }
    }
    let _ = solid_stress[0].sum();
}

/// Test ghost exchange updates values without replacing workspace buffers.
///
/// The FSI exchange contract is pointwise: for normal `n=(1,0,0)`,
/// `p_fluid_ghost = σ_xx` and `t_solid_ghost[0] = -p`.  Repeating the exchange
/// with different physical inputs must update those values while preserving the
/// solver-owned buffer addresses.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_exchange_ghost_cells_reuses_workspace_buffers() {
    let nx = 6usize;
    let interface = FsiInterface::new(water_steel_spec([1.0, 0.0, 0.0], nx)).unwrap();

    let mut solver = FluidStructureSolver::new(interface);
    let i_face = nx / 2;
    solver.interface.interface_mask[(i_face, 2, 2)] = true;

    let pressure_a = Array3::from_elem((nx, nx, nx), 10.0);
    let stress_a: [Array3<f64>; 6] = [
        Array3::from_elem((nx, nx, nx), 20.0),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
    ];
    solver.exchange_ghost_cells(&pressure_a, &stress_a).unwrap();

    let p_ptr = solver.p_fluid_ghost.as_ptr();
    let t_ptrs = [
        solver.t_solid_ghost[0].as_ptr(),
        solver.t_solid_ghost[1].as_ptr(),
        solver.t_solid_ghost[2].as_ptr(),
    ];
    assert_eq!(solver.p_fluid_ghost[(i_face, 2, 2)], 20.0);
    assert_eq!(solver.t_solid_ghost[0][(i_face, 2, 2)], -10.0);

    let pressure_b = Array3::from_elem((nx, nx, nx), 30.0);
    let stress_b: [Array3<f64>; 6] = [
        Array3::from_elem((nx, nx, nx), 40.0),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
    ];
    solver.exchange_ghost_cells(&pressure_b, &stress_b).unwrap();

    assert_eq!(solver.p_fluid_ghost.as_ptr(), p_ptr);
    assert_eq!(solver.t_solid_ghost[0].as_ptr(), t_ptrs[0]);
    assert_eq!(solver.t_solid_ghost[1].as_ptr(), t_ptrs[1]);
    assert_eq!(solver.t_solid_ghost[2].as_ptr(), t_ptrs[2]);
    assert_eq!(solver.p_fluid_ghost[(i_face, 2, 2)], 40.0);
    assert_eq!(solver.t_solid_ghost[0][(i_face, 2, 2)], -30.0);
}

/// Test ghost cell velocity continuity across interface.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_ghost_cell_velocity_continuity() {
    let nx = 8usize;
    let interface = FsiInterface::new(water_steel_spec([1.0, 0.0, 0.0], nx)).unwrap();

    let mut solver = FluidStructureSolver::new(interface);
    let i_face = nx / 2;
    for j in 0..nx {
        for k in 0..nx {
            solver.interface.interface_mask[(i_face, j, k)] = true;
        }
    }

    let v_normal = 0.1_f64;
    let fluid_velocity: [Array3<f64>; 3] = [
        Array3::from_elem((nx, nx, nx), v_normal),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
    ];
    let solid_velocity: [Array3<f64>; 3] = [
        Array3::from_elem((nx, nx, nx), v_normal),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
    ];

    let converged = solver
        .check_convergence(&fluid_velocity, &solid_velocity)
        .unwrap();
    assert!(
        converged,
        "Velocity continuity check failed: matching fluid/solid velocities must converge"
    );

    let solid_velocity_bad: [Array3<f64>; 3] = [
        Array3::from_elem((nx, nx, nx), v_normal + 1.0),
        Array3::zeros((nx, nx, nx)),
        Array3::zeros((nx, nx, nx)),
    ];
    let not_converged = solver
        .check_convergence(&fluid_velocity, &solid_velocity_bad)
        .unwrap();
    assert!(
        !not_converged,
        "Velocity mismatch of 1 m/s should not satisfy convergence criterion"
    );
}
