use ndarray::Array3;

use super::struct_impl::FluidStructureSolver;
use super::super::interface::FsiInterface;

/// Test zero normal vector rejection
#[test]
fn test_zero_normal_rejected() {
    let interface = FsiInterface::new(
        1000.0,
        1500.0,
        7850.0,
        5960.0,
        3240.0,
        [0.0, 0.0, 0.0], // Invalid!
        64,
        64,
        64,
    );
    assert!(interface.is_err());
}

/// Test ghost cell traction balance at planar water-steel interface.
#[test]
fn test_ghost_cell_traction_balance() {
    let nx = 8usize;
    let interface = FsiInterface::new(
        1000.0,
        1500.0,
        7850.0,
        5960.0,
        3240.0,
        [1.0, 0.0, 0.0],
        nx,
        nx,
        nx,
    )
    .unwrap();

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

/// Test ghost cell velocity continuity across interface.
#[test]
fn test_ghost_cell_velocity_continuity() {
    let nx = 8usize;
    let interface = FsiInterface::new(
        1000.0,
        1500.0,
        7850.0,
        5960.0,
        3240.0,
        [1.0, 0.0, 0.0],
        nx,
        nx,
        nx,
    )
    .unwrap();

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
