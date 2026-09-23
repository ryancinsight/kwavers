//! Time-step orchestration regressions shared by FDTD temporal schemes.

use crate::forward::fdtd::config::{FdtdConfig, TemporalScheme};
use crate::forward::fdtd::solver::FdtdSolver;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::{GridSource, SourceMode};
use leto::Array3;

#[test]
fn every_temporal_scheme_injects_each_velocity_source_once() {
    const N: usize = 6;
    const DX: f64 = 1.0e-3;
    const C0: f64 = 1_500.0;
    const RHO0: f64 = 1_000.0;
    const CFL: f64 = 0.1;
    const AMPLITUDE: f64 = 0.75;
    const SOURCE_CELL: [usize; 3] = [N / 2; 3];

    for scheme in [TemporalScheme::Leapfrog, TemporalScheme::Yoshida4] {
        let grid = Grid::new(N, N, N, DX, DX, DX).expect("test grid must be valid");
        let medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, &grid);
        let dt = CFL * DX / C0;

        let mut velocity_mask = Array3::zeros([N, N, N]);
        velocity_mask[SOURCE_CELL] = 1.0;
        let mut velocity_signal = Array3::zeros([3, 1, 1]);
        velocity_signal[[0, 0, 0]] = AMPLITUDE;
        let source = GridSource {
            u_mask: Some(velocity_mask),
            u_signal: Some(velocity_signal),
            u_mode: SourceMode::AdditiveNoCorrection,
            ..GridSource::new_empty()
        };
        let config = FdtdConfig {
            temporal_scheme: scheme,
            staggered_grid: true,
            spatial_order: 2,
            cfl_factor: CFL,
            dt,
            nt: 2,
            ..FdtdConfig::default()
        };
        let mut solver = FdtdSolver::new(config, &grid, &medium, source)
            .expect("velocity-source solver must be valid");

        solver.step_forward().expect("one stable step must succeed");

        // Additive-no-correction injection is 2*c*dt/dx times the supplied
        // signal. Eight roundings bound the scale, multiplication, and sum.
        let expected = 2.0 * C0 * dt / DX * AMPLITUDE;
        let error_bound = 8.0 * f64::EPSILON * expected.abs().max(1.0);
        let observed = solver.fields.ux[SOURCE_CELL];
        let total: f64 = solver.fields.ux.iter().sum();
        assert!(
            (observed - expected).abs() <= error_bound,
            "{scheme:?}: source cell {observed:.17e} differs from one injection {expected:.17e}"
        );
        assert!(
            (total - expected).abs() <= error_bound,
            "{scheme:?}: x-velocity total {total:.17e} differs from one injection {expected:.17e}"
        );
        assert!(
            solver.fields.uy.iter().all(|&value| value == 0.0)
                && solver.fields.uz.iter().all(|&value| value == 0.0),
            "{scheme:?}: an x-directed source changed another velocity component"
        );
    }
}

/// Without a CPML the staggered velocity update fuses each gradient into
/// its component's update; the velocities are the separate sweep and
/// pointwise pass's to the bit, for every stencil order the solver offers.
#[test]
fn the_fused_velocity_update_is_the_sweep_then_the_pass_to_the_bit() {
    for order in [2, 4, 6] {
        let mut solver = super::make_solver(9, 1.0e-4, 1500.0, 1000.0, 0.5, order);
        assert!(
            solver.cpml_boundary.is_none(),
            "the fused route needs no CPML"
        );
        let [nx, ny, nz] = solver.fields.p.shape();
        solver.fields.p = Array3::from_shape_fn((nx, ny, nz), |[i, j, k]| {
            ((i * 7 + j * 3 + k * 5) as f64 * 0.031).sin()
        });
        solver.fields.ux = Array3::from_shape_fn((nx, ny, nz), |[i, j, k]| {
            ((i * 2 + j * 11 + k) as f64 * 0.017).cos()
        });
        let dt = solver.config.dt;
        let expected = [
            (
                leto_ops::Axis::X,
                &solver.fields.ux,
                &solver.staggered_density[0],
            ),
            (
                leto_ops::Axis::Y,
                &solver.fields.uy,
                &solver.staggered_density[1],
            ),
            (
                leto_ops::Axis::Z,
                &solver.fields.uz,
                &solver.staggered_density[2],
            ),
        ]
        .map(|(axis, velocity, density)| {
            let mut gradient = Array3::zeros((nx, ny, nz));
            solver
                .leapfrog_operator
                .gradient_into(axis, solver.fields.p.view(), &mut gradient.view_mut())
                .expect("grid-shaped fields");
            Array3::from_shape_fn((nx, ny, nz), |at| {
                let rho = density[at];
                if rho > 1e-9 {
                    velocity[at] - dt / rho * gradient[at]
                } else {
                    velocity[at]
                }
            })
        });
        solver.update_velocity(dt).expect("velocity update");
        for (component, want, got) in [
            ("ux", &expected[0], &solver.fields.ux),
            ("uy", &expected[1], &solver.fields.uy),
            ("uz", &expected[2], &solver.fields.uz),
        ] {
            for (index, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "order {order}: {component} at flat {index}: {a} against {b}"
                );
            }
        }
    }
}
