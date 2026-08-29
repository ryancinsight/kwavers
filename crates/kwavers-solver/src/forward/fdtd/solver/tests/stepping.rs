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
