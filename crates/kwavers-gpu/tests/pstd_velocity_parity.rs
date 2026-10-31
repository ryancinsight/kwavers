//! GPU PSTD versus CPU PSTD differential test: finite x-velocity burst.
//!
//! The case injects one identical finite x-velocity burst into both providers
//! and compares the same per-step sensor trace. Both providers route the
//! burst through the one shared `2·c₀·Δt/Δx` quotient, so the traces agree
//! directly, with no normalisation. The shared harness lives in
//! `parity/harness.rs`; the pressure-burst cases live in `pstd_parity.rs`.

#![cfg(feature = "gpu")]

// The velocity case exercises the trace-comparison subset of the shared
// harness; the field-comparison helpers serve the pressure target alone.
#[allow(dead_code)]
#[path = "parity/harness.rs"]
mod harness;

use harness::{max_rel_err, run_cpu_gpu_traces, LOSSLESS};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::{GridSource, SourceMode};
use leto::Array3;

/// Construct the shared finite x-velocity burst used by both PSTD providers.
///
/// The `[3, 1, nt]` signal carries the pulse on the ux row only: the CPU
/// injects all three rows (ux/uy/uz) while the GPU path carries ux, so a
/// uy/uz-silent signal is what both providers must agree on.
fn point_velocity_burst(shape: [usize; 3], nt: usize, source: [usize; 3]) -> GridSource {
    let mut u_mask = Array3::<f64>::zeros(shape);
    u_mask[[source[0], source[1], source[2]]] = 1.0;
    let mut u_signal = Array3::<f64>::zeros((3, 1, nt));
    u_signal[[0, 0, 1]] = 1.0;
    GridSource {
        u_mask: Some(u_mask),
        u_signal: Some(u_signal),
        u_mode: SourceMode::AdditiveNoCorrection,
        ..GridSource::new_empty()
    }
}

/// Velocity sources reach both providers through the one shared
/// `2·c₀·Δt/Δx` quotient: the sensor traces agree directly, with no
/// normalisation, at CFL 0.2 and at CFL 0.3.
#[test]
#[ignore = "requires a GPU device; run on the scheduled self-hosted GPU parity job"]
fn pstd_cpu_gpu_parity_velocity_source() {
    let shape = [8, 8, 8];
    let nt = 4;
    let dx = 1.0e-3;
    let c0 = 1_500.0;
    let rho0 = 1_000.0;
    let grid = Grid::new(shape[0], shape[1], shape[2], dx, dx, dx).expect("valid grid");
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    let source = [3, 4, 4];
    let grid_source = point_velocity_burst(shape, nt, source);
    for cfl in [0.2, 0.3] {
        let dt = cfl * dx / c0;
        let (cpu, gpu) = run_cpu_gpu_traces(&grid, &medium, &grid_source, source, nt, dt, LOSSLESS);
        let trace_error = max_rel_err(&cpu, &gpu);
        assert!(
            trace_error < 1e-3,
            "velocity-source GPU/CPU mismatch at CFL {cfl}: trace={trace_error:.2e}"
        );
    }
}
