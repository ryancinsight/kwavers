//! End-to-end resource refusal on the normal Hephaestus dispatch path.

use super::gpu_request;
use crate::SimulationRunner;
use kwavers_core::error::KwaversError;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use leto::Array3;
use tracing_subscriber::fmt::format::FmtSpan;

#[test]
#[ignore = "requires a GPU device"]
fn declines_bluestein_workspace_over_the_buffer_limit() {
    // The X axis needs next_power_of_two(2 × 129 − 1) = 512 staged
    // samples per (Y, Z) line. One component occupies
    // 512 × 256 × 264 × 4 = 138,412,032 bytes, over the 128 MiB limit.
    assert_fft_refusal(
        [129, 256, 264],
        None,
        "FFT workspace requires 138412032, exceeding device max_buffer_size (bytes)=134217728",
    );
}

#[test]
#[ignore = "requires a GPU device"]
fn declines_bluestein_dispatch_over_the_workgroup_limit() {
    // Each axis stages 512 × 200 × 200 scalar elements. Their storage
    // fits, but 20,480,000 / 256 = 80,000 workgroups exceeds 65,535.
    let message =
        "FFT dispatch requires 80000, exceeding device max_compute_workgroups_per_dimension (workgroups)=65535";
    for sensor in [None, Some([100, 100, 100])] {
        assert_fft_refusal([200, 200, 200], sensor, message);
    }
}

fn assert_fft_refusal(shape: [usize; 3], sensor: Option<[usize; 3]>, expected: &str) {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let (grid, medium, sensor_mask) =
        tracing::info_span!("arrange", ?shape, ?sensor).in_scope(|| {
            let [nx, ny, nz] = shape;
            let grid = Grid::new(nx, ny, nz, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
            let mut medium = HomogeneousMedium::new(1000.0, 1500.0, 0.01, 0.1, &grid);
            medium
                .set_acoustic_properties(0.0, 0.0, 0.0)
                .expect("valid lossless acoustic medium");
            let sensor_mask = sensor.map(|position| {
                let mut mask = Array3::from_elem(shape, false);
                mask[position] = true;
                mask
            });
            (grid, medium, sensor_mask)
        });
    let request = crate::types::SimulationRunRequest {
        time_steps: 1,
        sensor_mask,
        ..gpu_request(&grid, &medium)
    };

    let result =
        tracing::info_span!("act").in_scope(|| SimulationRunner::run(&request, Vec::new()));
    tracing::info_span!("assert").in_scope(|| {
        let Err(KwaversError::ResourceLimitExceeded { message }) = result else {
            panic!("expected a typed capacity refusal, got {result:?}");
        };
        assert_eq!(message, expected);
    });
}
