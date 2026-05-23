use super::super::{FwiGeometry, FwiProcessor};
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::domain::grid::Grid;
use crate::domain::source::{GridSource, SourceMode};
use crate::solver::config::SolverType;
use crate::solver::inverse::seismic::parameters::FwiParameters;
use ndarray::{Array2, Array3};

/// Verify that `SolverType::PSTD` is accepted by `build_solver_for_forward` and
/// produces a non-trivial synthetic receiver trace.
///
/// ## Mathematical contract
///
/// The forward map `F_h(c; G)` returns an `Array2<f64>` of shape
/// `(N_receivers, nt)` with at least one non-zero entry when the source
/// produces a non-zero pressure field.  This is an input-sensitive smoke test
/// (any constant output would be rejected by `is_ok() + assert!` checks).
/// # Panics
/// - Panics on any assertion failure or solver error.
#[test]
fn test_fwi_pstd_solver_type_accepted_and_produces_nonzero_data() {
    let (nx, ny, nz) = (8usize, 8, 8);
    let dx = 1.5e-3_f64; // 1.5 mm — enough resolution for 500 kHz
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("grid");

    let model = Array3::from_elem((nx, ny, nz), SOUND_SPEED_WATER_SIM);

    let mut sensor_mask = Array3::from_elem((nx, ny, nz), false);
    for iy in 2..6 {
        for iz in 2..6 {
            sensor_mask[[6, iy, iz]] = true;
        }
    }

    let nt = 32usize;
    // dt satisfying CFL for PSTD: dt ≤ dx / (c * √3) * CFL.  Here CFL = 0.3.
    let dt = 1.7e-7_f64;

    let mut p_mask = Array3::from_elem((nx, ny, nz), 0.0_f64);
    p_mask[[2, 4, 4]] = 1.0;
    let mut p_signal = Array2::zeros((1, nt));
    for t in 0..8 {
        let phase = t as f64 * 0.4;
        p_signal[[0, t]] = (-phase * phase).exp() * (2.0 * phase).sin();
    }
    let source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: SourceMode::default(),
    };

    let parameters = FwiParameters {
        nt,
        dt,
        frequency: 5e5,
        solver_type: SolverType::PSTD,
        ..FwiParameters::default()
    };

    let processor = FwiProcessor::new(parameters);
    let geometry = FwiGeometry::new(source, sensor_mask);

    let synthetic = processor
        .generate_synthetic_data(&model, &geometry, &grid)
        .expect("PSTD forward model must succeed");

    assert_eq!(
        synthetic.nrows(),
        geometry.receiver_count(),
        "synthetic receiver count must match geometry"
    );
    assert_eq!(synthetic.ncols(), nt, "synthetic time length must match nt");

    // At least one non-zero receiver sample — any constant-zero output fails.
    let max_abs = synthetic
        .iter()
        .copied()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        max_abs > 0.0,
        "PSTD forward model must produce a non-zero receiver trace; got max_abs = {max_abs:e}"
    );
}

/// Verify that an unsupported `SolverType` is rejected with an error, not a panic.
/// # Panics
/// - Panics if the unsupported type is silently accepted.
#[test]
fn test_fwi_unsupported_solver_type_returns_error() {
    let (nx, ny, nz) = (8usize, 8, 8);
    let dx = 1e-3_f64;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).expect("grid");
    let model = Array3::from_elem((nx, ny, nz), SOUND_SPEED_WATER_SIM);

    let mut sensor_mask = Array3::from_elem((nx, ny, nz), false);
    sensor_mask[[6, 4, 4]] = true;

    let nt = 10usize;
    let dt = 1e-7_f64;
    let source = GridSource {
        p0: None,
        u0: None,
        p_mask: Some(Array3::zeros((nx, ny, nz))),
        p_signal: Some(Array2::zeros((1, nt))),
        p_mode: SourceMode::Additive,
        u_mask: None,
        u_signal: None,
        u_mode: SourceMode::default(),
    };

    let parameters = FwiParameters {
        nt,
        dt,
        frequency: 5e5,
        solver_type: SolverType::KSpace, // not yet supported
        ..FwiParameters::default()
    };

    let processor = FwiProcessor::new(parameters);
    let geometry = FwiGeometry::new(source, sensor_mask);

    let result = processor.generate_synthetic_data(&model, &geometry, &grid);
    assert!(
        result.is_err(),
        "unsupported SolverType::KSpace must return Err, not Ok"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("KSpace"),
        "error message must name the unsupported type; got: {msg}"
    );
}
