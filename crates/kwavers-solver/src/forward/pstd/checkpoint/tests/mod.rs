//! Checkpoint correctness tests for [`PSTDSolver`].
//!
//! Split into leaf modules by test concern:
//! - `mod velocity_recording`: velocity-field survival across checkpoint boundary.

mod velocity_recording;

use super::data::PSTDCheckpoint;
use crate::forward::pstd::config::{CompatibilityMode, PSTDConfig};
use crate::forward::pstd::PSTDSolver;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::error::KwaversError;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_receiver::recorder::simple::SensorRecorder;
use kwavers_source::GridSource;
use leto::Array3;

pub(super) fn build_solver_with_sensor(
    nx: usize,
    ny: usize,
    nz: usize,
    nt: usize,
    dt: f64,
) -> PSTDSolver {
    let config = PSTDConfig {
        compatibility_mode: CompatibilityMode::Reference,
        dt,
        nt,
        ..Default::default()
    };
    let grid = Grid::new(nx, ny, nz, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let source = GridSource::new_empty();
    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, source).unwrap();

    let mut mask = Array3::<bool>::from_elem([nx, ny, nz], false);
    mask[[nx / 2, ny / 2, nz / 2]] = true;
    solver.sensor_recorder = SensorRecorder::new(Some(&mask), (nx, ny, nz), nt + 1).unwrap();
    solver
}

/// Theorem: for a PSTD simulation split at step k, the sensor time series
/// produced by running k steps → checkpoint → restore → (nt−k) steps is
/// bit-exact (f64 equality) to running nt steps without checkpointing.
///
/// Proof: the restored state (p, ux, uy, uz, rhox, rhoy, rhoz) is a lossless
/// binary copy. All spectral operators (kappa, shift operators) are derived
/// deterministically from the fixed grid and config. Thus the numerical
/// trajectory after restoration is identical.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_checkpoint_bit_exact_continuation() {
    const NX: usize = 16;
    const NY: usize = 16;
    const NZ: usize = 16;
    const NT: usize = 20;
    const DT: f64 = 1e-8;
    const SPLIT: usize = 10;

    let mut solver_ref = build_solver_with_sensor(NX, NY, NZ, NT, DT);
    solver_ref.fields.p[[NX / 4, NY / 4, NZ / 4]] = 1.0e5;
    solver_ref.rhox[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));
    solver_ref.rhoy[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));
    solver_ref.rhoz[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));

    let ref_data = solver_ref.run_orchestrated(NT).unwrap().unwrap();

    let tmp_dir = std::env::temp_dir();
    let ckpt_path = tmp_dir.join("kwavers_test_checkpoint.bin");
    let _ = std::fs::remove_file(&ckpt_path);

    let mut solver_ckpt = build_solver_with_sensor(NX, NY, NZ, NT, DT);
    solver_ckpt.fields.p[[NX / 4, NY / 4, NZ / 4]] = 1.0e5;
    solver_ckpt.rhox[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));
    solver_ckpt.rhoy[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));
    solver_ckpt.rhoz[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));

    solver_ckpt.run_to_checkpoint(SPLIT, &ckpt_path).unwrap();
    assert!(ckpt_path.exists(), "Checkpoint file must be written");

    let mut solver_resume = build_solver_with_sensor(NX, NY, NZ, NT, DT);
    let remaining = NT - SPLIT;
    let resumed_data = solver_resume
        .run_from_checkpoint(&ckpt_path, remaining)
        .unwrap()
        .unwrap();

    assert!(
        !ckpt_path.exists(),
        "Checkpoint file must be deleted after successful restore"
    );

    assert_eq!(
        ref_data.shape(),
        resumed_data.shape(),
        "Sensor data shape mismatch"
    );
    for col in 0..ref_data.shape()[1] {
        for row in 0..ref_data.shape()[0] {
            let r = ref_data[[row, col]];
            let c = resumed_data[[row, col]];
            assert_eq!(
                r.to_bits(),
                c.to_bits(),
                "Bit mismatch at sensor {row} step {col}: ref={r:.6e} resumed={c:.6e}"
            );
        }
    }
}

/// Verify PSTDCheckpoint serialisation round-trip (field data preserved exactly).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_checkpoint_roundtrip_serialisation() {
    let tmp_dir = std::env::temp_dir();
    let ckpt_path = tmp_dir.join("kwavers_test_ckpt_rt.bin");
    let _ = std::fs::remove_file(&ckpt_path);

    let nx = 4;
    let ny = 4;
    let nz = 4;
    let n = nx * ny * nz;

    let make_arr = |offset: f64| -> Array3<f64> {
        Array3::from_shape_fn((nx, ny, nz), |[i, j, k]| {
            (i * ny * nz + j * nz + k) as f64 + offset
        })
    };

    let sensor_data = leto::Array2::from_shape_fn((3, 5), |[i, j]| (i * 5 + j) as f64);

    let ckpt = PSTDCheckpoint {
        nx,
        ny,
        nz,
        time_step_index: 7,
        total_steps: 20,
        dt: 1e-8,
        p: make_arr(0.0),
        ux: make_arr(1000.0),
        uy: make_arr(2000.0),
        uz: make_arr(3000.0),
        rhox: make_arr(4000.0),
        rhoy: make_arr(5000.0),
        rhoz: make_arr(6000.0),
        sensor_data: Some(sensor_data.clone()),
        sensor_next_step: 5,
        sensor_expected_steps: 21,
    };

    ckpt.save(&ckpt_path).unwrap();
    let loaded = PSTDCheckpoint::load(&ckpt_path).unwrap();

    assert_eq!(loaded.nx, nx);
    assert_eq!(loaded.ny, ny);
    assert_eq!(loaded.nz, nz);
    assert_eq!(loaded.time_step_index, 7);
    assert_eq!(loaded.total_steps, 20);
    assert!((loaded.dt - 1e-8).abs() < 1e-30);
    assert_eq!(loaded.sensor_next_step, 5);
    assert_eq!(loaded.sensor_expected_steps, 21);

    for idx in 0..n {
        let i = idx / (ny * nz);
        let j = (idx / nz) % ny;
        let k = idx % nz;
        assert_eq!(loaded.p[[i, j, k]].to_bits(), ckpt.p[[i, j, k]].to_bits());
        assert_eq!(
            loaded.rhoz[[i, j, k]].to_bits(),
            ckpt.rhoz[[i, j, k]].to_bits()
        );
    }

    let loaded_sd = loaded.sensor_data.unwrap();
    for i in 0..3 {
        for j in 0..5 {
            assert_eq!(loaded_sd[[i, j]].to_bits(), sensor_data[[i, j]].to_bits());
        }
    }

    let _ = std::fs::remove_file(&ckpt_path);
}

/// Validate that checkpoint restore rejects mismatched solver metadata.
/// # Panics
/// - Panics if `matching checkpoint metadata must validate`.
/// - Panics with `"expected InvalidInput for total_steps mismatch, got {total_steps_err:?}"`.
/// - Panics with `"expected InvalidInput for dt mismatch, got {dt_err:?}"`.
///
#[test]
fn test_checkpoint_validate_restore_contract_rejects_mismatch() {
    let ckpt = PSTDCheckpoint {
        nx: 2,
        ny: 2,
        nz: 2,
        time_step_index: 1,
        total_steps: 4,
        dt: 1.0e-8,
        p: Array3::zeros((2, 2, 2)),
        ux: Array3::zeros((2, 2, 2)),
        uy: Array3::zeros((2, 2, 2)),
        uz: Array3::zeros((2, 2, 2)),
        rhox: Array3::zeros((2, 2, 2)),
        rhoy: Array3::zeros((2, 2, 2)),
        rhoz: Array3::zeros((2, 2, 2)),
        sensor_data: None,
        sensor_next_step: 0,
        sensor_expected_steps: 0,
    };

    ckpt.validate_restore_contract(2, 2, 2, 4, 1.0e-8)
        .expect("matching checkpoint metadata must validate");

    let total_steps_err = ckpt
        .validate_restore_contract(2, 2, 2, 5, 1.0e-8)
        .unwrap_err();
    // Must be InvalidInput whose message references "total_steps".
    let KwaversError::InvalidInput(ref ts_msg) = total_steps_err else {
        panic!("expected InvalidInput for total_steps mismatch, got {total_steps_err:?}");
    };
    assert!(
        ts_msg.contains("total_steps"),
        "expected total_steps in validation message, got {ts_msg}"
    );

    let dt_err = ckpt
        .validate_restore_contract(2, 2, 2, 4, 2.0e-8)
        .unwrap_err();
    // Must be InvalidInput whose message references "dt".
    let KwaversError::InvalidInput(ref dt_msg) = dt_err else {
        panic!("expected InvalidInput for dt mismatch, got {dt_err:?}");
    };
    assert!(
        dt_msg.contains("dt"),
        "expected dt in validation message, got {dt_msg}"
    );
}
