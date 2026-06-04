//! Velocity-field survival across checkpoint boundary.
//!
//! ## Theorem
//! Velocity data recorded during `run_from_checkpoint` is numerically identical
//! to velocity data recorded over the full run without checkpointing.
//!
//! ## Proof sketch
//! `SensorRecorder::with_spec` allocates ux/uy buffers when the spec requests
//! them. `run_from_checkpoint_loaded` calls `step_forward` on the restored
//! solver, which invokes `sensor_recorder.record_velocity_step` on each step.
//! The restored state is bit-exact (proven in `test_checkpoint_bit_exact_continuation`),
//! so the velocity trajectory is identical and the recorded buffers match.

use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_domain::medium::HomogeneousMedium;
use kwavers_domain::sensor::recorder::fields::{SensorRecordField, SensorRecordSpec};
use kwavers_domain::sensor::recorder::simple::SensorRecorder;
use kwavers_domain::source::GridSource;
use crate::forward::pstd::config::{CompatibilityMode, PSTDConfig};
use crate::forward::pstd::PSTDSolver;
use ndarray::Array3;

fn build_with_velocity(nx: usize, ny: usize, nz: usize, nt: usize, dt: f64) -> PSTDSolver {
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

    let mut mask = Array3::<bool>::from_elem((nx, ny, nz), false);
    mask[[nx / 2, ny / 2, nz / 2]] = true;

    let spec = SensorRecordSpec::from_fields(&[
        SensorRecordField::Pressure,
        SensorRecordField::VelocityX,
        SensorRecordField::VelocityY,
        SensorRecordField::VelocityZ,
    ]);
    solver.sensor_recorder =
        SensorRecorder::with_spec(Some(&mask), (nx, ny, nz), nt + 1, spec).unwrap();
    solver
}

/// # Panics
/// - Panics if `reference ux must be recorded`.
/// - Panics if `reference uy must be recorded`.
/// - Panics if `resumed ux must be recorded — velocity must survive checkpoint resume`.
/// - Panics if `resumed uy must be recorded — velocity must survive checkpoint resume`.
///
#[test]
fn test_checkpoint_velocity_recording_survives_resume() {
    const NX: usize = 16;
    const NY: usize = 16;
    const NZ: usize = 16;
    const NT: usize = 20;
    const DT: f64 = 1e-8;
    const SPLIT: usize = 10;

    // Reference run: no checkpoint.
    let mut solver_ref = build_with_velocity(NX, NY, NZ, NT, DT);
    solver_ref.fields.p[[NX / 4, NY / 4, NZ / 4]] = 1.0e5;
    solver_ref.rhox[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));
    solver_ref.rhoy[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));
    solver_ref.rhoz[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));
    solver_ref.run_orchestrated(NT).unwrap();
    let ref_ux = solver_ref
        .sensor_recorder
        .extract_ux_data()
        .expect("reference ux must be recorded");
    let ref_uy = solver_ref
        .sensor_recorder
        .extract_uy_data()
        .expect("reference uy must be recorded");

    assert!(
        ref_ux.iter().any(|&v| v != 0.0),
        "reference ux must contain non-zero values"
    );

    // Checkpoint run.
    let tmp_dir = std::env::temp_dir();
    let ckpt_path = tmp_dir.join("kwavers_test_velocity_checkpoint.bin");
    let _ = std::fs::remove_file(&ckpt_path);

    let mut solver_ckpt = build_with_velocity(NX, NY, NZ, NT, DT);
    solver_ckpt.fields.p[[NX / 4, NY / 4, NZ / 4]] = 1.0e5;
    solver_ckpt.rhox[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));
    solver_ckpt.rhoy[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));
    solver_ckpt.rhoz[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * SOUND_SPEED_WATER_SIM.powi(2));
    solver_ckpt.run_to_checkpoint(SPLIT, &ckpt_path).unwrap();

    let mut solver_resume = build_with_velocity(NX, NY, NZ, NT, DT);
    let remaining = NT - SPLIT;
    solver_resume
        .run_from_checkpoint(&ckpt_path, remaining)
        .unwrap();

    let resumed_ux = solver_resume
        .sensor_recorder
        .extract_ux_data()
        .expect("resumed ux must be recorded — velocity must survive checkpoint resume");
    let resumed_uy = solver_resume
        .sensor_recorder
        .extract_uy_data()
        .expect("resumed uy must be recorded — velocity must survive checkpoint resume");

    // GridDimension invariant.
    assert_eq!(
        ref_ux.dim(),
        resumed_ux.dim(),
        "ux shape mismatch after checkpoint resume"
    );

    // Value invariant: bit-exact match for the post-checkpoint steps.
    //
    // Both `run_orchestrated` and `run_to_checkpoint` record the initial pressure
    // (time_step_index == 0) before entering the time loop, consuming one recorder
    // slot at column 0 without writing velocity. After SPLIT steps,
    // `sensor_next_step = SPLIT + 1`. This value is persisted in the checkpoint.
    //
    // `restore_from_checkpoint` sets `next_step = SPLIT + 1` on the resumed
    // recorder. Velocity is NOT stored in the checkpoint, so velocity buffers are
    // zero-initialised at construction with `expected_steps = NT + 1`.
    //
    // `record_velocity_step` writes to column `self.next_step - 1` (after
    // `record_step` increments `next_step`). Therefore the first post-checkpoint
    // velocity sample lands at column SPLIT + 1, the second at SPLIT + 2, …,
    // the last at SPLIT + remaining.
    //
    // The reference run follows the same column mapping: velocity at sim step s
    // occupies column s (not s−1) because the initial slot is col 0.
    //
    // Both reference and resumed buffers share the same absolute column index
    // SPLIT + 1 + i for post-split step i.
    let vel_offset = SPLIT + 1; // = sensor_next_step captured by checkpoint
    for i in 0..remaining {
        let col = vel_offset + i;
        let r = ref_ux[[0, col]];
        let c = resumed_ux[[0, col]];
        assert_eq!(
            r.to_bits(),
            c.to_bits(),
            "ux bit mismatch at post-split step {i} (sim step {col}): ref={r:.6e} resumed={c:.6e}",
        );
    }
    for i in 0..remaining {
        let col = vel_offset + i;
        let r = ref_uy[[0, col]];
        let c = resumed_uy[[0, col]];
        assert_eq!(
            r.to_bits(),
            c.to_bits(),
            "uy bit mismatch at post-split step {i} (sim step {col}): ref={r:.6e} resumed={c:.6e}",
        );
    }
    let _ = std::fs::remove_file(&ckpt_path);
}
