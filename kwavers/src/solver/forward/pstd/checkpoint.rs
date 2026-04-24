//! Binary checkpoint format for PSTD solver mid-simulation state persistence.
//!
//! # Format (little-endian throughout)
//!
//! ```text
//! Offset | Size | Type     | Field
//! -------|------|----------|----------------------------------------------
//! 0      |    4 | [u8; 4]  | Magic: b"KWCP"
//! 4      |    4 | u32      | Format version: 1
//! 8      |    8 | u64      | nx
//! 16     |    8 | u64      | ny
//! 24     |    8 | u64      | nz
//! 32     |    8 | u64      | time_step_index (steps completed)
//! 40     |    8 | u64      | total_steps (config.nt)
//! 48     |    8 | f64      | dt
//! 56     |    1 | u8       | has_sensor_data (0 or 1)
//!
//! [if has_sensor_data == 1]
//! 57     |    8 | u64      | n_sensors
//! 65     |    8 | u64      | n_recorded  (== next_step)
//! 73     |    8 | u64      | expected_steps (config.nt + 1)
//! 81     |    8*n_sensors*n_recorded | f64[] | pressure[sensor, step] row-major
//!
//! [after sensor block or at byte 57]
//! ...    |    8*nx*ny*nz | f64[] | p   (C-order)
//! ...    |    8*nx*ny*nz | f64[] | ux
//! ...    |    8*nx*ny*nz | f64[] | uy
//! ...    |    8*nx*ny*nz | f64[] | uz
//! ...    |    8*nx*ny*nz | f64[] | rhox
//! ...    |    8*nx*ny*nz | f64[] | rhoy
//! ...    |    8*nx*ny*nz | f64[] | rhoz
//! ```
//!
//! # Invariants
//! - Grid dimensions in the checkpoint must match the solver that loads it.
//! - `dt` must match to within 1e-20 (bit-exact continuation requires identical stepping).
//! - `n_recorded == next_step`: no gaps in the sensor time series.
//! - Field arrays are in C (row-major) order, matching ndarray's default layout.
//!
//! # References
//! - Treeby & Cox (2010) §4: k-Wave checkpointing convention (HDF5-based; we use binary).

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array2, Array3};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic bytes identifying a PSTD checkpoint file.
pub(crate) const MAGIC: [u8; 4] = *b"KWCP";
/// Binary format version — increment when the layout changes incompatibly.
pub(crate) const FORMAT_VERSION: u32 = 1;

/// Complete mutable state required to resume a PSTD simulation bit-exactly.
#[derive(Debug)]
///
/// The 7 primary acoustic field arrays, the current step counter, and any
/// partial sensor time series accumulated before the checkpoint are sufficient
/// to continue from an arbitrary point in the time loop.  All precomputed
/// spectral operators (kappa, shift operators, absorption kernels) are
/// deterministically rederived from the grid and `PSTDConfig` at construction
/// time and therefore need not be saved.
pub struct PSTDCheckpoint {
    /// Grid x-dimension (voxels).
    pub nx: usize,
    /// Grid y-dimension (voxels).
    pub ny: usize,
    /// Grid z-dimension (voxels).
    pub nz: usize,
    /// Number of time steps already completed when the checkpoint was taken.
    pub time_step_index: usize,
    /// Total simulation steps (`config.nt`).
    pub total_steps: usize,
    /// Time step size [s].
    pub dt: f64,
    pub p: Array3<f64>,
    pub ux: Array3<f64>,
    pub uy: Array3<f64>,
    pub uz: Array3<f64>,
    pub rhox: Array3<f64>,
    pub rhoy: Array3<f64>,
    pub rhoz: Array3<f64>,
    /// Sensor pressure time series recorded so far, shape `(n_sensors, n_recorded)`.
    /// `n_recorded == sensor_next_step`.  `None` when no sensor mask is active.
    pub sensor_data: Option<Array2<f64>>,
    /// Number of recorder columns already written (== `SensorRecorder::next_step`).
    pub sensor_next_step: usize,
    /// Total recorder capacity (`config.nt + 1`, one slot per step plus initial).
    pub sensor_expected_steps: usize,
}

impl PSTDCheckpoint {
    /// Serialize to a binary file at `path`.
    pub fn save(&self, path: &Path) -> KwaversResult<()> {
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(&MAGIC)?;
        w.write_all(&FORMAT_VERSION.to_le_bytes())?;
        w.write_all(&(self.nx as u64).to_le_bytes())?;
        w.write_all(&(self.ny as u64).to_le_bytes())?;
        w.write_all(&(self.nz as u64).to_le_bytes())?;
        w.write_all(&(self.time_step_index as u64).to_le_bytes())?;
        w.write_all(&(self.total_steps as u64).to_le_bytes())?;
        w.write_all(&self.dt.to_le_bytes())?;

        match &self.sensor_data {
            None => {
                w.write_all(&[0u8])?;
            }
            Some(data) => {
                w.write_all(&[1u8])?;
                let (n_sensors, n_recorded) = data.dim();
                w.write_all(&(n_sensors as u64).to_le_bytes())?;
                w.write_all(&(n_recorded as u64).to_le_bytes())?;
                w.write_all(&(self.sensor_expected_steps as u64).to_le_bytes())?;
                for &v in data.iter() {
                    w.write_all(&v.to_le_bytes())?;
                }
            }
        }

        for arr in [
            &self.p, &self.ux, &self.uy, &self.uz, &self.rhox, &self.rhoy, &self.rhoz,
        ] {
            for &v in arr.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }

        w.flush()?;
        Ok(())
    }

    /// Deserialize from a binary file at `path`.
    pub fn load(path: &Path) -> KwaversResult<Self> {
        let file = std::fs::File::open(path)?;
        let mut r = BufReader::new(file);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(KwaversError::InvalidInput(format!(
                "Invalid checkpoint magic: expected {:?}, got {:?}",
                MAGIC, magic
            )));
        }

        let version = read_u32(&mut r)?;
        if version != FORMAT_VERSION {
            return Err(KwaversError::InvalidInput(format!(
                "Unsupported checkpoint version {version}; expected {FORMAT_VERSION}"
            )));
        }

        let nx = read_u64(&mut r)? as usize;
        let ny = read_u64(&mut r)? as usize;
        let nz = read_u64(&mut r)? as usize;
        let time_step_index = read_u64(&mut r)? as usize;
        let total_steps = read_u64(&mut r)? as usize;
        let dt = read_f64(&mut r)?;

        let has_sensor = read_u8(&mut r)?;
        let (sensor_data, sensor_next_step, sensor_expected_steps) = if has_sensor == 1 {
            let n_sensors = read_u64(&mut r)? as usize;
            let n_recorded = read_u64(&mut r)? as usize;
            let expected_steps = read_u64(&mut r)? as usize;
            let mut flat = vec![0.0f64; n_sensors * n_recorded];
            for v in &mut flat {
                *v = read_f64(&mut r)?;
            }
            let data = Array2::from_shape_vec((n_sensors, n_recorded), flat)
                .map_err(KwaversError::Shape)?;
            (Some(data), n_recorded, expected_steps)
        } else {
            (None, 0, 0)
        };

        let n = nx * ny * nz;
        let p = read_f64_array3(&mut r, n, nx, ny, nz)?;
        let ux = read_f64_array3(&mut r, n, nx, ny, nz)?;
        let uy = read_f64_array3(&mut r, n, nx, ny, nz)?;
        let uz = read_f64_array3(&mut r, n, nx, ny, nz)?;
        let rhox = read_f64_array3(&mut r, n, nx, ny, nz)?;
        let rhoy = read_f64_array3(&mut r, n, nx, ny, nz)?;
        let rhoz = read_f64_array3(&mut r, n, nx, ny, nz)?;

        Ok(Self {
            nx,
            ny,
            nz,
            time_step_index,
            total_steps,
            dt,
            p,
            ux,
            uy,
            uz,
            rhox,
            rhoy,
            rhoz,
            sensor_data,
            sensor_next_step,
            sensor_expected_steps,
        })
    }

    /// Validate the checkpoint against a target PSTD solver configuration.
    pub fn validate_restore_contract(
        &self,
        expected_nx: usize,
        expected_ny: usize,
        expected_nz: usize,
        expected_total_steps: usize,
        expected_dt: f64,
    ) -> KwaversResult<()> {
        if self.nx != expected_nx || self.ny != expected_ny || self.nz != expected_nz {
            return Err(KwaversError::DimensionMismatch(format!(
                "checkpoint grid ({},{},{}) ≠ solver grid ({},{},{})",
                self.nx, self.ny, self.nz, expected_nx, expected_ny, expected_nz
            )));
        }
        if self.total_steps != expected_total_steps {
            return Err(KwaversError::InvalidInput(format!(
                "checkpoint total_steps {} ≠ solver total_steps {}",
                self.total_steps, expected_total_steps
            )));
        }
        if self.time_step_index > expected_total_steps {
            return Err(KwaversError::InvalidInput(format!(
                "checkpoint time_step_index {} exceeds total_steps {}",
                self.time_step_index, expected_total_steps
            )));
        }
        if (self.dt - expected_dt).abs() > 1e-20 {
            return Err(KwaversError::InvalidInput(format!(
                "checkpoint dt {} ≠ solver dt {}",
                self.dt, expected_dt
            )));
        }
        if self.sensor_data.is_some() {
            let expected_sensor_steps = expected_total_steps.checked_add(1).ok_or_else(|| {
                KwaversError::InvalidInput(
                    "expected_total_steps overflow when computing recorder capacity".to_string(),
                )
            })?;
            if self.sensor_expected_steps != expected_sensor_steps {
                return Err(KwaversError::InvalidInput(format!(
                    "checkpoint sensor_expected_steps {} ≠ expected {}",
                    self.sensor_expected_steps, expected_sensor_steps
                )));
            }
        } else if self.sensor_next_step != 0 || self.sensor_expected_steps != 0 {
            return Err(KwaversError::InvalidInput(
                "checkpoint sensor metadata present without sensor data".to_string(),
            ));
        }
        Ok(())
    }
}

fn read_u8(r: &mut impl Read) -> KwaversResult<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u32(r: &mut impl Read) -> KwaversResult<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> KwaversResult<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> KwaversResult<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_f64_array3(
    r: &mut impl Read,
    n: usize,
    nx: usize,
    ny: usize,
    nz: usize,
) -> KwaversResult<Array3<f64>> {
    let mut flat = vec![0.0f64; n];
    for v in &mut flat {
        *v = read_f64(r)?;
    }
    Array3::from_shape_vec((nx, ny, nz), flat).map_err(KwaversError::Shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::domain::medium::HomogeneousMedium;
    use crate::domain::sensor::recorder::simple::SensorRecorder;
    use crate::domain::source::GridSource;
    use crate::solver::forward::pstd::config::{CompatibilityMode, PSTDConfig};
    use crate::solver::forward::pstd::PSTDSolver;
    use ndarray::Array3;

    fn build_solver_with_sensor(nx: usize, ny: usize, nz: usize, nt: usize, dt: f64) -> PSTDSolver {
        let config = PSTDConfig {
            compatibility_mode: CompatibilityMode::Reference,
            dt,
            nt,
            ..Default::default()
        };
        let grid = Grid::new(nx, ny, nz, 1e-3, 1e-3, 1e-3).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
        let source = GridSource::new_empty();
        let mut solver = PSTDSolver::new(config, grid.clone(), &medium, source).unwrap();

        // Single-point sensor at grid centre for bit-exact comparison
        let mut mask = Array3::<bool>::from_elem((nx, ny, nz), false);
        mask[[nx / 2, ny / 2, nz / 2]] = true;
        solver.sensor_recorder = SensorRecorder::new(Some(&mask), (nx, ny, nz), nt + 1).unwrap();
        solver
    }

    /// Theorem: for a PSTD simulation split at step k, the sensor time series
    /// produced by running k steps → checkpoint → restore → (nt−k) steps is
    /// bit-exact (f64 equality) to running nt steps without checkpointing.
    ///
    /// Proof: the restored state (p, ux, uy, uz, rhox, rhoy, rhoz) is a lossless
    /// binary copy.  All spectral operators (kappa, shift operators) are derived
    /// deterministically from the fixed grid and config.  Thus the numerical
    /// trajectory after restoration is identical.
    #[test]
    fn test_checkpoint_bit_exact_continuation() {
        const NX: usize = 16;
        const NY: usize = 16;
        const NZ: usize = 16;
        const NT: usize = 20;
        const DT: f64 = 1e-8;
        const SPLIT: usize = 10; // checkpoint at half-way

        // Inject a non-trivial initial pressure so the fields are non-zero
        let mut solver_ref = build_solver_with_sensor(NX, NY, NZ, NT, DT);
        solver_ref.fields.p[[NX / 4, NY / 4, NZ / 4]] = 1.0e5;
        solver_ref.rhox[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * 1500.0_f64.powi(2));
        solver_ref.rhoy[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * 1500.0_f64.powi(2));
        solver_ref.rhoz[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * 1500.0_f64.powi(2));

        // Reference: run all NT steps without checkpoint
        let ref_data = solver_ref.run_orchestrated(NT).unwrap().unwrap();

        // Checkpoint run: use a tempfile to avoid leaving files around
        let tmp_dir = std::env::temp_dir();
        let ckpt_path = tmp_dir.join("kwavers_test_checkpoint.bin");
        let _ = std::fs::remove_file(&ckpt_path); // clean up any leftovers

        let mut solver_ckpt = build_solver_with_sensor(NX, NY, NZ, NT, DT);
        solver_ckpt.fields.p[[NX / 4, NY / 4, NZ / 4]] = 1.0e5;
        solver_ckpt.rhox[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * 1500.0_f64.powi(2));
        solver_ckpt.rhoy[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * 1500.0_f64.powi(2));
        solver_ckpt.rhoz[[NX / 4, NY / 4, NZ / 4]] = 1.0e5 / (3.0 * 1500.0_f64.powi(2));

        solver_ckpt.run_to_checkpoint(SPLIT, &ckpt_path).unwrap();
        assert!(ckpt_path.exists(), "Checkpoint file must be written");

        // Resume solver
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

        // Bit-exact comparison: every sensor sample must match to the last f64 bit.
        // The run_orchestrated path records step 0 + steps 1..NT = NT+1 columns total.
        // resumed_data has the same NT+1 columns; ref_data also has NT+1 columns.
        assert_eq!(
            ref_data.dim(),
            resumed_data.dim(),
            "Sensor data shape mismatch"
        );
        for col in 0..ref_data.ncols() {
            for row in 0..ref_data.nrows() {
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
            Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                (i * ny * nz + j * nz + k) as f64 + offset
            })
        };

        let sensor_data = ndarray::Array2::from_shape_fn((3, 5), |(i, j)| (i * 5 + j) as f64);

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
        match total_steps_err {
            KwaversError::InvalidInput(msg) => {
                assert!(
                    msg.contains("total_steps"),
                    "expected total_steps validation message, got {msg}"
                );
            }
            other => panic!("expected InvalidInput for total_steps mismatch, got {other:?}"),
        }

        let dt_err = ckpt
            .validate_restore_contract(2, 2, 2, 4, 2.0e-8)
            .unwrap_err();
        match dt_err {
            KwaversError::InvalidInput(msg) => {
                assert!(
                    msg.contains("dt"),
                    "expected dt validation message, got {msg}"
                );
            }
            other => panic!("expected InvalidInput for dt mismatch, got {other:?}"),
        }
    }
}
