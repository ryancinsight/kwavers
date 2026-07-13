//! `PSTDCheckpoint` struct, serialisation, and IO helpers.
//!
//! # Binary format (little-endian throughout)
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
//! - `dt` must match to within 1e-20 (bit-exact continuation).
//! - `n_recorded == next_step`: no gaps in the sensor time series.
//! - Field arrays are in C (row-major) order, matching ndarray's default layout.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3 as LetoArray3;
use leto::{Array2, Array3};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic bytes identifying a PSTD checkpoint file.
pub(crate) const MAGIC: [u8; 4] = *b"KWCP";
/// Binary format version — increment on incompatible layout changes.
pub(crate) const FORMAT_VERSION: u32 = 1;

/// Complete mutable state required to resume a PSTD simulation bit-exactly.
///
/// The 7 primary acoustic field arrays, the current step counter, and any
/// partial sensor time series accumulated before the checkpoint are sufficient
/// to continue from an arbitrary point in the time loop.  All precomputed
/// spectral operators (kappa, shift operators, absorption kernels) are
/// deterministically rederived from the grid and `PSTDConfig` at construction
/// time and therefore need not be saved.
#[derive(Debug)]
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
    /// Time step size (s).
    pub dt: f64,
    pub p: LetoArray3<f64>,
    pub ux: LetoArray3<f64>,
    pub uy: LetoArray3<f64>,
    pub uz: LetoArray3<f64>,
    pub rhox: Array3<f64>,
    pub rhoy: Array3<f64>,
    pub rhoz: Array3<f64>,
    /// Sensor pressure time series recorded so far, shape `(n_sensors, n_recorded)`.
    /// `n_recorded == sensor_next_step`. `None` when no sensor mask is active.
    pub sensor_data: Option<Array2<f64>>,
    /// Number of recorder columns already written (`SensorRecorder::next_step`).
    pub sensor_next_step: usize,
    /// Total recorder capacity (`config.nt + 1`, one slot per step plus initial).
    pub sensor_expected_steps: usize,
}

impl PSTDCheckpoint {
    /// Serialize to a binary file at `path`.
    ///
    /// Prefer [`Self::save_borrowed`] in hot paths — it writes directly from borrowed
    /// field slices without cloning the arrays into a `PSTDCheckpoint` struct first.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn save(&self, path: &Path) -> KwaversResult<()> {
        Self::save_borrowed(
            path,
            self.nx,
            self.ny,
            self.nz,
            self.time_step_index,
            self.total_steps,
            self.dt,
            &self.p,
            &self.ux,
            &self.uy,
            &self.uz,
            &self.rhox,
            &self.rhoy,
            &self.rhoz,
            self.sensor_data.as_ref(),
            self.sensor_next_step,
            self.sensor_expected_steps,
        )
    }

    /// Serialize directly from borrowed field references — zero-clone path.
    ///
    /// Writes the KWCP binary header and 7 acoustic field arrays without
    /// allocating intermediate `Array3<f64>` copies.  Accepts all fields as
    /// borrowed references so the caller (e.g. `PSTDSolver::run_to_checkpoint`)
    /// can pass `&self.fields.p`, `&self.fields.ux`, etc. directly.
    ///
    /// # Memory savings
    /// For a 256³ grid, this avoids 7 × 256³ × 8 = 896 MiB of intermediate
    /// allocations and copies per checkpoint vs. the struct-based `save()` path.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn save_borrowed(
        path: &Path,
        nx: usize,
        ny: usize,
        nz: usize,
        time_step_index: usize,
        total_steps: usize,
        dt: f64,
        p: &LetoArray3<f64>,
        ux: &LetoArray3<f64>,
        uy: &LetoArray3<f64>,
        uz: &LetoArray3<f64>,
        rhox: &Array3<f64>,
        rhoy: &Array3<f64>,
        rhoz: &Array3<f64>,
        sensor_data: Option<&Array2<f64>>,
        _sensor_next_step: usize,
        sensor_expected_steps: usize,
    ) -> KwaversResult<()> {
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(&MAGIC)?;
        w.write_all(&FORMAT_VERSION.to_le_bytes())?;
        w.write_all(&(nx as u64).to_le_bytes())?;
        w.write_all(&(ny as u64).to_le_bytes())?;
        w.write_all(&(nz as u64).to_le_bytes())?;
        w.write_all(&(time_step_index as u64).to_le_bytes())?;
        w.write_all(&(total_steps as u64).to_le_bytes())?;
        w.write_all(&dt.to_le_bytes())?;

        match sensor_data {
            None => {
                w.write_all(&[0u8])?;
            }
            Some(data) => {
                w.write_all(&[1u8])?;
                let [n_sensors, n_recorded] = data.shape();
                w.write_all(&(n_sensors as u64).to_le_bytes())?;
                w.write_all(&(n_recorded as u64).to_le_bytes())?;
                w.write_all(&(sensor_expected_steps as u64).to_le_bytes())?;
                for &v in data.iter() {
                    w.write_all(&v.to_le_bytes())?;
                }
            }
        }

        write_f64_iter(&mut w, p.iter())?;
        write_f64_iter(&mut w, ux.iter())?;
        write_f64_iter(&mut w, uy.iter())?;
        write_f64_iter(&mut w, uz.iter())?;
        write_f64_iter(&mut w, rhox.iter())?;
        write_f64_iter(&mut w, rhoy.iter())?;
        write_f64_iter(&mut w, rhoz.iter())?;

        w.flush()?;
        Ok(())
    }

    /// Deserialize from a binary file at `path`.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
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

        let version: u32 = read_le(&mut r)?;
        if version != FORMAT_VERSION {
            return Err(KwaversError::InvalidInput(format!(
                "Unsupported checkpoint version {version}; expected {FORMAT_VERSION}"
            )));
        }

        let nx = read_le::<u64>(&mut r)? as usize;
        let ny = read_le::<u64>(&mut r)? as usize;
        let nz = read_le::<u64>(&mut r)? as usize;
        let time_step_index = read_le::<u64>(&mut r)? as usize;
        let total_steps = read_le::<u64>(&mut r)? as usize;
        let dt: f64 = read_le(&mut r)?;

        let has_sensor: u8 = read_le(&mut r)?;
        let (sensor_data, sensor_next_step, sensor_expected_steps) = if has_sensor == 1 {
            let n_sensors = read_le::<u64>(&mut r)? as usize;
            let n_recorded = read_le::<u64>(&mut r)? as usize;
            let expected_steps = read_le::<u64>(&mut r)? as usize;
            let mut flat = vec![0.0f64; n_sensors * n_recorded];
            for v in &mut flat {
                *v = read_le(&mut r)?;
            }
            let data = Array2::from_shape_vec([n_sensors, n_recorded], flat)
                .map_err(|e| KwaversError::Shape(e.to_string()))?;
            (Some(data), n_recorded, expected_steps)
        } else {
            (None, 0, 0)
        };

        let n = nx * ny * nz;
        let p = read_leto_array3(&mut r, n, nx, ny, nz)?;
        let ux = read_leto_array3(&mut r, n, nx, ny, nz)?;
        let uy = read_leto_array3(&mut r, n, nx, ny, nz)?;
        let uz = read_leto_array3(&mut r, n, nx, ny, nz)?;
        let rhox = read_array3(&mut r, n, nx, ny, nz)?;
        let rhoy = read_array3(&mut r, n, nx, ny, nz)?;
        let rhoz = read_array3(&mut r, n, nx, ny, nz)?;

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
    ///
    /// Returns `Err` if any dimension, step count, or `dt` diverges from expectations.
    /// # Errors
    /// - Returns [`crate::KwaversError::DimensionMismatch`] if the precondition for mismatched array or grid dimensions is violated.
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
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
                    "expected_total_steps overflow when computing recorder capacity".to_owned(),
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
                "checkpoint sensor metadata present without sensor data".to_owned(),
            ));
        }
        Ok(())
    }
}

// --- Private IO helpers ---

/// Sealed trait for little-endian wire-format primitives used in the KWCP
/// binary format.  All arithmetic types that appear in checkpoint files
/// implement this trait; the sealed design prevents external callers from
/// accidentally extending it to types whose memory representation does not
/// match the on-disk layout.
mod wire_primitive_seal {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for f64 {}
}

trait WirePrimitive: wire_primitive_seal::Sealed + Default + Copy {
    /// Number of bytes occupied by this type in the wire format.
    const WIRE_BYTES: usize;
    /// Decode `bytes[..Self::WIRE_BYTES]` as a little-endian value.
    fn from_le_wire(bytes: &[u8]) -> Self;
}

impl WirePrimitive for u8 {
    const WIRE_BYTES: usize = 1;
    #[inline]
    fn from_le_wire(bytes: &[u8]) -> Self {
        bytes[0]
    }
}

impl WirePrimitive for u32 {
    const WIRE_BYTES: usize = 4;
    #[inline]
    fn from_le_wire(bytes: &[u8]) -> Self {
        u32::from_le_bytes(bytes[..4].try_into().expect("u32 wire slice"))
    }
}

impl WirePrimitive for u64 {
    const WIRE_BYTES: usize = 8;
    #[inline]
    fn from_le_wire(bytes: &[u8]) -> Self {
        u64::from_le_bytes(bytes[..8].try_into().expect("u64 wire slice"))
    }
}

impl WirePrimitive for f64 {
    const WIRE_BYTES: usize = 8;
    #[inline]
    fn from_le_wire(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes[..8].try_into().expect("f64 wire slice"))
    }
}

/// Read one little-endian wire primitive from `r`.
fn read_le<T: WirePrimitive>(r: &mut impl Read) -> KwaversResult<T> {
    let mut buf = [0u8; 8]; // 8 bytes covers all WirePrimitive impls (max size = f64/u64)
    let bytes = &mut buf[..T::WIRE_BYTES];
    r.read_exact(bytes)?;
    Ok(T::from_le_wire(bytes))
}

/// Read `n` little-endian `T` values from `r` and reshape into an
/// `(nx, ny, nz)` `Array3<T>`.
fn read_array3<T: WirePrimitive>(
    r: &mut impl Read,
    n: usize,
    nx: usize,
    ny: usize,
    nz: usize,
) -> KwaversResult<Array3<T>> {
    let mut flat = vec![T::default(); n];
    for v in &mut flat {
        *v = read_le(r)?;
    }
    Array3::from_shape_vec([nx, ny, nz], flat).map_err(|e| KwaversError::Shape(e.to_string()))
}

fn write_f64_iter<'a>(
    w: &mut impl Write,
    values: impl IntoIterator<Item = &'a f64>,
) -> KwaversResult<()> {
    for value in values {
        w.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_leto_array3(
    r: &mut impl Read,
    n: usize,
    nx: usize,
    ny: usize,
    nz: usize,
) -> KwaversResult<LetoArray3<f64>> {
    let mut flat = vec![0.0_f64; n];
    for v in &mut flat {
        *v = read_le(r)?;
    }
    LetoArray3::from_shape_vec([nx, ny, nz], flat).map_err(|e| KwaversError::Shape(e.to_string()))
}
