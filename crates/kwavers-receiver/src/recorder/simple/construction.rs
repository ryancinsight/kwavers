//! SensorRecorder constructor methods.
//!
//! All constructors share the invariant that `sensor_indices` is ordered
//! x-fastest (Fortran order) matching k-Wave's `sensor.mask` convention.

use crate::recorder::config::RecordingMode;
use crate::recorder::fields::{SensorRecordField, SensorRecordSpec};
use crate::recorder::pressure_statistics::PressureFieldStatistics;
use crate::recorder::velocity_statistics::VelocityComponentStats;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use leto::{Array1, Array2, Array3 as LetoArray3};
use leto::Array3 as NdArray3;

use super::SensorRecorder;

#[doc(hidden)]
pub trait SensorMask3 {
    fn shape3(&self) -> [usize; 3];
    fn value_at(&self, i: usize, j: usize, k: usize) -> bool;
}

impl SensorMask3 for LetoArray3<bool> {
    fn shape3(&self) -> [usize; 3] {
        self.shape()
    }

    fn value_at(&self, i: usize, j: usize, k: usize) -> bool {
        self[[i, j, k]]
    }
}

impl SensorMask3 for NdArray3<bool> {
    fn shape3(&self) -> [usize; 3] {
        let dim = self.shape();
        [dim.0, dim.1, dim.2]
    }

    fn value_at(&self, i: usize, j: usize, k: usize) -> bool {
        self[[i, j, k]]
    }
}

impl SensorRecorder {
    /// Create a basic pressure-only recorder.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new<M>(
        sensor_mask: Option<&M>,
        shape: (usize, usize, usize),
        expected_steps: usize,
    ) -> KwaversResult<Self>
    where
        M: SensorMask3,
    {
        Self::with_modes(sensor_mask, shape, expected_steps, &[])
    }

    /// Create a recorder that also tracks pressure spatial statistics.
    ///
    /// Pass `modes` containing any of [`RecordingMode::MaxPressure`],
    /// [`RecordingMode::MinPressure`], [`RecordingMode::RmsPressure`],
    /// [`RecordingMode::FinalPressure`], or [`RecordingMode::AllStatistics`].
    /// An empty slice disables statistics (equivalent to `new`).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn with_modes<M>(
        sensor_mask: Option<&M>,
        shape: (usize, usize, usize),
        expected_steps: usize,
        modes: &[RecordingMode],
    ) -> KwaversResult<Self>
    where
        M: SensorMask3,
    {
        let sensor_indices = Self::build_sensor_indices(sensor_mask, shape)?;

        let pressure = if sensor_indices.is_empty() {
            None
        } else {
            Some(Array2::zeros([sensor_indices.len(), expected_steps]))
        };

        let needs_stats = modes.iter().any(|m| {
            matches!(
                m,
                RecordingMode::MaxPressure
                    | RecordingMode::MinPressure
                    | RecordingMode::RmsPressure
                    | RecordingMode::FinalPressure
                    | RecordingMode::AllStatistics
                    | RecordingMode::MaxMinPressure
                    | RecordingMode::MaxPressureAll
                    | RecordingMode::MinPressureAll
            )
        });

        let stats = if needs_stats {
            Some(PressureFieldStatistics::new(shape.0, shape.1, shape.2))
        } else {
            None
        };

        Ok(Self {
            sensor_indices,
            pressure,
            expected_steps,
            next_step: 0,
            stats,
            record_spec: SensorRecordSpec::pressure_only(),
            ux_data: None,
            uy_data: None,
            uz_data: None,
            ix_data: None,
            iy_data: None,
            iz_data: None,
            ix_sum: None,
            iy_sum: None,
            iz_sum: None,
            ux_stats: None,
            uy_stats: None,
            uz_stats: None,
        })
    }

    /// Create a recorder driven by a [`SensorRecordSpec`] (full k-Wave parity).
    ///
    /// Allocates velocity time-series buffers and statistics accumulators
    /// according to `spec`.  This is the preferred entry point for any
    /// simulation that records velocity or intensity.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn with_spec<M>(
        sensor_mask: Option<&M>,
        shape: (usize, usize, usize),
        expected_steps: usize,
        spec: SensorRecordSpec,
    ) -> KwaversResult<Self>
    where
        M: SensorMask3,
    {
        let sensor_indices = Self::build_sensor_indices(sensor_mask, shape)?;
        let n = sensor_indices.len();

        let pressure = if spec.records_pressure() && n > 0 {
            Some(Array2::zeros([n, expected_steps]))
        } else {
            None
        };

        let stats = if spec.needs_pressure_stats() {
            Some(PressureFieldStatistics::new(shape.0, shape.1, shape.2))
        } else {
            None
        };

        let alloc_ts = |needs: bool| -> Option<Array2<f64>> {
            if needs && n > 0 {
                Some(Array2::zeros([n, expected_steps]))
            } else {
                None
            }
        };
        let alloc_vstats = |needs: bool| -> Option<VelocityComponentStats> {
            if needs {
                Some(VelocityComponentStats::new(shape.0, shape.1, shape.2))
            } else {
                None
            }
        };

        Ok(Self {
            sensor_indices,
            pressure,
            expected_steps,
            next_step: 0,
            stats,
            ux_data: alloc_ts(spec.records_ux()),
            uy_data: alloc_ts(spec.records_uy()),
            uz_data: alloc_ts(spec.records_uz()),
            ix_data: alloc_ts(spec.contains(SensorRecordField::IntensityX)),
            iy_data: alloc_ts(spec.contains(SensorRecordField::IntensityY)),
            iz_data: alloc_ts(spec.contains(SensorRecordField::IntensityZ)),
            ix_sum: if spec.records_intensity_x() {
                Some(Array1::zeros([n]))
            } else {
                None
            },
            iy_sum: if spec.records_intensity_y() {
                Some(Array1::zeros([n]))
            } else {
                None
            },
            iz_sum: if spec.records_intensity_z() {
                Some(Array1::zeros([n]))
            } else {
                None
            },
            ux_stats: alloc_vstats(spec.needs_ux_stats()),
            uy_stats: alloc_vstats(spec.needs_uy_stats()),
            uz_stats: alloc_vstats(spec.needs_uz_stats()),
            record_spec: spec,
        })
    }

    /// Create a `SensorRecorder` from a pre-ordered list of grid indices.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn from_ordered_indices(
        indices: Vec<(usize, usize, usize)>,
        expected_steps: usize,
    ) -> KwaversResult<Self> {
        let pressure = if indices.is_empty() {
            None
        } else {
            Some(Array2::zeros([indices.len(), expected_steps]))
        };
        Ok(Self {
            sensor_indices: indices,
            pressure,
            expected_steps,
            next_step: 0,
            stats: None,
            record_spec: SensorRecordSpec::pressure_only(),
            ux_data: None,
            uy_data: None,
            uz_data: None,
            ix_data: None,
            iy_data: None,
            iz_data: None,
            ix_sum: None,
            iy_sum: None,
            iz_sum: None,
            ux_stats: None,
            uy_stats: None,
            uz_stats: None,
        })
    }

    // ── Internal helper ──────────────────────────────────────────────────────

    /// Build a Fortran-order (x-fastest) index list from a boolean sensor mask.
    ///
    /// Returns an empty `Vec` when `sensor_mask` is `None`.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn build_sensor_indices<M>(
        sensor_mask: Option<&M>,
        shape: (usize, usize, usize),
    ) -> KwaversResult<Vec<(usize, usize, usize)>>
    where
        M: SensorMask3,
    {
        let Some(mask) = sensor_mask else {
            return Ok(Vec::new());
        };
        let mask_dim = mask.shape3();
        let expected_shape = [shape.0, shape.1, shape.2];
        if mask_dim != expected_shape {
            if mask_dim == [1, 1, 1] && !mask.value_at(0, 0, 0) {
                return Ok(Vec::new());
            }
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Sensor mask shape mismatch: expected {shape:?}, got {mask_dim:?}"
                    ),
                },
            ));
        }
        // Fortran-order (x-fastest) for k-Wave compatibility.
        let (nx, ny, nz) = shape;
        let mut indices = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if mask.value_at(i, j, k) {
                        indices.push((i, j, k));
                    }
                }
            }
        }
        Ok(indices)
    }
}
