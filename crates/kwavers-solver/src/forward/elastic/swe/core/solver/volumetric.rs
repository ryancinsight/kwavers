//! Volumetric propagation and wavefront tracking for `ElasticWaveSolver`.

use super::super::super::integration::integrator::PreparedBodyForces;
use super::super::super::integration::TimeIntegrator;
use super::super::super::scratch::ElasticStepScratch;
use super::super::super::types::{
    ElasticBodyForceConfig, ElasticWaveField, VolumetricQualityMetrics, VolumetricSource,
    WaveFrontTracker,
};
use super::definition::ElasticWaveSolver;
use kwavers_core::error::{KwaversResult, NumericalError};
use leto::Array3;

mod tracking;

use tracking::{FullFieldRecorder, MagnitudeRecorder, SnapshotRecorder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotSchedule {
    stride: usize,
    capacity: usize,
}

fn snapshot_schedule(steps: usize, max_snapshots: usize) -> KwaversResult<SnapshotSchedule> {
    if max_snapshots < 2 {
        return Err(NumericalError::InvalidOperation(
            "Volumetric max_snapshots must be at least two".to_owned(),
        )
        .into());
    }
    let intervals = max_snapshots - 1;
    // A retained initial state leaves `max_snapshots - 1` intervals. Ceiling
    // division is required: floor division retained every state when `steps`
    // was only slightly larger than the configured bound.
    let stride = steps.div_ceil(intervals).max(1);
    let capacity = steps.div_ceil(stride) + 1;
    debug_assert!(capacity <= max_snapshots);
    Ok(SnapshotSchedule { stride, capacity })
}

impl ElasticWaveSolver {
    /// Propagate body-force-driven waves and retain every saved field.
    ///
    /// Use [`Self::track_volumetric_waves_with_body_forces`] when only the
    /// returned [`WaveFrontTracker`] is required. This operation preserves the
    /// full six-component [`ElasticWaveField`] history for callers that inspect
    /// displacement or velocity snapshots.
    ///
    /// # Errors
    ///
    /// Returns an error when body-force timing is inconsistent, propagation
    /// configuration is invalid, retained history storage cannot be reserved,
    /// or the integrator fails.
    pub fn propagate_volumetric_waves_with_body_forces(
        &self,
        body_forces: &[ElasticBodyForceConfig],
        push_times: &[f64],
        _sources: &[VolumetricSource],
    ) -> KwaversResult<(Vec<ElasticWaveField>, WaveFrontTracker)> {
        self.propagate_volumetric_body_forces::<FullFieldRecorder>(body_forces, push_times)
    }

    /// Propagate body-force-driven waves and return only wavefront tracking.
    ///
    /// Unlike [`Self::propagate_volumetric_waves_with_body_forces`], this
    /// operation retains one displacement magnitude per eligible tracking
    /// voxel and saved time instead of cloning all six field components.
    /// Simulation steps, snapshot cadence, and arrival detection are identical.
    ///
    /// # Errors
    ///
    /// Returns an error when body-force timing is inconsistent, propagation
    /// configuration is invalid, retained tracking storage cannot be reserved,
    /// or the integrator fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use kwavers_core::error::KwaversResult;
    /// use kwavers_grid::Grid;
    /// use kwavers_medium::HomogeneousMedium;
    /// use kwavers_solver::forward::elastic::swe::{
    ///     ElasticBodyForceConfig, ElasticWaveConfig, ElasticWaveSolver,
    /// };
    ///
    /// # fn main() -> KwaversResult<()> {
    /// let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3)?;
    /// let medium = HomogeneousMedium::new(1_000.0, 1_500.0, 0.5, 1.0, &grid);
    /// let solver = ElasticWaveSolver::new(&grid, &medium, ElasticWaveConfig::default())?;
    /// let force = ElasticBodyForceConfig::GaussianImpulse {
    ///     center_m: [3.5e-3; 3],
    ///     sigma_m: [1.0e-3; 3],
    ///     direction: [1.0, 0.0, 0.0],
    ///     t0_s: 0.0,
    ///     sigma_t_s: 1.0e-6,
    ///     impulse_n_per_m3_s: 1.0e6,
    /// };
    /// let tracker = solver.track_volumetric_waves_with_body_forces(&[force], &[0.0])?;
    /// assert_eq!(tracker.arrival_times.shape(), [8, 8, 8]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn track_volumetric_waves_with_body_forces(
        &self,
        body_forces: &[ElasticBodyForceConfig],
        push_times: &[f64],
    ) -> KwaversResult<WaveFrontTracker> {
        self.propagate_volumetric_body_forces::<MagnitudeRecorder>(body_forces, push_times)
    }

    fn propagate_volumetric_body_forces<R>(
        &self,
        body_forces: &[ElasticBodyForceConfig],
        push_times: &[f64],
    ) -> KwaversResult<R::Output>
    where
        R: SnapshotRecorder,
    {
        if body_forces.len() != push_times.len() {
            return Err(NumericalError::InvalidOperation(
                "body_forces and push_times must have the same length".to_owned(),
            )
            .into());
        }
        let mut shifted_forces = Vec::with_capacity(body_forces.len());
        for (bf, &t0) in body_forces.iter().zip(push_times.iter()) {
            let mut bf_shifted = bf.clone();
            let ElasticBodyForceConfig::GaussianImpulse { t0_s, .. } = &mut bf_shifted;
            *t0_s = t0;
            shifted_forces.push(bf_shifted);
        }
        let (nx, ny, nz) = self.grid.dimensions();
        let mut current_field = ElasticWaveField::new(nx, ny, nz);
        let integrator =
            TimeIntegrator::new(&self.grid, &self.lambda, &self.mu, &self.density, &self.pml);
        let dt = if self.config.time_step > 0.0 {
            self.config.time_step
        } else {
            integrator.calculate_stable_timestep(self.config.cfl_factor)
        };
        if dt <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Calculated time step is non-positive".to_owned(),
            )
            .into());
        }
        let duration_s = self.volumetric_config.duration_s;
        if !duration_s.is_finite() || duration_s <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Volumetric duration must be positive".to_owned(),
            )
            .into());
        }
        let steps = (duration_s / dt).ceil() as usize;
        let schedule = snapshot_schedule(steps, self.volumetric_config.max_snapshots)?;
        let mut recorder = R::try_new(self, schedule)?;
        let mut scratch = ElasticStepScratch::new(nx, ny, nz);
        let mut prepared_forces = PreparedBodyForces::new(&self.grid, &shifted_forces)?;
        recorder.record(self, &current_field);
        let mut last_recorded_time = current_field.time;
        for step_idx in 0..steps {
            integrator.step_with_prepared_body_forces(
                &mut current_field,
                dt,
                &mut prepared_forces,
                &mut scratch,
            )?;
            current_field.time += dt;
            if (step_idx + 1) % schedule.stride == 0 {
                recorder.record(self, &current_field);
                last_recorded_time = current_field.time;
            }
        }
        if last_recorded_time != current_field.time {
            recorder.record(self, &current_field);
        }
        Ok(recorder.finish(self))
    }
    /// Propagate volumetric waves with sources.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn propagate_volumetric_waves_with_sources(
        &self,
        initial_displacements: &[Array3<f64>],
        push_times: &[f64],
        sources: &[VolumetricSource],
    ) -> KwaversResult<(Vec<ElasticWaveField>, WaveFrontTracker)> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut initial_field = ElasticWaveField::new(nx, ny, nz);
        for disp in initial_displacements {
            if disp.shape() != [nx, ny, nz] {
                return Err(NumericalError::InvalidOperation(
                    "Initial displacement shape does not match grid".to_owned(),
                )
                .into());
            }
            for (a, b) in initial_field.uz.iter_mut().zip(disp.iter()) {
                *a += b;
            }
        }
        let integrator =
            TimeIntegrator::new(&self.grid, &self.lambda, &self.mu, &self.density, &self.pml);
        let dt = if self.config.time_step > 0.0 {
            self.config.time_step
        } else {
            integrator.calculate_stable_timestep(self.config.cfl_factor)
        };
        if dt <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Calculated time step is non-positive".to_owned(),
            )
            .into());
        }
        let duration_s = self.volumetric_config.duration_s;
        if !duration_s.is_finite() || duration_s <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Volumetric duration must be positive".to_owned(),
            )
            .into());
        }
        let steps = (duration_s / dt).ceil() as usize;
        let schedule = snapshot_schedule(steps, self.volumetric_config.max_snapshots)?;
        let mut scratch = ElasticStepScratch::new(nx, ny, nz);
        let mut current_field = initial_field;
        let mut history = Vec::with_capacity(schedule.capacity);
        history.push(current_field.clone());
        for step_idx in 0..steps {
            integrator.step(&mut current_field, dt, None, &mut scratch)?;
            current_field.time += dt;
            if (step_idx + 1) % schedule.stride == 0 {
                history.push(current_field.clone());
            }
        }
        let needs_final = match history.last() {
            None => true,
            Some(f) => f.time != current_field.time,
        };
        if needs_final {
            history.push(current_field.clone());
        }
        if !push_times.is_empty() && push_times.len() != sources.len() {
            return Err(NumericalError::InvalidOperation(
                "push_times and sources must have the same length when provided".to_owned(),
            )
            .into());
        }
        for w in push_times.windows(2) {
            if w[1] < w[0] {
                return Err(NumericalError::InvalidOperation(
                    "push_times must be non-decreasing".to_owned(),
                )
                .into());
            }
        }
        let _ = sources;
        let tracker = self.compute_wavefront_tracker(history.as_slice());
        Ok((history, tracker))
    }

    /// Summarize wavefront arrivals over voxels eligible for tracking.
    ///
    /// Coverage excludes PML cells and cells skipped by tracking decimation,
    /// matching the domain on which internal wavefront tracking attempts
    /// arrival detection.
    #[must_use]
    pub fn calculate_volumetric_quality(
        &self,
        tracker: &WaveFrontTracker,
    ) -> VolumetricQualityMetrics {
        let mut eligible = 0usize;
        let mut valid = 0usize;
        let mut quality_sum = 0.0;
        for (([i, j, k], &t), &q) in tracker
            .arrival_times
            .indexed_iter()
            .zip(tracker.tracking_quality.iter())
        {
            if !self.is_tracking_voxel(i, j, k) {
                continue;
            }
            eligible += 1;
            if t.is_finite() && q > 0.0 {
                valid += 1;
                quality_sum += q;
            }
        }
        VolumetricQualityMetrics {
            coverage: if eligible == 0 {
                0.0
            } else {
                valid as f64 / eligible as f64
            },
            average_quality: if valid == 0 {
                0.0
            } else {
                quality_sum / valid as f64
            },
            valid_tracking_points: valid,
        }
    }
}

#[cfg(test)]
mod tests;
