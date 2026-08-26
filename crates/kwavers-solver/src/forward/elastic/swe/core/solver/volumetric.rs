//! Volumetric propagation and wavefront tracking for `ElasticWaveSolver`.

use super::super::super::integration::integrator::PreparedBodyForces;
use super::super::super::integration::TimeIntegrator;
use super::super::super::scratch::ElasticStepScratch;
use super::super::super::types::{
    ArrivalDetection, ElasticBodyForceConfig, ElasticWaveField, VolumetricQualityMetrics,
    VolumetricSource, WaveFrontTracker,
};
use super::definition::ElasticWaveSolver;
use kwavers_core::error::{KwaversResult, NumericalError};
use leto::Array3;

#[derive(Clone, Copy, Debug, Default)]
struct WindowMetric {
    correlation: f64,
    quality: f64,
    amplitude: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotSchedule {
    stride: usize,
    capacity: usize,
}

fn snapshot_schedule(steps: usize, max_snapshots: usize) -> SnapshotSchedule {
    let max_snapshots = max_snapshots.max(2);
    let intervals = max_snapshots - 1;
    // A retained initial state leaves `max_snapshots - 1` intervals. Ceiling
    // division is required: floor division retained every state when `steps`
    // was only slightly larger than the configured bound.
    let stride = steps.div_ceil(intervals).max(1);
    let capacity = steps.div_ceil(stride) + 1;
    debug_assert!(capacity <= max_snapshots);
    SnapshotSchedule { stride, capacity }
}

impl ElasticWaveSolver {
    /// Propagate volumetric waves with body forces.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn propagate_volumetric_waves_with_body_forces(
        &self,
        body_forces: &[ElasticBodyForceConfig],
        push_times: &[f64],
        _sources: &[VolumetricSource],
    ) -> KwaversResult<(Vec<ElasticWaveField>, WaveFrontTracker)> {
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
        let schedule = snapshot_schedule(steps, self.volumetric_config.max_snapshots);
        let mut scratch = ElasticStepScratch::new(nx, ny, nz);
        let mut prepared_forces = PreparedBodyForces::new(&self.grid, &shifted_forces)?;
        let mut history = Vec::with_capacity(schedule.capacity);
        history.push(current_field.clone());
        for step_idx in 0..steps {
            integrator.step_with_prepared_body_forces(
                &mut current_field,
                dt,
                &mut prepared_forces,
                &mut scratch,
            )?;
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
        let tracker = self.compute_wavefront_tracker(&history);
        Ok((history, tracker))
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
        let schedule = snapshot_schedule(steps, self.volumetric_config.max_snapshots);
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
        let tracker = self.compute_wavefront_tracker(&history);
        Ok((history, tracker))
    }

    /// Summarize wavefront arrivals over voxels eligible for tracking.
    ///
    /// Coverage excludes PML cells and cells skipped by tracking decimation,
    /// matching the domain on which [`Self::compute_wavefront_tracker`]
    /// attempts arrival detection.
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

    pub(super) fn compute_wavefront_tracker(
        &self,
        history: &[ElasticWaveField],
    ) -> WaveFrontTracker {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut arrival_times = Array3::<f64>::from_elem([nx, ny, nz], f64::NAN);
        let mut amplitudes = Array3::<f64>::zeros((nx, ny, nz));
        let mut tracking_quality = Array3::<f64>::zeros((nx, ny, nz));

        if history.len() < 2 {
            return WaveFrontTracker {
                arrival_times,
                amplitudes,
                tracking_quality,
            };
        }

        let sample_count = history.len();
        let mut series = vec![0.0_f64; sample_count];
        let mut smoothed = vec![0.0_f64; sample_count];
        let mut diff_series = vec![0.0_f64; sample_count];
        let mut window_metrics = vec![WindowMetric::default(); sample_count];

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if !self.is_tracking_voxel(i, j, k) {
                        continue;
                    }
                    // Track the displacement magnitude, not a single
                    // component: a wavefront tracker for 3-D SWE must be
                    // direction-agnostic. Reading only `uz` blinds the tracker
                    // to pushes whose shear displacement is transverse to z
                    // (in the phantom suite, the ±X/±Y pushes are invisible in
                    // `uz` along their own axis, so the arrival detector locks
                    // onto unrelated events and the TOF estimate is garbage).
                    for (value, f) in series.iter_mut().zip(history) {
                        let ux = f.ux[[i, j, k]];
                        let uy = f.uy[[i, j, k]];
                        let uz = f.uz[[i, j, k]];
                        *value = (ux * ux + uy * uy + uz * uz).sqrt();
                    }
                    // High-pass the series: subtract a moving average
                    // whose span covers the push duration. The ARF push leaves a
                    // quasi-static displacement that appears at every voxel in
                    // the push window regardless of distance (its 1/d² tail
                    // exceeds the 1/d shear wavefront out to several push radii),
                    // so without removal the detector locks onto the static
                    // rise and reports a distance-independent early arrival.
                    if series.len() > 8 {
                        let window = (series.len() / 8).max(3);
                        let mut acc = 0.0_f64;
                        for (i, &v) in series.iter().enumerate() {
                            acc += v;
                            if i >= window {
                                acc -= series[i - window];
                            }
                            smoothed[i] = acc / (i + 1).min(window) as f64;
                        }
                        for (v, &m) in series.iter_mut().zip(smoothed.iter()) {
                            *v -= m;
                        }
                    }
                    // Detect on the time derivative of the (high-passed)
                    // magnitude. A shear wavefront is a sharp step — its
                    // derivative is a strong spike — while the slow equilibrium
                    // drift and the ring-down are comparatively flat or
                    // oscillatory, so the derivative sharpens the wavefront
                    // contrast that the matched filter needs.
                    diff_series[0] = 0.0;
                    for idx in 1..series.len() {
                        diff_series[idx] = series[idx] - series[idx - 1];
                    }
                    match &self.volumetric_config.arrival_detection {
                        ArrivalDetection::EnergyThreshold { threshold } => {
                            let thr = *threshold;
                            if thr > 0.0 {
                                let mut found = false;
                                for (idx, &s) in series.iter().enumerate() {
                                    let a = s.abs();
                                    if a >= thr {
                                        arrival_times[[i, j, k]] = history[idx].time;
                                        amplitudes[[i, j, k]] = a;
                                        tracking_quality[[i, j, k]] = (a / (thr + 1e-30)).min(1.0);
                                        found = true;
                                        break;
                                    }
                                }
                                if !found {
                                    let (best_idx, best_amp) = series
                                        .iter()
                                        .enumerate()
                                        .map(|(idx, &s)| (idx, s.abs()))
                                        .fold(
                                            (0usize, 0.0_f64),
                                            |a, b| if b.1 > a.1 { b } else { a },
                                        );
                                    if best_amp > 0.0 {
                                        arrival_times[[i, j, k]] = history[best_idx].time;
                                        amplitudes[[i, j, k]] = best_amp;
                                        tracking_quality[[i, j, k]] =
                                            (best_amp / (thr + 1e-30)).min(1.0);
                                    }
                                }
                            } else {
                                let (best_idx, best_amp) = series
                                    .iter()
                                    .enumerate()
                                    .map(|(idx, &s)| (idx, s.abs()))
                                    .fold((0usize, 0.0_f64), |a, b| if b.1 > a.1 { b } else { a });
                                if best_amp > 0.0 {
                                    arrival_times[[i, j, k]] = history[best_idx].time;
                                    amplitudes[[i, j, k]] = best_amp;
                                    tracking_quality[[i, j, k]] = 1.0;
                                }
                            }
                        }
                        ArrivalDetection::MatchedFilter { template, min_corr } => {
                            let l = template.len();
                            if l == 0 || l > diff_series.len() {
                                continue;
                            }
                            let Some((start, chosen)) = matched_filter_window(
                                &series,
                                &diff_series,
                                template,
                                *min_corr,
                                &mut window_metrics,
                            ) else {
                                continue;
                            };
                            let center = start + (l / 2);
                            let idx = center.min(history.len() - 1);
                            arrival_times[[i, j, k]] = history[idx].time;
                            amplitudes[[i, j, k]] = chosen.amplitude;
                            tracking_quality[[i, j, k]] = chosen.quality;
                        }
                    }
                }
            }
        }
        WaveFrontTracker {
            arrival_times,
            amplitudes,
            tracking_quality,
        }
    }

    fn is_tracking_voxel(&self, i: usize, j: usize, k: usize) -> bool {
        let (nx, ny, nz) = self.grid.dimensions();
        if i >= nx || j >= ny || k >= nz {
            return false;
        }
        let [dx, dy, dz] = self.volumetric_config.tracking_decimation;
        i % dx.max(1) == 0
            && j % dy.max(1) == 0
            && k % dz.max(1) == 0
            // The absorbing layer is artificial, so an arrival there is
            // attenuation of an already-absorbed wave rather than a
            // measurable shear wavefront.
            && !self.pml.is_in_pml(i, j, k)
    }
}

fn matched_filter_window(
    series: &[f64],
    derivative: &[f64],
    template: &[f64],
    min_correlation: f64,
    metrics: &mut [WindowMetric],
) -> Option<(usize, WindowMetric)> {
    if template.is_empty() || template.len() > derivative.len() || series.len() != derivative.len()
    {
        return None;
    }
    let template_norm = template
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if template_norm <= 0.0 {
        return None;
    }
    let window_count = derivative.len() - template.len() + 1;
    if metrics.len() < window_count {
        return None;
    }

    let mut best_correlation: Option<f64> = None;
    for (start, metric) in metrics.iter_mut().take(window_count).enumerate() {
        let mut dot = 0.0_f64;
        let mut signal_energy = 0.0_f64;
        let mut amplitude = 0.0_f64;
        for offset in 0..template.len() {
            let signal = derivative[start + offset];
            dot += template[offset] * signal;
            signal_energy += signal * signal;
            amplitude = amplitude.max(series[start + offset].abs());
        }
        let correlation = dot.abs();
        let denominator = template_norm.mul_add(signal_energy.sqrt(), 1.0e-30);
        *metric = WindowMetric {
            correlation,
            quality: (correlation / denominator).min(1.0),
            amplitude,
        };
        if meets_correlation_floor(correlation, min_correlation) {
            best_correlation =
                Some(best_correlation.map_or(correlation, |best| best.max(correlation)));
        }
    }

    let best_correlation = best_correlation?;
    metrics
        .iter()
        .take(window_count)
        .copied()
        .enumerate()
        .find(|(_, metric)| {
            meets_correlation_floor(metric.correlation, min_correlation)
                && metric.correlation >= 0.5 * best_correlation
        })
}

fn meets_correlation_floor(correlation: f64, minimum: f64) -> bool {
    !matches!(
        correlation.partial_cmp(&minimum),
        Some(std::cmp::Ordering::Less)
    )
}

#[cfg(test)]
mod tests;
