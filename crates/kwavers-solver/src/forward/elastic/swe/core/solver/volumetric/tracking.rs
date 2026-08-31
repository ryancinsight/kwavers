//! Wavefront sample retention and arrival detection.

use super::super::definition::ElasticWaveSolver;
use super::SnapshotSchedule;
use crate::forward::elastic::swe::{ArrivalDetection, ElasticWaveField, WaveFrontTracker};
use kwavers_core::error::{KwaversResult, NumericalError, SystemError};
use leto::Array3;
use std::alloc::Layout;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct WindowMetric {
    correlation: f64,
    pub(super) quality: f64,
    pub(super) amplitude: f64,
}

pub(super) trait SnapshotRecorder {
    type Output;

    fn try_new(solver: &ElasticWaveSolver, schedule: SnapshotSchedule) -> KwaversResult<Self>
    where
        Self: Sized;

    fn record(&mut self, solver: &ElasticWaveSolver, field: &ElasticWaveField);

    fn finish(self, solver: &ElasticWaveSolver) -> Self::Output;
}

pub(super) struct FullFieldRecorder {
    history: Vec<ElasticWaveField>,
}

impl SnapshotRecorder for FullFieldRecorder {
    type Output = (Vec<ElasticWaveField>, WaveFrontTracker);

    fn try_new(_solver: &ElasticWaveSolver, schedule: SnapshotSchedule) -> KwaversResult<Self> {
        let mut history = Vec::new();
        try_reserve_exact(
            &mut history,
            schedule.capacity,
            "volumetric full-field history",
        )?;
        Ok(Self { history })
    }

    fn record(&mut self, _solver: &ElasticWaveSolver, field: &ElasticWaveField) {
        self.history.push(field.clone());
    }

    fn finish(self, solver: &ElasticWaveSolver) -> Self::Output {
        let tracker = solver.compute_wavefront_tracker(self.history.as_slice());
        (self.history, tracker)
    }
}

pub(super) struct MagnitudeRecorder {
    voxel_count: usize,
    maximum_sample_count: usize,
    times: Vec<f64>,
    // Snapshot-major storage keeps every propagation-time write sequential.
    magnitudes: Vec<f64>,
}

impl SnapshotRecorder for MagnitudeRecorder {
    type Output = WaveFrontTracker;

    fn try_new(solver: &ElasticWaveSolver, schedule: SnapshotSchedule) -> KwaversResult<Self> {
        let voxel_count = solver.tracking_voxel_count();
        let magnitude_count = voxel_count.checked_mul(schedule.capacity).ok_or_else(|| {
            NumericalError::InvalidOperation(
                "Volumetric tracker sample count exceeds addressable memory".to_owned(),
            )
        })?;
        let mut times = Vec::new();
        try_reserve_exact(
            &mut times,
            schedule.capacity,
            "volumetric tracker sample times",
        )?;
        let mut magnitudes = Vec::new();
        try_reserve_exact(
            &mut magnitudes,
            magnitude_count,
            "volumetric tracker magnitudes",
        )?;
        Ok(Self {
            voxel_count,
            maximum_sample_count: schedule.capacity,
            times,
            magnitudes,
        })
    }

    fn record(&mut self, solver: &ElasticWaveSolver, field: &ElasticWaveField) {
        debug_assert!(self.times.len() < self.maximum_sample_count);
        let before = self.magnitudes.len();
        let (nx, ny, nz) = solver.grid.dimensions();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if solver.is_tracking_voxel(i, j, k) {
                        self.magnitudes
                            .push(displacement_magnitude(field, [i, j, k]));
                    }
                }
            }
        }
        debug_assert_eq!(self.magnitudes.len() - before, self.voxel_count);
        self.times.push(field.time);
    }

    fn finish(self, solver: &ElasticWaveSolver) -> Self::Output {
        debug_assert!(self.times.len() <= self.maximum_sample_count);
        debug_assert_eq!(self.magnitudes.len(), self.voxel_count * self.times.len());
        solver.compute_wavefront_tracker(&self)
    }
}

pub(super) trait WavefrontSamples {
    fn sample_count(&self) -> usize;

    fn sample_time(&self, sample_index: usize) -> f64;

    fn fill_series(&self, voxel_index: usize, position: [usize; 3], series: &mut [f64]);
}

impl WavefrontSamples for [ElasticWaveField] {
    fn sample_count(&self) -> usize {
        self.len()
    }

    fn sample_time(&self, sample_index: usize) -> f64 {
        self[sample_index].time
    }

    fn fill_series(&self, _voxel_index: usize, position: [usize; 3], series: &mut [f64]) {
        for (value, field) in series.iter_mut().zip(self) {
            *value = displacement_magnitude(field, position);
        }
    }
}

impl WavefrontSamples for MagnitudeRecorder {
    fn sample_count(&self) -> usize {
        self.times.len()
    }

    fn sample_time(&self, sample_index: usize) -> f64 {
        self.times[sample_index]
    }

    fn fill_series(&self, voxel_index: usize, _position: [usize; 3], series: &mut [f64]) {
        if self.voxel_count == 0 {
            return;
        }
        for (sample, value) in self.magnitudes.chunks_exact(self.voxel_count).zip(series) {
            *value = sample[voxel_index];
        }
    }
}

impl ElasticWaveSolver {
    pub(super) fn compute_wavefront_tracker<S>(&self, samples: &S) -> WaveFrontTracker
    where
        S: WavefrontSamples + ?Sized,
    {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut arrival_times = Array3::<f64>::from_elem([nx, ny, nz], f64::NAN);
        let mut amplitudes = Array3::<f64>::zeros((nx, ny, nz));
        let mut tracking_quality = Array3::<f64>::zeros((nx, ny, nz));

        let sample_count = samples.sample_count();
        if sample_count < 2 {
            return WaveFrontTracker {
                arrival_times,
                amplitudes,
                tracking_quality,
            };
        }

        let mut series = vec![0.0_f64; sample_count];
        let mut smoothed = vec![0.0_f64; sample_count];
        let mut diff_series = vec![0.0_f64; sample_count];
        let mut window_metrics = vec![WindowMetric::default(); sample_count];
        let mut voxel_index = 0usize;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if !self.is_tracking_voxel(i, j, k) {
                        continue;
                    }
                    samples.fill_series(voxel_index, [i, j, k], &mut series);
                    voxel_index += 1;
                    // The ARF push leaves a quasi-static displacement whose
                    // tail can dominate the traveling shear wave. Subtract a
                    // moving average spanning the push before detection.
                    if series.len() > 8 {
                        let window = (series.len() / 8).max(3);
                        let mut acc = 0.0_f64;
                        for (index, &value) in series.iter().enumerate() {
                            acc += value;
                            if index >= window {
                                acc -= series[index - window];
                            }
                            smoothed[index] = acc / (index + 1).min(window) as f64;
                        }
                        for (value, &mean) in series.iter_mut().zip(&smoothed) {
                            *value -= mean;
                        }
                    }
                    // A traveling wavefront is a sharper event than the
                    // residual displacement, so arrival matching uses the
                    // high-passed magnitude's time derivative.
                    diff_series[0] = 0.0;
                    for index in 1..series.len() {
                        diff_series[index] = series[index] - series[index - 1];
                    }
                    match &self.volumetric_config.arrival_detection {
                        ArrivalDetection::EnergyThreshold { threshold } => {
                            let threshold = *threshold;
                            if threshold > 0.0 {
                                let mut found = false;
                                for (index, &sample) in series.iter().enumerate() {
                                    let amplitude = sample.abs();
                                    if amplitude >= threshold {
                                        arrival_times[[i, j, k]] = samples.sample_time(index);
                                        amplitudes[[i, j, k]] = amplitude;
                                        tracking_quality[[i, j, k]] =
                                            (amplitude / (threshold + 1e-30)).min(1.0);
                                        found = true;
                                        break;
                                    }
                                }
                                if !found {
                                    let (best_index, best_amplitude) = series
                                        .iter()
                                        .enumerate()
                                        .map(|(index, &sample)| (index, sample.abs()))
                                        .fold(
                                            (0usize, 0.0_f64),
                                            |a, b| {
                                                if b.1 > a.1 {
                                                    b
                                                } else {
                                                    a
                                                }
                                            },
                                        );
                                    if best_amplitude > 0.0 {
                                        arrival_times[[i, j, k]] = samples.sample_time(best_index);
                                        amplitudes[[i, j, k]] = best_amplitude;
                                        tracking_quality[[i, j, k]] =
                                            (best_amplitude / (threshold + 1e-30)).min(1.0);
                                    }
                                }
                            } else {
                                let (best_index, best_amplitude) = series
                                    .iter()
                                    .enumerate()
                                    .map(|(index, &sample)| (index, sample.abs()))
                                    .fold((0usize, 0.0_f64), |a, b| if b.1 > a.1 { b } else { a });
                                if best_amplitude > 0.0 {
                                    arrival_times[[i, j, k]] = samples.sample_time(best_index);
                                    amplitudes[[i, j, k]] = best_amplitude;
                                    tracking_quality[[i, j, k]] = 1.0;
                                }
                            }
                        }
                        ArrivalDetection::MatchedFilter { template, min_corr } => {
                            let template_len = template.len();
                            if template_len == 0 || template_len > diff_series.len() {
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
                            let center = start + (template_len / 2);
                            let sample_index = center.min(sample_count - 1);
                            arrival_times[[i, j, k]] = samples.sample_time(sample_index);
                            amplitudes[[i, j, k]] = chosen.amplitude;
                            tracking_quality[[i, j, k]] = chosen.quality;
                        }
                    }
                }
            }
        }
        debug_assert_eq!(voxel_index, self.tracking_voxel_count());
        WaveFrontTracker {
            arrival_times,
            amplitudes,
            tracking_quality,
        }
    }

    pub(super) fn is_tracking_voxel(&self, i: usize, j: usize, k: usize) -> bool {
        let (nx, ny, nz) = self.grid.dimensions();
        if i >= nx || j >= ny || k >= nz {
            return false;
        }
        let [dx, dy, dz] = self.volumetric_config.tracking_decimation;
        i % dx.max(1) == 0
            && j % dy.max(1) == 0
            && k % dz.max(1) == 0
            && !self.pml.is_in_pml(i, j, k)
    }

    fn tracking_voxel_count(&self) -> usize {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut count = 0usize;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    count += usize::from(self.is_tracking_voxel(i, j, k));
                }
            }
        }
        count
    }
}

fn displacement_magnitude(field: &ElasticWaveField, [i, j, k]: [usize; 3]) -> f64 {
    // Direction-invariant tracking keeps transverse pushes observable when a
    // single displacement component is zero along the propagation axis.
    let ux = field.ux[[i, j, k]];
    let uy = field.uy[[i, j, k]];
    let uz = field.uz[[i, j, k]];
    (ux * ux + uy * uy + uz * uz).sqrt()
}

pub(super) fn matched_filter_window(
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

fn try_reserve_exact<T>(values: &mut Vec<T>, count: usize, label: &str) -> KwaversResult<()> {
    let layout = Layout::array::<T>(count).map_err(|_| {
        NumericalError::InvalidOperation(format!("{label} exceeds addressable memory"))
    })?;
    values
        .try_reserve_exact(count)
        .map_err(|error| SystemError::MemoryAllocation {
            requested_bytes: layout.size(),
            reason: format!("{label} reservation failed: {error}"),
        })?;
    Ok(())
}

#[cfg(test)]
pub(super) fn compact_history_bytes(voxel_count: usize, sample_count: usize) -> Option<usize> {
    let magnitude_count = voxel_count.checked_mul(sample_count)?;
    let magnitudes = Layout::array::<f64>(magnitude_count).ok()?.size();
    let times = Layout::array::<f64>(sample_count).ok()?.size();
    magnitudes.checked_add(times)
}
