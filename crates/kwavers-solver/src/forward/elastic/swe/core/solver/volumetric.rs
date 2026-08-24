//! Volumetric propagation and wavefront tracking for `ElasticWaveSolver`.

use super::super::super::integration::TimeIntegrator;
use super::super::super::scratch::ElasticStepScratch;
use super::super::super::types::{
    ArrivalDetection, ElasticBodyForceConfig, ElasticWaveField, VolumetricQualityMetrics,
    VolumetricSource, WaveFrontTracker,
};
use super::definition::ElasticWaveSolver;
use kwavers_core::error::{KwaversResult, NumericalError};
use leto::Array3;

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
        let mut stride = (steps / self.volumetric_config.max_snapshots.max(2)).max(1);
        let min_snapshots = 10usize;
        if steps / stride + 1 < min_snapshots {
            stride = (steps / (min_snapshots - 1)).max(1);
        }
        let snapshot_cap = steps / stride + 2;
        let mut scratch = ElasticStepScratch::new(nx, ny, nz);
        let mut history = Vec::with_capacity(snapshot_cap);
        history.push(current_field.clone());
        for step_idx in 0..steps {
            integrator.step_with_body_forces(
                &mut current_field,
                dt,
                &shifted_forces,
                &mut scratch,
            )?;
            current_field.time += dt;
            if (step_idx + 1) % stride == 0 {
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
        let mut stride = (steps / self.volumetric_config.max_snapshots.max(2)).max(1);
        let min_snapshots = 10usize;
        if steps / stride + 1 < min_snapshots {
            stride = (steps / (min_snapshots - 1)).max(1);
        }
        let snapshot_cap = steps / stride + 2;
        let mut scratch = ElasticStepScratch::new(nx, ny, nz);
        let mut current_field = initial_field;
        let mut history = Vec::with_capacity(snapshot_cap);
        history.push(current_field.clone());
        for step_idx in 0..steps {
            integrator.step(&mut current_field, dt, None, &mut scratch)?;
            current_field.time += dt;
            if (step_idx + 1) % stride == 0 {
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

    #[must_use]
    pub fn calculate_volumetric_quality(
        &self,
        tracker: &WaveFrontTracker,
    ) -> VolumetricQualityMetrics {
        let total = tracker.arrival_times.len();
        let mut valid = 0usize;
        let mut quality_sum = 0.0;
        for (&t, &q) in tracker
            .arrival_times
            .iter()
            .zip(tracker.tracking_quality.iter())
        {
            if t.is_finite() && q > 0.0 {
                valid += 1;
                quality_sum += q;
            }
        }
        VolumetricQualityMetrics {
            coverage: if total == 0 {
                0.0
            } else {
                valid as f64 / total as f64
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

        let [dx, dy, dz] = self.volumetric_config.tracking_decimation;
        let (dx, dy, dz) = (dx.max(1), dy.max(1), dz.max(1));

        for k in 0..nz {
            if k % dz != 0 {
                continue;
            }
            for j in 0..ny {
                if j % dy != 0 {
                    continue;
                }
                for i in 0..nx {
                    if i % dx != 0 {
                        continue;
                    }
                    // Skip PML voxels: the absorbing layer is artificial, so an
                    // arrival there is attenuation of an already-absorbed wave,
                    // not a measurable shear wavefront. Without this, the
                    // detector locks onto numerical noise in the PML ring and
                    // the TOF reconstruction reports garbage speeds there.
                    if self.pml.is_in_pml(i, j, k) {
                        continue;
                    }
                    // Track the displacement magnitude, not a single
                    // component: a wavefront tracker for 3-D SWE must be
                    // direction-agnostic. Reading only `uz` blinds the tracker
                    // to pushes whose shear displacement is transverse to z
                    // (in the phantom suite, the ±X/±Y pushes are invisible in
                    // `uz` along their own axis, so the arrival detector locks
                    // onto unrelated events and the TOF estimate is garbage).
                    let mut series: Vec<f64> = history
                        .iter()
                        .map(|f| {
                            let ux = f.ux[[i, j, k]];
                            let uy = f.uy[[i, j, k]];
                            let uz = f.uz[[i, j, k]];
                            (ux * ux + uy * uy + uz * uz).sqrt()
                        })
                        .collect();
                    // High-pass the series: subtract a centered moving average
                    // whose span covers the push duration. The ARF push leaves a
                    // quasi-static displacement that appears at every voxel in
                    // the push window regardless of distance (its 1/d² tail
                    // exceeds the 1/d shear wavefront out to several push radii),
                    // so without removal the detector locks onto the static
                    // rise and reports a distance-independent early arrival.
                    if series.len() > 8 {
                        let window = (series.len() / 8).max(3);
                        let mut smoothed = vec![0.0_f64; series.len()];
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
                    let diff_series: Vec<f64> = series
                        .iter()
                        .enumerate()
                        .map(|(idx, &v)| {
                            if idx == 0 {
                                0.0
                            } else {
                                v - series[idx - 1]
                            }
                        })
                        .collect();
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
                            let template_norm = template.iter().map(|x| x * x).sum::<f64>().sqrt();
                            if template_norm <= 0.0 {
                                continue;
                            }
                            // Evaluate every window position in time order.
                            // Correlation uses the derivative series; the
                            // recorded amplitude stays the displacement
                            // magnitude at that window.
                            let mut windows: Vec<(f64, f64, f64)> = Vec::new();
                            for start in 0..=(diff_series.len() - l) {
                                let mut dot = 0.0_f64;
                                let mut sig_energy = 0.0_f64;
                                let mut amp = 0.0_f64;
                                for n in 0..l {
                                    let s = diff_series[start + n];
                                    let t = template[n];
                                    dot += t * s;
                                    sig_energy += s * s;
                                    amp = amp.max(series[start + n].abs());
                                }
                                let corr = dot.abs();
                                if corr < *min_corr {
                                    continue;
                                }
                                let denom = template_norm.mul_add(sig_energy.sqrt(), 1e-30);
                                let quality = (corr / denom).min(1.0);
                                windows.push((corr, quality, amp));
                            }
                            if windows.is_empty() {
                                continue;
                            }
                            let best = windows
                                .iter()
                                .max_by(|a, b| a.0.total_cmp(&b.0))
                                .copied()
                                .unwrap();
                            // First-arrival policy: take the earliest window
                            // whose correlation is at least half the strongest.
                            // A plain global maximum can lock onto reflected or
                            // scattered waves arriving after the direct shear
                            // wavefront, biasing the TOF estimate low; the
                            // direct arrival is what SWE measures.
                            let chosen = match windows
                                .iter()
                                .enumerate()
                                .find(|(_, w)| w.0 >= 0.5 * best.0)
                            {
                                Some((start, &w)) => (start, w),
                                None => {
                                    let start = windows
                                        .iter()
                                        .position(|w| *w == best)
                                        .unwrap_or(0);
                                    (start, best)
                                }
                            };
                            let (start, chosen) = chosen;
                            let center = start + (l / 2);
                            let idx = center.min(history.len() - 1);
                            arrival_times[[i, j, k]] = history[idx].time;
                            amplitudes[[i, j, k]] = chosen.2;
                            tracking_quality[[i, j, k]] = chosen.1;
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
}
