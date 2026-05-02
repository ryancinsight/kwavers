use super::super::boundary::{PMLBoundary, PMLConfig};
use super::super::integration::TimeIntegrator;
use super::super::types::{
    ArrivalDetection, ElasticBodyForceConfig, ElasticWaveConfig, ElasticWaveField,
    VolumetricQualityMetrics, VolumetricSource, VolumetricWaveConfig, WaveFrontTracker,
};
use crate::core::error::{KwaversResult, NumericalError};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::sensor::recorder::simple::SensorRecorder;
use ndarray::{Array2, Array3};

/// 3D Elastic Wave Solver.
#[derive(Debug)]
pub struct ElasticWaveSolver {
    grid: Grid,
    density: Array3<f64>,
    lambda: Array3<f64>,
    mu: Array3<f64>,
    pml: PMLBoundary,
    config: ElasticWaveConfig,
    volumetric_config: VolumetricWaveConfig,
    pub(crate) sensor_recorder: SensorRecorder,
}

impl ElasticWaveSolver {
    pub fn new(grid: &Grid, medium: &dyn Medium, config: ElasticWaveConfig) -> KwaversResult<Self> {
        let (nx, ny, nz) = grid.dimensions();
        let density = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| medium.density(i, j, k));
        let lambda = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            let (x, y, z) = grid.indices_to_coordinates(i, j, k);
            medium.lame_lambda(x, y, z, grid)
        });
        let mu = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            let (x, y, z) = grid.indices_to_coordinates(i, j, k);
            medium.lame_mu(x, y, z, grid)
        });
        let pml_config = PMLConfig {
            thickness: config.pml_thickness,
            sigma_max: 100.0,
            profile_order: 2,
            reflection_target: 1e-4,
        };
        let pml = PMLBoundary::new(grid, pml_config);
        let shape = (nx, ny, nz);
        let sensor_recorder = SensorRecorder::new(config.sensor_mask.as_ref(), shape, 0)?;
        Ok(Self {
            grid: grid.clone(),
            density,
            lambda,
            mu,
            pml,
            config,
            volumetric_config: VolumetricWaveConfig::default(),
            sensor_recorder,
        })
    }

    pub fn set_volumetric_config(&mut self, volumetric_config: VolumetricWaveConfig) {
        self.volumetric_config = volumetric_config;
    }

    pub fn propagate(
        &mut self,
        initial_field: &ElasticWaveField,
        duration: f64,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<ElasticWaveField> {
        let mut current_field = initial_field.clone();
        let integrator = TimeIntegrator::new(
            &self.grid,
            &self.lambda,
            &self.mu,
            &self.density,
            self.pml.attenuation_field(),
        );
        let dt = if self.config.time_step > 0.0 {
            self.config.time_step
        } else {
            integrator.calculate_stable_timestep(self.config.cfl_factor)
        };
        if dt <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Calculated time step is non-positive".to_string(),
            )
            .into());
        }
        let steps = (duration / dt).ceil() as usize;
        let save_every = self.config.save_every.max(1);
        let recorded_steps = steps.div_ceil(save_every);
        let (nx, ny, nz) = self.grid.dimensions();
        self.sensor_recorder = SensorRecorder::new(
            self.config.sensor_mask.as_ref(),
            (nx, ny, nz),
            recorded_steps,
        )?;
        for step in 0..steps {
            integrator.step(&mut current_field, dt, body_force)?;
            current_field.time += dt;
            if step % save_every == 0 {
                self.sensor_recorder.record_step(&current_field.uz)?;
            }
        }
        Ok(current_field)
    }

    pub fn extract_recorded_data(&self) -> Option<Array2<f64>> {
        self.sensor_recorder.extract_pressure_data()
    }

    pub fn propagate_waves(
        &self,
        initial_displacement: &Array3<f64>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let (nx, ny, nz) = self.grid.dimensions();
        if initial_displacement.dim() != (nx, ny, nz) {
            return Err(NumericalError::InvalidOperation(
                "Initial displacement shape does not match grid".to_string(),
            )
            .into());
        }
        let mut initial_field = ElasticWaveField::new(nx, ny, nz);
        initial_field.uz.assign(initial_displacement);
        self.propagate_history(&initial_field, self.config.simulation_time, None)
    }

    pub fn propagate_volumetric_waves_with_body_forces(
        &self,
        body_forces: &[ElasticBodyForceConfig],
        push_times: &[f64],
        _sources: &[VolumetricSource],
    ) -> KwaversResult<(Vec<ElasticWaveField>, WaveFrontTracker)> {
        if body_forces.len() != push_times.len() {
            return Err(NumericalError::InvalidOperation(
                "body_forces and push_times must have the same length".to_string(),
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
        let integrator = TimeIntegrator::new(
            &self.grid,
            &self.lambda,
            &self.mu,
            &self.density,
            self.pml.attenuation_field(),
        );
        let dt = if self.config.time_step > 0.0 {
            self.config.time_step
        } else {
            integrator.calculate_stable_timestep(self.config.cfl_factor)
        };
        if dt <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Calculated time step is non-positive".to_string(),
            )
            .into());
        }
        let duration_s = self.volumetric_config.duration_s;
        if !duration_s.is_finite() || duration_s <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Volumetric duration must be positive".to_string(),
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
        let mut history = Vec::with_capacity(snapshot_cap);
        history.push(current_field.clone());
        for step_idx in 0..steps {
            integrator.step_with_body_forces(&mut current_field, dt, &shifted_forces)?;
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

    pub fn propagate_volumetric_waves_with_sources(
        &self,
        initial_displacements: &[Array3<f64>],
        push_times: &[f64],
        sources: &[VolumetricSource],
    ) -> KwaversResult<(Vec<ElasticWaveField>, WaveFrontTracker)> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut initial_field = ElasticWaveField::new(nx, ny, nz);
        for disp in initial_displacements {
            if disp.dim() != (nx, ny, nz) {
                return Err(NumericalError::InvalidOperation(
                    "Initial displacement shape does not match grid".to_string(),
                )
                .into());
            }
            initial_field.uz.zip_mut_with(disp, |a, &b| *a += b);
        }
        let integrator = TimeIntegrator::new(
            &self.grid,
            &self.lambda,
            &self.mu,
            &self.density,
            self.pml.attenuation_field(),
        );
        let dt = if self.config.time_step > 0.0 {
            self.config.time_step
        } else {
            integrator.calculate_stable_timestep(self.config.cfl_factor)
        };
        if dt <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Calculated time step is non-positive".to_string(),
            )
            .into());
        }
        let duration_s = self.volumetric_config.duration_s;
        if !duration_s.is_finite() || duration_s <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Volumetric duration must be positive".to_string(),
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
        let mut current_field = initial_field;
        let mut history = Vec::with_capacity(snapshot_cap);
        history.push(current_field.clone());
        for step_idx in 0..steps {
            integrator.step(&mut current_field, dt, None)?;
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
                "push_times and sources must have the same length when provided".to_string(),
            )
            .into());
        }
        for w in push_times.windows(2) {
            if w[1] < w[0] {
                return Err(NumericalError::InvalidOperation(
                    "push_times must be non-decreasing".to_string(),
                )
                .into());
            }
        }
        let _ = sources;
        let tracker = self.compute_wavefront_tracker(&history);
        Ok((history, tracker))
    }

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

    pub fn propagate_waves_with_body_force_only_override(
        &self,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let initial_field = ElasticWaveField::new(nx, ny, nz);
        self.propagate_history(&initial_field, self.config.simulation_time, body_force)
    }

    fn propagate_history(
        &self,
        initial_field: &ElasticWaveField,
        duration_s: f64,
        body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let mut current_field = initial_field.clone();
        let integrator = TimeIntegrator::new(
            &self.grid,
            &self.lambda,
            &self.mu,
            &self.density,
            self.pml.attenuation_field(),
        );
        let dt = if self.config.time_step > 0.0 {
            self.config.time_step
        } else {
            integrator.calculate_stable_timestep(self.config.cfl_factor)
        };
        if dt <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Calculated time step is non-positive".to_string(),
            )
            .into());
        }
        if !duration_s.is_finite() || duration_s <= 0.0 {
            return Err(NumericalError::InvalidOperation(
                "Simulation duration must be positive".to_string(),
            )
            .into());
        }
        let save_every = self.config.save_every.max(1);
        let steps = (duration_s / dt).ceil() as usize;
        let mut history = Vec::new();
        history.push(current_field.clone());
        for step_idx in 0..steps {
            integrator.step(&mut current_field, dt, body_force)?;
            current_field.time += dt;
            if (step_idx + 1) % save_every == 0 {
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
        Ok(history)
    }

    fn compute_wavefront_tracker(&self, history: &[ElasticWaveField]) -> WaveFrontTracker {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut arrival_times = Array3::<f64>::from_elem((nx, ny, nz), f64::NAN);
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
                    let series: Vec<f64> = history.iter().map(|f| f.uz[[i, j, k]]).collect();
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
                            if l == 0 || l > series.len() {
                                continue;
                            }
                            let template_norm = template.iter().map(|x| x * x).sum::<f64>().sqrt();
                            if template_norm <= 0.0 {
                                continue;
                            }
                            let mut best_idx = None;
                            let mut best_corr = 0.0_f64;
                            let mut best_quality = 0.0_f64;
                            let mut best_amp = 0.0_f64;
                            for start in 0..=(series.len() - l) {
                                let mut dot = 0.0_f64;
                                let mut sig_energy = 0.0_f64;
                                let mut amp = 0.0_f64;
                                for n in 0..l {
                                    let s = series[start + n];
                                    let t = template[n];
                                    dot += t * s;
                                    sig_energy += s * s;
                                    amp = amp.max(s.abs());
                                }
                                let corr = dot.abs();
                                if corr < *min_corr {
                                    continue;
                                }
                                let denom = template_norm * sig_energy.sqrt() + 1e-30;
                                let quality = (corr / denom).min(1.0);
                                if best_idx.is_none() || corr > best_corr {
                                    best_idx = Some(start);
                                    best_corr = corr;
                                    best_quality = quality;
                                    best_amp = amp;
                                }
                            }
                            if let Some(start) = best_idx {
                                let center = start + (l / 2);
                                let idx = center.min(history.len() - 1);
                                arrival_times[[i, j, k]] = history[idx].time;
                                amplitudes[[i, j, k]] = best_amp;
                                tracking_quality[[i, j, k]] = best_quality;
                            }
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
