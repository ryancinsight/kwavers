//! CEUS Simulation Orchestrator

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;
use ndarray::Array4;

// Domain imports
use crate::domain::imaging::ultrasound::ceus::MicrobubblePopulation;

// Physics imports
use crate::physics::acoustics::imaging::modalities::ceus::{
    CEUSReconstruction, ContrastImage, FlowKinetics, NonlinearScattering, PerfusionModel,
};

// Orchestrator
#[derive(Debug)]
pub struct ContrastEnhancedUltrasound {
    /// Microbubble population
    microbubbles: MicrobubblePopulation,
    /// Nonlinear scattering model
    scattering: NonlinearScattering,
    /// Perfusion kinetics model
    perfusion: PerfusionModel,
    /// Image reconstruction
    reconstruction: CEUSReconstruction,
    /// Computational grid
    grid: Grid,
}

impl ContrastEnhancedUltrasound {
    pub fn new(
        grid: &Grid,
        medium: &dyn Medium,
        bubble_concentration: f64,
        bubble_size: f64,
    ) -> KwaversResult<Self> {
        let microbubbles = MicrobubblePopulation::new(bubble_concentration, bubble_size)?;
        let scattering = NonlinearScattering::new()?;
        let perfusion = PerfusionModel::new(grid, medium)?;
        let reconstruction = CEUSReconstruction::new(grid)?;

        Ok(Self {
            microbubbles,
            scattering,
            perfusion,
            reconstruction,
            grid: grid.clone(),
        })
    }

    /// Get microbubble concentration
    pub fn get_concentration(&self) -> f64 {
        self.microbubbles.get_concentration()
    }

    pub fn simulate_imaging_sequence(
        &mut self,
        injection_rate: f64,
        total_time: f64,
        frame_rate: f64,
        acoustic_pressure: f64,
        frequency: f64,
    ) -> KwaversResult<Vec<ContrastImage>> {
        let n_frames = (total_time * frame_rate) as usize;
        let dt = 1.0 / frame_rate;
        let mut images = Vec::with_capacity(n_frames);
        let bolus_profile = self.create_bolus_profile(injection_rate, total_time, n_frames);

        for (frame, &current_concentration) in bolus_profile.iter().enumerate().take(n_frames) {
            let time = frame as f64 * dt;
            self.perfusion
                .update_concentration(current_concentration, dt)?;
            let scattered_signals =
                self.simulate_acoustic_response(acoustic_pressure, frequency, time)?;
            let contrast_image = self.reconstruction.process_frame(&scattered_signals)?;
            images.push(contrast_image);
        }
        Ok(images)
    }

    pub fn simulate_bolus_injection(&self, total_bubbles: f64) -> KwaversResult<Vec<f64>> {
        if !total_bubbles.is_finite() || total_bubbles <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "simulate_bolus_injection: total_bubbles must be finite and > 0".to_string(),
            ));
        }

        let frame_rate_hz = 10.0;
        let duration_s = 30.0;
        let n_frames = (frame_rate_hz * duration_s) as usize;
        let dt = 1.0 / frame_rate_hz;

        let alpha = 3.0;
        let beta = 1.5;
        let tau = 0.5;

        let mut unscaled = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let t = i as f64 * dt;
            let v = if t > 0.0 {
                (t / tau).powf(alpha) * (-(t - tau) / beta).exp()
            } else {
                0.0
            };
            unscaled.push(v.max(0.0));
        }

        let area = unscaled.iter().sum::<f64>() * dt;
        if !area.is_finite() || area <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "simulate_bolus_injection: invalid gamma-variate area".to_string(),
            ));
        }

        let scale = total_bubbles / area;
        Ok(unscaled.into_iter().map(|v| v * scale).collect())
    }

    pub fn simulate_contrast_signal(
        &self,
        injection_profile_bubbles_per_s: &[f64],
        total_time_s: f64,
    ) -> KwaversResult<Array4<f32>> {
        if injection_profile_bubbles_per_s.is_empty() {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "simulate_contrast_signal: injection_profile must be non-empty".to_string(),
            ));
        }
        if !total_time_s.is_finite() || total_time_s <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "simulate_contrast_signal: total_time_s must be finite and > 0".to_string(),
            ));
        }

        let n_frames = injection_profile_bubbles_per_s.len();
        let dt = total_time_s / n_frames as f64;

        let (nx, ny, nz) = self.grid.dimensions();
        let mut signal = Array4::<f32>::zeros((n_frames, nx, ny, nz));

        let baseline = 1.0_f32;
        for i in 0..nx {
            let fx = i as f64 / (nx.saturating_sub(1).max(1) as f64);
            for j in 0..ny {
                let fy = j as f64 / (ny.saturating_sub(1).max(1) as f64);
                for k in 0..nz {
                    let fz = k as f64 / (nz.saturating_sub(1).max(1) as f64);
                    let mean_transit_time_s = 6.0 + 10.0 * (0.3 * fx + 0.5 * fy + 0.2 * fz);
                    let decay = (-dt / mean_transit_time_s).exp();

                    let local_gain = 1.0e-7_f64 * (1.0 + 0.2 * (fx - fy).abs());
                    let mut concentration = 0.0_f64;

                    for (t, &inj_rate) in injection_profile_bubbles_per_s.iter().enumerate() {
                        if !inj_rate.is_finite() || inj_rate < 0.0 {
                            return Err(crate::core::error::KwaversError::InvalidInput(
                                "simulate_contrast_signal: injection_profile contains invalid values"
                                    .to_string(),
                            ));
                        }

                        concentration = concentration * decay + inj_rate * dt;
                        let s = baseline as f64 + local_gain * concentration;
                        signal[(t, i, j, k)] = (s as f32).max(1.0e-6);
                    }
                }
            }
        }

        Ok(signal)
    }

    pub fn estimate_perfusion(
        &self,
        contrast_signal: &Array4<f32>,
        perfusion_model: &FlowKinetics,
    ) -> KwaversResult<Array3<f32>> {
        let (nt, nx, ny, nz) = contrast_signal.dim();
        if nt == 0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "estimate_perfusion: contrast_signal must have nt > 0".to_string(),
            ));
        }
        if perfusion_model.mean_transit_time <= 0.0
            || !perfusion_model.mean_transit_time.is_finite()
        {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "estimate_perfusion: mean_transit_time must be finite and > 0".to_string(),
            ));
        }

        let mut perfusion_map = Array3::<f32>::zeros((nx, ny, nz));
        let eps = 1.0e-6_f32;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let baseline = contrast_signal[(0, i, j, k)].max(eps);
                    let mut peak = baseline;
                    for t in 1..nt {
                        peak = peak.max(contrast_signal[(t, i, j, k)]);
                    }

                    let enhancement_ratio = (peak / baseline).max(eps);
                    perfusion_map[(i, j, k)] = enhancement_ratio;
                }
            }
        }

        Ok(perfusion_map)
    }

    fn simulate_acoustic_response(
        &self,
        acoustic_pressure: f64,
        frequency: f64,
        time: f64,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut scattered_signals = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let concentration = self.perfusion.concentration(i, j, k);
                    let local_pressure =
                        acoustic_pressure * (2.0 * std::f64::consts::PI * frequency * time).cos();
                    let scattering_response = self.scattering.compute_scattering(
                        &self.microbubbles,
                        concentration,
                        local_pressure,
                        frequency,
                    )?;
                    scattered_signals[[i, j, k]] = scattering_response;
                }
            }
        }
        Ok(scattered_signals)
    }

    fn create_bolus_profile(
        &self,
        injection_rate: f64,
        total_time: f64,
        n_frames: usize,
    ) -> Vec<f64> {
        let alpha = 3.0;
        let beta = 1.5;
        let tau = 0.5;
        let amplitude = injection_rate * 1000.0;
        let dt = total_time / n_frames as f64;

        (0..n_frames)
            .map(|i| {
                let t = i as f64 * dt;
                if t > 0.0 {
                    amplitude * (t / tau).powf(alpha) * (-(t - tau) / beta).exp()
                } else {
                    0.0
                }
            })
            .collect()
    }
}

// Implement domain CEUS orchestration interface
impl crate::domain::imaging::CEUSOrchestrator for ContrastEnhancedUltrasound {
    fn update(&mut self, pressure_field: &Array3<f64>, _time: f64) -> KwaversResult<Array3<f64>> {
        // Simplified: use max pressure as representative
        let max_pressure = pressure_field
            .iter()
            .max_by(|a, b| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(1.0);

        let frequency = 1.0e6; // Default 1 MHz
        self.simulate_acoustic_response(max_pressure, frequency, _time)
    }

    fn get_perfusion_data(&self) -> KwaversResult<Array3<f64>> {
        // Return perfusion field as 3D concentration map
        let (nx, ny, nz) = self.grid.dimensions();
        let mut perfusion_map = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    perfusion_map[[i, j, k]] = self.perfusion.concentration(i, j, k);
                }
            }
        }

        Ok(perfusion_map)
    }

    fn get_concentration_map(&self) -> KwaversResult<Array3<f64>> {
        // Return microbubble concentration map
        let (nx, ny, nz) = self.grid.dimensions();
        let concentration = self.get_concentration();
        Ok(Array3::from_elem((nx, ny, nz), concentration))
    }

    fn name(&self) -> &str {
        "SimulationCEUSOrchestrator"
    }
}
