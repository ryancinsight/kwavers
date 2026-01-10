//! Perfusion Modeling for Contrast-Enhanced Ultrasound
//!
//! Implements blood flow kinetics, tissue uptake, and perfusion quantification
//! for CEUS imaging applications.

use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;

/// Perfusion model for contrast agent kinetics
#[derive(Debug)]
pub struct PerfusionModel {
    /// Concentration field (bubbles/m³)
    concentration: Array3<f64>,
    /// Blood flow velocity field (m/s)
    velocity: Array3<(f64, f64, f64)>,
    /// Tissue permeability
    permeability: f64,
}

impl PerfusionModel {
    /// Create new perfusion model
    pub fn new(grid: &Grid, _medium: &dyn Medium) -> KwaversResult<Self> {
        let (nx, ny, nz) = grid.dimensions();

        Ok(Self {
            concentration: Array3::zeros((nx, ny, nz)),
            velocity: Array3::from_elem((nx, ny, nz), (0.0, 0.0, 0.0)),
            permeability: 1e-6, // m/s (typical capillary permeability)
        })
    }

    /// Update concentration field over time
    pub fn update_concentration(
        &mut self,
        inflow_concentration: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        // Simplified advection-diffusion model
        let (nx, ny, nz) = self.concentration.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Advection term in contrast agent transport
                    let advective_flux = self.velocity[[i, j, k]].0 * dt; // x-direction only

                    // Diffusion and uptake
                    let diffusion = self.permeability * dt;
                    let uptake = -self.concentration[[i, j, k]] * 0.1 * dt; // Decay term

                    // Update concentration
                    self.concentration[[i, j, k]] += advective_flux + diffusion + uptake;

                    // Add inflow
                    if i == 0 {
                        // Inlet boundary
                        self.concentration[[i, j, k]] += inflow_concentration * dt;
                    }

                    // Ensure non-negative
                    self.concentration[[i, j, k]] = self.concentration[[i, j, k]].max(0.0);
                }
            }
        }

        Ok(())
    }

    /// Get concentration at specific location
    #[must_use]
    pub fn concentration(&self, i: usize, j: usize, k: usize) -> f64 {
        self.concentration[[i, j, k]]
    }

    /// Get concentration field reference
    #[must_use]
    pub fn concentration_field(&self) -> &Array3<f64> {
        &self.concentration
    }

    /// Create a default gamma variate flow kinetics model suitable for CEUS
    /// bolus injection simulation.
    pub fn gamma_variate_model() -> FlowKinetics {
        // Default frame rate and duration for model synthesis
        let frame_rate = 10.0; // Hz
        let duration = 30.0; // s
        let n_frames = (frame_rate * duration) as usize;
        let dt = 1.0 / frame_rate;

        // Gamma variate parameters
        let alpha = 3.0;
        let beta = 1.5; // s
        let tau = 0.5; // s

        let mut arterial_input = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let t = i as f64 * dt;
            let val = if t > 0.0 {
                (t / tau).powf(alpha) * (-(t - tau) / beta).exp()
            } else {
                0.0
            };
            arterial_input.push(val.max(0.0));
        }

        // Residue function approximated as exponential decay with MTT
        let mean_transit_time = 10.0; // s
        let mut residue_function = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let t = i as f64 * dt;
            residue_function.push((-(t / mean_transit_time)).exp());
        }

        FlowKinetics {
            arterial_input,
            residue_function,
            mean_transit_time,
        }
    }
}

/// Flow kinetics model
#[derive(Debug)]
pub struct FlowKinetics {
    /// Arterial input function
    pub arterial_input: Vec<f64>,
    /// Tissue residue function
    pub residue_function: Vec<f64>,
    /// Mean transit time (s)
    pub mean_transit_time: f64,
}

impl FlowKinetics {
    /// Create new flow kinetics model
    pub fn new() -> Self {
        Self {
            arterial_input: Vec::new(),
            residue_function: Vec::new(),
            mean_transit_time: 10.0, // 10 seconds typical
        }
    }

    /// Compute perfusion parameters from time-intensity curve
    pub fn analyze_tic(&self, tic: &[f64], frame_rate: f64) -> PerfusionParameters {
        if tic.is_empty() {
            return PerfusionParameters::default();
        }

        // Find peak
        let peak_idx = tic
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let peak_intensity = tic[peak_idx];
        let time_to_peak = peak_idx as f64 / frame_rate;

        // Compute area under curve
        let mut auc = 0.0;
        let dt = 1.0 / frame_rate;
        for i in 0..tic.len().saturating_sub(1) {
            auc += (tic[i] + tic[i + 1]) * dt / 2.0;
        }

        // Wash-in rate (initial slope)
        let wash_in_rate = if peak_idx > 1 {
            (tic[1] - tic[0]) / dt
        } else {
            0.0
        };

        // Wash-out rate (final slope)
        let wash_out_rate = if tic.len() > peak_idx + 2 {
            (tic[tic.len() - 1] - tic[tic.len() - 2]) / dt
        } else {
            0.0
        };

        PerfusionParameters {
            peak_intensity,
            time_to_peak,
            area_under_curve: auc,
            wash_in_rate,
            wash_out_rate,
            mean_transit_time: self.mean_transit_time,
        }
    }
}

impl Default for FlowKinetics {
    fn default() -> Self {
        Self::new()
    }
}

/// Perfusion parameters from time-intensity curve analysis
#[derive(Debug, Clone)]
pub struct PerfusionParameters {
    /// Peak intensity (dB)
    pub peak_intensity: f64,
    /// Time to peak (s)
    pub time_to_peak: f64,
    /// Area under curve (dB·s)
    pub area_under_curve: f64,
    /// Wash-in rate (dB/s)
    pub wash_in_rate: f64,
    /// Wash-out rate (dB/s)
    pub wash_out_rate: f64,
    /// Mean transit time (s)
    pub mean_transit_time: f64,
}

impl Default for PerfusionParameters {
    fn default() -> Self {
        Self {
            peak_intensity: 0.0,
            time_to_peak: 0.0,
            area_under_curve: 0.0,
            wash_in_rate: 0.0,
            wash_out_rate: 0.0,
            mean_transit_time: 10.0,
        }
    }
}

/// Tissue uptake model
#[derive(Debug)]
pub struct TissueUptake {
    /// Uptake rate constant (1/s)
    pub uptake_rate: f64,
    /// Clearance rate constant (1/s)
    pub clearance_rate: f64,
    /// Partition coefficient
    pub partition_coefficient: f64,
}

impl TissueUptake {
    /// Create new tissue uptake model
    pub fn new() -> Self {
        Self {
            uptake_rate: 0.1,           // 0.1 /s
            clearance_rate: 0.05,       // 0.05 /s
            partition_coefficient: 0.2, // Dimensionless
        }
    }

    /// Compute tissue concentration over time
    pub fn tissue_concentration(&self, plasma_concentration: f64, time: f64) -> f64 {
        // Two-compartment model
        let k1 = self.uptake_rate;
        let k2 = self.clearance_rate;
        let v = self.partition_coefficient;

        // Analytical solution for tissue concentration
        if time <= 0.0 {
            0.0
        } else {
            v * k1 * plasma_concentration * (1.0 - (-k2 * time).exp()) / k2
        }
    }
}

impl Default for TissueUptake {
    fn default() -> Self {
        Self::new()
    }
}
