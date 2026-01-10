//! CEUS domain definitions

use crate::domain::core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::Array3;

/// CEUS imaging parameters
#[derive(Debug, Clone)]
pub struct CEUSImagingParameters {
    /// Transmit frequency (Hz)
    pub frequency: f64,
    /// Mechanical index
    pub mechanical_index: f64,
    /// Frame rate (Hz)
    pub frame_rate: f64,
    /// Dynamic range (dB)
    pub dynamic_range: f64,
    /// Field of view (mm)
    pub fov: (f64, f64),
    /// Imaging depth (mm)
    pub depth: f64,
}

impl Default for CEUSImagingParameters {
    fn default() -> Self {
        Self {
            frequency: 3.0e6,      // 3 MHz
            mechanical_index: 0.1, // Low MI for CEUS
            frame_rate: 10.0,      // 10 fps
            dynamic_range: 60.0,   // 60 dB
            fov: (80.0, 60.0),     // 80x60 mm
            depth: 150.0,          // 150 mm
        }
    }
}

/// Individual microbubble properties
#[derive(Debug, Clone)]
pub struct Microbubble {
    /// Equilibrium radius (m)
    pub radius_eq: f64,
    /// Shell thickness (m)
    pub shell_thickness: f64,
    /// Shell elasticity (Pa)
    pub shell_elasticity: f64,
    /// Shell viscosity (Pa·s)
    pub shell_viscosity: f64,
    /// Gas polytropic index
    pub polytropic_index: f64,
    /// Surface tension (N/m)
    pub surface_tension: f64,
}

impl Microbubble {
    /// Create new microbubble with typical contrast agent properties
    pub fn new(radius: f64, shell_elasticity: f64, shell_viscosity: f64) -> Self {
        Self {
            radius_eq: radius * 1e-6,                 // Convert μm to m
            shell_thickness: radius * 1e-6 * 0.1,     // 10% of radius
            shell_elasticity: shell_elasticity * 1e3, // Convert kPa to Pa
            shell_viscosity,
            polytropic_index: 1.07, // Typical for encapsulated bubbles
            surface_tension: 0.072, // N/m for water-air interface
        }
    }

    /// Create SonoVue-like microbubble (typical clinical contrast agent)
    pub fn sono_vue() -> Self {
        Self::new(1.5, 1.0, 0.5) // 1.5 μm radius, 1 kPa elasticity, 0.5 Pa·s viscosity
    }

    /// Create Definity-like microbubble
    pub fn definit_y() -> Self {
        Self::new(2.0, 2.5, 1.0) // 2.0 μm radius, 2.5 kPa elasticity, 1.0 Pa·s viscosity
    }

    /// Compute natural resonance frequency (Hz)
    #[must_use]
    pub fn resonance_frequency(&self, _ambient_pressure: f64, _liquid_density: f64) -> f64 {
        let diameter_um = self.radius_eq * 2.0 * 1e6; // Convert to μm
        let base_freq_mhz = 6.0 / diameter_um; // MHz
        let shell_factor = 1.0 + (self.shell_elasticity / 1000.0) * 0.2; // 20% increase per kPa
        base_freq_mhz * shell_factor * 1e6 // Convert to Hz
    }

    /// Validate microbubble parameters
    pub fn validate(&self) -> KwaversResult<()> {
        if self.radius_eq <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "radius_eq".to_string(),
                value: self.radius_eq,
                reason: "must be positive".to_string(),
            }));
        }
        if self.shell_elasticity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_elasticity".to_string(),
                value: self.shell_elasticity,
                reason: "must be non-negative".to_string(),
            }));
        }
        if self.shell_viscosity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "shell_viscosity".to_string(),
                value: self.shell_viscosity,
                reason: "must be non-negative".to_string(),
            }));
        }
        Ok(())
    }

    /// Compute scattering cross-section at resonance
    #[must_use]
    pub fn scattering_cross_section(&self, frequency: f64) -> f64 {
        let _ka = 2.0 * std::f64::consts::PI * frequency * self.radius_eq / 343.0; // k*a
        let resonance_factor =
            1.0 / (1.0 - (frequency / self.resonance_frequency(101325.0, 1000.0)).powi(2)).powi(2);

        // Simplified scattering cross-section
        std::f64::consts::PI * self.radius_eq * self.radius_eq * resonance_factor
    }
}

/// Population of microbubbles with size distribution
#[derive(Debug, Clone)]
pub struct MicrobubblePopulation {
    /// Reference microbubble (mean properties)
    pub reference_bubble: Microbubble,
    /// Size distribution parameters (log-normal)
    pub size_distribution: SizeDistribution,
    /// Initial concentration (bubbles/m³)
    pub concentration: f64,
}

impl MicrobubblePopulation {
    /// Create new microbubble population
    pub fn new(concentration: f64, mean_diameter: f64) -> KwaversResult<Self> {
        let reference_bubble = Microbubble::new(mean_diameter / 2.0, 1.5, 0.8);

        // Typical log-normal distribution for contrast agents
        let size_distribution = SizeDistribution {
            mean_radius: reference_bubble.radius_eq,
            std_dev: reference_bubble.radius_eq * 0.3, // 30% coefficient of variation
        };

        Ok(Self {
            reference_bubble,
            size_distribution,
            concentration: concentration * 1e6, // Convert bubbles/mL to bubbles/m³
        })
    }

    /// Global concentration (simplified)
    pub fn get_concentration(&self) -> f64 {
        self.concentration
    }

    pub fn effective_scattering(
        &self,
        frequency: f64,
        ambient_pressure: f64,
        liquid_density: f64,
        sound_speed: f64,
        liquid_viscosity: f64,
    ) -> KwaversResult<f64> {
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "frequency".to_string(),
                value: frequency,
                reason: "must be positive and finite".to_string(),
            }));
        }
        if !ambient_pressure.is_finite() || ambient_pressure <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "ambient_pressure".to_string(),
                value: ambient_pressure,
                reason: "must be positive and finite".to_string(),
            }));
        }
        if !liquid_density.is_finite() || liquid_density <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "liquid_density".to_string(),
                value: liquid_density,
                reason: "must be positive and finite".to_string(),
            }));
        }
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "sound_speed".to_string(),
                value: sound_speed,
                reason: "must be positive and finite".to_string(),
            }));
        }
        if !liquid_viscosity.is_finite() || liquid_viscosity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "liquid_viscosity".to_string(),
                value: liquid_viscosity,
                reason: "must be non-negative and finite".to_string(),
            }));
        }

        let mean_radius = self.size_distribution.mean_radius;
        let std_dev = self.size_distribution.std_dev;
        if !mean_radius.is_finite() || mean_radius <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "mean_radius".to_string(),
                value: mean_radius,
                reason: "must be positive and finite".to_string(),
            }));
        }
        if !std_dev.is_finite() || std_dev < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "std_dev".to_string(),
                value: std_dev,
                reason: "must be non-negative and finite".to_string(),
            }));
        }

        let shell_elasticity = self.reference_bubble.shell_elasticity;
        let shell_viscosity = self.reference_bubble.shell_viscosity;

        let scattering_at_radius = |radius_eq: f64| -> KwaversResult<f64> {
            if !radius_eq.is_finite() || radius_eq <= 0.0 {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: "radius_eq".to_string(),
                    value: radius_eq,
                    reason: "must be positive and finite".to_string(),
                }));
            }

            let diameter_um = radius_eq * 2.0 * 1e6;
            let base_freq_mhz = 6.0 / diameter_um;
            let shell_factor = 1.0 + (shell_elasticity / 1000.0) * 0.2;
            let f0_hz = base_freq_mhz * shell_factor * 1e6;

            if !f0_hz.is_finite() || f0_hz <= 0.0 {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: "resonance_frequency".to_string(),
                    value: f0_hz,
                    reason: "must be positive and finite".to_string(),
                }));
            }

            let omega = 2.0 * std::f64::consts::PI * frequency;
            let omega0 = 2.0 * std::f64::consts::PI * f0_hz;

            let effective_viscosity = liquid_viscosity + shell_viscosity;
            let delta_visc = 4.0 * effective_viscosity / (liquid_density * radius_eq * radius_eq);
            let delta_rad = omega0 * omega0 * radius_eq / sound_speed;
            let delta = delta_visc + delta_rad;

            let denom = (omega0 * omega0 - omega * omega).powi(2) + (delta * omega).powi(2);
            if denom <= 0.0 || !denom.is_finite() {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: "scattering_denom".to_string(),
                    value: denom,
                    reason: "must be positive and finite".to_string(),
                }));
            }

            let sigma = 4.0 * std::f64::consts::PI * omega.powi(4) * radius_eq.powi(2) / denom;
            if !sigma.is_finite() || sigma < 0.0 {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: "scattering_cross_section".to_string(),
                    value: sigma,
                    reason: "must be non-negative and finite".to_string(),
                }));
            }
            Ok(sigma)
        };

        if std_dev == 0.0 {
            return scattering_at_radius(mean_radius);
        }

        let variance = std_dev * std_dev;
        let sigma2_ln = (1.0 + variance / (mean_radius * mean_radius)).ln();
        let sigma_ln = sigma2_ln.sqrt();
        if !sigma_ln.is_finite() || sigma_ln <= 0.0 {
            return scattering_at_radius(mean_radius);
        }
        let mu_ln = mean_radius.ln() - 0.5 * sigma2_ln;

        let a = mu_ln - 6.0 * sigma_ln;
        let b = mu_ln + 6.0 * sigma_ln;
        let n: usize = 128;
        let h = (b - a) / n as f64;
        let norm = 1.0 / (sigma_ln * (2.0 * std::f64::consts::PI).sqrt());

        let mut sum = 0.0;
        for i in 0..=n {
            let y = a + h * i as f64;
            let z = (y - mu_ln) / sigma_ln;
            let pdf = norm * (-0.5 * z * z).exp();
            let radius = y.exp();
            let integrand = scattering_at_radius(radius)? * pdf;
            let weight = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            sum += weight * integrand;
        }

        let result = (h / 3.0) * sum;
        if !result.is_finite() || result < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "effective_scattering".to_string(),
                value: result,
                reason: "must be non-negative and finite".to_string(),
            }));
        }

        Ok(result)
    }
}

/// Size distribution parameters
#[derive(Debug, Clone)]
pub struct SizeDistribution {
    /// Mean radius (m)
    pub mean_radius: f64,
    /// Standard deviation (m)
    pub std_dev: f64,
}

/// Perfusion map containing quantitative perfusion parameters
#[derive(Debug, Clone)]
pub struct PerfusionMap {
    /// Peak intensity (dB)
    pub peak_intensity: Array3<f64>,
    /// Time to peak intensity (s)
    pub time_to_peak: Array3<f64>,
    /// Area under the time-intensity curve (dB·s)
    pub area_under_curve: Array3<f64>,
}

impl PerfusionMap {
    /// Get perfusion statistics for a region of interest
    pub fn roi_statistics(
        &self,
        x_range: (usize, usize),
        y_range: (usize, usize),
        z_range: (usize, usize),
    ) -> PerfusionStatistics {
        let mut peak_values = Vec::new();
        let mut ttp_values = Vec::new();
        let mut auc_values = Vec::new();

        for i in x_range.0..=x_range.1 {
            for j in y_range.0..=y_range.1 {
                for k in z_range.0..=z_range.1 {
                    if self.peak_intensity[[i, j, k]] > 0.0 {
                        peak_values.push(self.peak_intensity[[i, j, k]]);
                        ttp_values.push(self.time_to_peak[[i, j, k]]);
                        auc_values.push(self.area_under_curve[[i, j, k]]);
                    }
                }
            }
        }

        PerfusionStatistics::from_samples(&peak_values, &ttp_values, &auc_values)
    }
}

/// Perfusion statistics for a region of interest
#[derive(Debug, Clone)]
pub struct PerfusionStatistics {
    /// Mean peak intensity (dB)
    pub mean_peak: f64,
    /// Standard deviation of peak intensity
    pub std_peak: f64,
    /// Mean time to peak (s)
    pub mean_ttp: f64,
    /// Standard deviation of time to peak
    pub std_ttp: f64,
    /// Mean area under curve (dB·s)
    pub mean_auc: f64,
    /// Standard deviation of area under curve
    pub std_auc: f64,
}

impl PerfusionStatistics {
    pub fn from_samples(peaks: &[f64], ttp: &[f64], auc: &[f64]) -> Self {
        let n = peaks.len() as f64;
        let mean_peak = peaks.iter().sum::<f64>() / n;
        let std_peak = (peaks.iter().map(|x| (x - mean_peak).powi(2)).sum::<f64>() / n).sqrt();
        let mean_ttp = ttp.iter().sum::<f64>() / n;
        let std_ttp = (ttp.iter().map(|x| (x - mean_ttp).powi(2)).sum::<f64>() / n).sqrt();
        let mean_auc = auc.iter().sum::<f64>() / n;
        let std_auc = (auc.iter().map(|x| (x - mean_auc).powi(2)).sum::<f64>() / n).sqrt();

        Self {
            mean_peak,
            std_peak,
            mean_ttp,
            std_ttp,
            mean_auc,
            std_auc,
        }
    }
}
