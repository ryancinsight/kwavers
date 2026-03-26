use super::models::PermeabilityModels;
use super::types::{BBBParameters, PermeabilityEnhancement};
use crate::core::error::KwaversResult;
use log::info;
use ndarray::Array3;

/// BBB opening simulation
#[derive(Debug)]
pub struct BBBOpening {
    /// Acoustic pressure field (Pa)
    pub acoustic_pressure: Array3<f64>,
    /// Microbubble concentration (bubbles/mL)
    pub microbubble_concentration: Array3<f64>,
    /// Treatment parameters
    pub parameters: BBBParameters,
    /// Permeability enhancement results
    pub permeability: PermeabilityEnhancement,
}

impl BBBOpening {
    /// Create new BBB opening simulation
    pub fn new(
        acoustic_pressure: Array3<f64>,
        microbubble_concentration: Array3<f64>,
        parameters: BBBParameters,
    ) -> Self {
        let dims = acoustic_pressure.dim();
        let permeability = PermeabilityEnhancement {
            permeability_factor: Array3::zeros(dims),
            opening_duration: Array3::zeros(dims),
            recovery_time: Array3::zeros(dims),
            microbubble_effect: Array3::zeros(dims),
        };

        Self {
            acoustic_pressure,
            microbubble_concentration,
            parameters,
            permeability,
        }
    }

    /// Simulate BBB opening process
    pub fn simulate_opening(&mut self) -> KwaversResult<()> {
        info!("Simulating BBB opening with parameters:");
        info!("  Frequency: {:.1} MHz", self.parameters.frequency / 1e6);
        info!("  MI target: {:.2}", self.parameters.target_mi);
        info!("  Duration: {:.1} s", self.parameters.duration);

        let (nx, ny, nz) = self.acoustic_pressure.dim();
        let models = PermeabilityModels::new(&self.parameters);

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let pressure = self.acoustic_pressure[[i, j, k]];
                    let bubble_conc = self.microbubble_concentration[[i, j, k]];

                    // Calculate local permeability enhancement
                    let enhancement = models.calculate_permeability_enhancement(pressure, bubble_conc);
                    self.permeability.permeability_factor[[i, j, k]] = enhancement;

                    // Calculate opening duration
                    let duration = models.calculate_opening_duration(pressure, bubble_conc);
                    self.permeability.opening_duration[[i, j, k]] = duration;

                    // Calculate recovery time
                    let recovery = models.calculate_recovery_time(pressure, enhancement);
                    self.permeability.recovery_time[[i, j, k]] = recovery;

                    // Calculate microbubble effect
                    let effect = models.calculate_microbubble_effect(bubble_conc, pressure);
                    self.permeability.microbubble_effect[[i, j, k]] = effect;
                }
            }
        }

        Ok(())
    }

    /// Get permeability enhancement results
    pub fn permeability(&self) -> &PermeabilityEnhancement {
        &self.permeability
    }
}
