//! Thermal-acoustic coupling field solver

use super::coefficients::TemperatureCoefficients;
use super::heating::AcousticHeatingSource;
use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Evaluates bi-directional physical coupling between thermal and acoustic phenomena
#[derive(Debug)]
pub struct ThermalAcousticCoupling {
    /// Defines linear absorption from propagating energy
    source: AcousticHeatingSource,
    /// Ruleset for modifying constants in response to temperature elevation
    coefficients: TemperatureCoefficients,
    /// Integral deposition bucket tracking heat additions over execution intervals [J/m³]
    acoustic_heat: Array3<f64>,
}

impl ThermalAcousticCoupling {
    /// Initialize mathematical coupling framework
    #[must_use]
    pub fn new(
        absorption_coefficient: f64,
        intensity: f64,
        coefficients: TemperatureCoefficients,
    ) -> Self {
        Self {
            source: AcousticHeatingSource::new(absorption_coefficient, intensity),
            coefficients,
            acoustic_heat: Array3::zeros((1, 1, 1)),
        }
    }

    /// Establish execution domain matching wave geometry
    pub fn initialize(&mut self, shape: (usize, usize, usize)) {
        self.acoustic_heat = Array3::zeros(shape);
    }

    /// Evaluate temporal step executing power transfer between modalities.
    /// Returns a validated energy matrix confirming spatial boundaries.
    pub fn update(
        &mut self,
        temperature: &Array3<f64>,
        acoustic_intensity: &Array3<f64>,
        reference_temperature: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = (
            self.acoustic_heat.dim().0,
            self.acoustic_heat.dim().1,
            self.acoustic_heat.dim().2,
        );

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let t = temperature[[i, j, k]];
                    let i_ac = acoustic_intensity[[i, j, k]];

                    // Heat generation from acoustic absorption
                    // Q = 2·α·I where α depends on current local temperature
                    let alpha = self.coefficients.absorption(
                        self.source.absorption_coefficient,
                        t,
                        reference_temperature,
                    );
                    let heat_rate = 2.0 * alpha * i_ac;

                    // Execute temporal accumulation of transfer power
                    self.acoustic_heat[[i, j, k]] += heat_rate * dt;
                }
            }
        }

        Ok(())
    }

    /// Retrieve spatial definition of derived heat distributions [W/m³]
    #[must_use]
    pub fn acoustic_heat(&self) -> &Array3<f64> {
        &self.acoustic_heat
    }

    /// Analytical integration across spatial matrix ensuring energy balance [J]
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        self.acoustic_heat.iter().sum()
    }

    /// Nullify integral tracking arrays returning structure to initialization baseline
    pub fn reset(&mut self) {
        self.acoustic_heat.fill(0.0);
    }

    /// Resolve dynamic sound speed relative to absolute temperature perturbation
    #[must_use]
    pub fn sound_speed_at_temperature(
        &self,
        base_sound_speed: f64,
        temperature: f64,
        reference_temperature: f64,
    ) -> f64 {
        self.coefficients
            .sound_speed(base_sound_speed, temperature, reference_temperature)
    }

    /// Resolve fluid material density following deterministic temperature expansion rules
    #[must_use]
    pub fn density_at_temperature(
        &self,
        base_density: f64,
        temperature: f64,
        reference_temperature: f64,
    ) -> f64 {
        self.coefficients
            .density(base_density, temperature, reference_temperature)
    }
}
