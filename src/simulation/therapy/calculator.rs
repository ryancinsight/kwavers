//! Therapy calculator orchestration

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::therapy::types::{TherapyModality, TherapyParameters, TreatmentMetrics};
use crate::physics::thermal::PennesSolver;
use ndarray::{Array3, Zip};
use std::sync::Arc;

/// Main therapy calculator
#[derive(Debug)]
pub struct TherapyCalculator {
    /// Treatment modality
    pub modality: TherapyModality,
    /// Treatment parameters
    pub parameters: TherapyParameters,
    /// Thermal calculator (optional)
    pub thermal: Option<PennesSolver>,
    /// Treatment metrics
    pub metrics: TreatmentMetrics,
    /// Grid reference
    grid_shape: (usize, usize, usize),
}

impl TherapyCalculator {
    /// Create a new therapy calculator
    pub fn new(modality: TherapyModality, parameters: TherapyParameters, grid: &Grid) -> Self {
        // Initialize components based on modality
        let thermal = if modality.has_thermal_effects() {
            use crate::physics::thermal::ThermalPropertyData;
            let properties = ThermalPropertyData::new(
                0.5,          // conductivity (W/m/K)
                3600.0,       // specific_heat (J/kg/K)
                1050.0,       // density (kg/m³)
                Some(0.5e-3), // blood_perfusion (kg/m³/s)
                Some(3617.0), // blood_specific_heat (J/kg/K)
            )
            .expect("Valid thermal properties");
            let arterial_temperature = 37.0; // °C
            let metabolic_heat = 420.0; // W/m³
            PennesSolver::new(
                grid.nx,
                grid.ny,
                grid.nz,
                grid.dx,
                grid.dy,
                grid.dz,
                0.001, // dt = 1ms
                properties,
                arterial_temperature,
                metabolic_heat,
            )
            .ok()
        } else {
            None
        };

        Self {
            modality,
            parameters,
            thermal,
            metrics: TreatmentMetrics::default(),
            grid_shape: (grid.nx, grid.ny, grid.nz),
        }
    }

    /// Calculate therapy effects
    pub fn calculate(
        &mut self,
        pressure: &Array3<f64>,
        temperature: &mut Array3<f64>,
        dt: f64,
        medium: &Arc<dyn Medium>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        // Calculate thermal effects if applicable
        if self.thermal.is_some() {
            // Calculate heat source from acoustic absorption
            let heat_source = self.calculate_heat_source(pressure, medium, grid)?;

            // Update temperature
            if let Some(ref mut thermal_calc) = self.thermal {
                thermal_calc.step(&heat_source);
            }

            // Update thermal dose
            self.metrics.thermal_dose += TreatmentMetrics::calculate_thermal_dose(temperature, dt);
            self.metrics.update_peak_temperature(temperature);
        }

        // Update safety and efficiency metrics
        self.metrics.calculate_safety_index();
        self.metrics.calculate_efficiency(self.get_target_dose());

        Ok(())
    }

    /// Calculate heat source from acoustic absorption
    fn calculate_heat_source(
        &self,
        pressure: &Array3<f64>,
        medium: &Arc<dyn Medium>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let mut heat_source = Array3::zeros(self.grid_shape);

        // Q = 2 * α * I, where I = p²/(2*ρ*c)
        Zip::indexed(&mut heat_source)
            .and(pressure)
            .for_each(|(i, j, k), q, &p| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = crate::domain::medium::density_at(medium.as_ref(), x, y, z, grid);
                let c = crate::domain::medium::sound_speed_at(medium.as_ref(), x, y, z, grid);
                let alpha = crate::domain::medium::AcousticProperties::absorption_coefficient(
                    medium.as_ref(),
                    x,
                    y,
                    z,
                    grid,
                    self.parameters.frequency,
                );

                // Acoustic intensity
                let intensity = p * p / (2.0 * rho * c);

                // Heat generation rate [W/m³]
                *q = 2.0 * alpha * intensity;
            });

        Ok(heat_source)
    }

    /// Get target dose based on modality
    fn get_target_dose(&self) -> f64 {
        match self.modality {
            TherapyModality::HIFU => 240.0,      // 240 CEM43 for ablation
            TherapyModality::LIFU => 0.0,        // No thermal goal
            TherapyModality::Histotripsy => 0.0, // Mechanical disruption
            TherapyModality::BBBOpening => 0.0,  // Mechanical opening
            _ => 10.0,                           // Default mild hyperthermia
        }
    }

    /// Check if treatment is complete
    pub fn is_complete(&self) -> bool {
        self.metrics.is_successful(self.get_target_dose(), 0.8)
    }

    /// Get treatment summary
    pub fn summary(&self) -> String {
        format!(
            "Therapy: {:?}\n\
             Frequency: {:.1} MHz\n\
             Pressure: {:.1} MPa\n\
             Duration: {:.1} s\n\
             {}",
            self.modality,
            self.parameters.frequency / 1e6,
            self.parameters.peak_negative_pressure / 1e6,
            self.parameters.treatment_duration,
            self.metrics.summary()
        )
    }
}
