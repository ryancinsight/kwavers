//! Therapy calculator orchestration

use crate::parallel::zip_indexed_mut_ref3;
use aequitas::systems::si::quantities::{
    MassDensity, MassDensityRate, SpecificHeatCapacity, ThermalConductivity,
};
use kwavers_core::constants::fundamental::DENSITY_TISSUE;
use kwavers_core::constants::medical::{BLOOD_SPECIFIC_HEAT, TISSUE_PERFUSION_RATE};
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
use kwavers_core::constants::tissue_thermal::SPECIFIC_HEAT_TISSUE;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::properties::ThermalPropertyData;
use kwavers_medium::Medium;
use kwavers_physics::therapy::types::{
    DomainTherapyModality, DomainTherapyParameters, DomainTreatmentMetrics,
};
use kwavers_solver::forward::thermal::PennesSolver;
use leto::Array3;
use std::sync::Arc;

/// Main therapy calculator
#[derive(Debug)]
pub struct TherapyCalculator {
    /// Treatment modality
    pub modality: DomainTherapyModality,
    /// Treatment parameters
    pub parameters: DomainTherapyParameters,
    /// Thermal calculator (optional)
    pub thermal: Option<PennesSolver>,
    /// Treatment metrics
    pub metrics: DomainTreatmentMetrics,
    /// Grid reference
    grid_shape: (usize, usize, usize),
}

impl TherapyCalculator {
    /// Create a new therapy calculator
    /// # Panics
    /// - Panics if `Valid thermal properties`.
    ///
    pub fn new(
        modality: DomainTherapyModality,
        parameters: DomainTherapyParameters,
        grid: &Grid,
    ) -> Self {
        // Initialize components based on modality
        let thermal = if modality.has_thermal_effects() {
            let properties = ThermalPropertyData::new(
                ThermalConductivity::from_base(0.5),
                SpecificHeatCapacity::from_base(SPECIFIC_HEAT_TISSUE),
                MassDensity::from_base(DENSITY_TISSUE),
                Some(MassDensityRate::from_base(TISSUE_PERFUSION_RATE)),
                Some(SpecificHeatCapacity::from_base(BLOOD_SPECIFIC_HEAT)),
            )
            .expect("Valid thermal properties");
            let arterial_temperature = BODY_TEMPERATURE_C;
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
            metrics: DomainTreatmentMetrics::default(),
            grid_shape: (grid.nx, grid.ny, grid.nz),
        }
    }

    /// Calculate therapy effects
    /// # Errors
    /// - Propagates errors returned by called functions.
    ///
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
            self.metrics.thermal_dose +=
                DomainTreatmentMetrics::calculate_thermal_dose(temperature, dt)?;
            self.metrics.update_peak_temperature(temperature);
        }

        // Update safety and efficiency metrics
        self.metrics.calculate_safety_index();
        self.metrics.calculate_efficiency(self.get_target_dose());

        Ok(())
    }

    /// Calculate heat source from acoustic absorption
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn calculate_heat_source(
        &self,
        pressure: &Array3<f64>,
        medium: &Arc<dyn Medium>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let mut heat_source = Array3::zeros(self.grid_shape);

        // Q = 2 * α * I, where I = p²/(2*ρ*c)
        zip_indexed_mut_ref3(
            heat_source.view_mut(),
            pressure.view(),
            |(i, j, k), q, &p| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = kwavers_medium::density_at(medium.as_ref(), x, y, z, grid);
                let c = kwavers_medium::sound_speed_at(medium.as_ref(), x, y, z, grid);
                let alpha = kwavers_medium::AcousticProperties::absorption_coefficient(
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
            },
        );

        Ok(heat_source)
    }

    /// Get target dose based on modality
    fn get_target_dose(&self) -> f64 {
        match self.modality {
            DomainTherapyModality::HIFU => 240.0, // 240 CEM43 for ablation
            DomainTherapyModality::LIFU => 0.0,   // No thermal goal
            DomainTherapyModality::Histotripsy => 0.0, // Mechanical disruption
            DomainTherapyModality::BBBOpening => 0.0, // Mechanical opening
            _ => 10.0,                            // Default mild hyperthermia
        }
    }

    /// Check if treatment is complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.metrics.is_successful(self.get_target_dose(), 0.8)
    }

    /// Get treatment summary
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Therapy: {:?}\n\
             Frequency: {:.1} MHz\n\
             Pressure: {:.1} MPa\n\
             Duration: {:.1} s\n\
             {}",
            self.modality,
            self.parameters.frequency / MHZ_TO_HZ,
            self.parameters.peak_negative_pressure / MPA_TO_PA,
            self.parameters.treatment_duration,
            self.metrics.summary()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_medium::HomogeneousMedium;

    #[test]
    fn heat_source_matches_absorbed_intensity_formula() {
        let grid = Grid::new(2, 2, 1, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let density = 1_000.0;
        let sound_speed = 1_500.0;
        let absorption = 3.0;
        let mut medium = HomogeneousMedium::new(density, sound_speed, 0.0, 0.0, &grid);
        medium
            .set_acoustic_properties(absorption, 0.0, 5.0)
            .unwrap();
        let medium: Arc<dyn Medium> = Arc::new(medium);
        let parameters = DomainTherapyParameters::new(1.0e6, 1.0e6, 1.0);
        let calculator = TherapyCalculator::new(DomainTherapyModality::HIFU, parameters, &grid);
        let pressure =
            Array3::from_shape_vec((grid.nx, grid.ny, grid.nz), vec![0.0, 10.0, 20.0, 30.0])
                .unwrap();

        let heat_source = calculator
            .calculate_heat_source(&pressure, &medium, &grid)
            .unwrap();
        let expected = pressure.mapv(|p| absorption * p * p / (density * sound_speed));

        assert_eq!(heat_source, expected);
    }
}
