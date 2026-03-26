//! Core sonochemistry model and yield tracking

use super::super::{
    plasma_reactions::PlasmaChemistry,
    radical_kinetics::RadicalKinetics,
    ros_species::{calculate_ros_generation, ROSConcentrations, ROSSpecies},
};
use super::transfer::estimate_collapse_energy;
use crate::core::constants::fundamental::AVOGADRO;
use ndarray::Array3;
use std::collections::HashMap;

/// Sonochemical yield parameters
#[derive(Debug, Clone)]
pub struct SonochemicalYield {
    /// OH radical yield per collapse (molecules)
    pub oh_yield: f64,
    /// H2O2 yield per collapse (molecules)
    pub h2o2_yield: f64,
    /// Total ROS yield per collapse (molecules)
    pub total_ros_yield: f64,
    /// Energy efficiency (molecules/J)
    pub energy_efficiency: f64,
}

/// Bubble state for sonochemistry calculations
#[derive(Debug, Clone)]
pub struct BubbleState {
    pub radius: f64,
    pub temperature: f64,
    pub pressure_internal: f64,
    pub n_gas: f64,
    pub n_vapor: f64,
    pub is_collapsing: bool,
}

/// Integrated sonochemistry model
#[derive(Debug)]
pub struct SonochemistryModel {
    /// ROS concentration fields
    pub ros_concentrations: ROSConcentrations,
    /// Plasma chemistry solver
    pub plasma_chemistry: Option<PlasmaChemistry>,
    /// Radical kinetics solver
    pub radical_kinetics: RadicalKinetics,
    /// Sonochemical yields
    pub yields: HashMap<(usize, usize, usize), SonochemicalYield>,
    /// Grid shape
    shape: (usize, usize, usize),
    /// pH field
    pub(crate) ph_field: Array3<f64>,
}

impl SonochemistryModel {
    /// Create new sonochemistry model
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, initial_ph: f64) -> Self {
        let shape = (nx, ny, nz);

        Self {
            ros_concentrations: ROSConcentrations::new(nx, ny, nz),
            plasma_chemistry: None,
            radical_kinetics: RadicalKinetics::new(initial_ph, 298.15),
            yields: HashMap::new(),
            shape,
            ph_field: Array3::from_elem(shape, initial_ph),
        }
    }

    /// Update ROS generation from bubble collapse
    pub fn update_ros_generation(
        &mut self,
        bubble_states: &Array3<BubbleState>,
        dt: f64,
        grid_spacing: (f64, f64, f64),
    ) {
        self.yields.clear();

        for ((i, j, k), state) in bubble_states.indexed_iter() {
            if state.is_collapsing && state.temperature > 2000.0 {
                let water_vapor_fraction = state.n_vapor / (state.n_gas + state.n_vapor);
                let generation_rates = calculate_ros_generation(
                    state.temperature,
                    state.pressure_internal,
                    water_vapor_fraction,
                );

                if state.temperature > 3000.0 {
                    let mut plasma =
                        PlasmaChemistry::new(state.temperature, state.pressure_internal);

                    let collapse_time = 1e-9; // ~1 ns typical
                    plasma.update(collapse_time);

                    let plasma_ros = plasma.get_ros_concentrations();

                    let bubble_volume = 4.0 / 3.0 * std::f64::consts::PI * state.radius.powi(3);
                    let mut yield_data = SonochemicalYield {
                        oh_yield: 0.0,
                        h2o2_yield: 0.0,
                        total_ros_yield: 0.0,
                        energy_efficiency: 0.0,
                    };

                    for (species, &conc) in &plasma_ros {
                        let molecules = conc * bubble_volume * AVOGADRO;

                        match species {
                            ROSSpecies::HydroxylRadical => yield_data.oh_yield = molecules,
                            ROSSpecies::HydrogenPeroxide => yield_data.h2o2_yield = molecules,
                            _ => {}
                        }

                        yield_data.total_ros_yield += molecules;
                    }

                    let collapse_energy = estimate_collapse_energy(state);
                    if collapse_energy > 0.0 {
                        yield_data.energy_efficiency = yield_data.total_ros_yield / collapse_energy;
                    }

                    self.yields.insert((i, j, k), yield_data);

                    self.transfer_ros_to_liquid(i, j, k, plasma_ros, state.radius);
                } else {
                    for (species, rate) in generation_rates {
                        if let Some(conc) = self.ros_concentrations.get_mut(species) {
                            conc[[i, j, k]] += rate * dt;
                        }
                    }
                }
            }
        }

        self.update_liquid_phase_chemistry(dt, grid_spacing);
    }

    /// Transfer ROS from bubble to surrounding liquid
    fn transfer_ros_to_liquid(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        bubble_ros: HashMap<ROSSpecies, f64>,
        bubble_radius: f64,
    ) {
        let shell_thickness = bubble_radius * 0.1;
        let shell_volume = 4.0 / 3.0
            * std::f64::consts::PI
            * ((bubble_radius + shell_thickness).powi(3) - bubble_radius.powi(3));

        for (species, &bubble_conc) in &bubble_ros {
            if let Some(liquid_conc) = self.ros_concentrations.get_mut(*species) {
                let transfer_fraction = match species {
                    ROSSpecies::HydroxylRadical => 0.9,
                    ROSSpecies::HydrogenPeroxide => 1.0,
                    ROSSpecies::AtomicOxygen => 0.1,
                    ROSSpecies::AtomicHydrogen => 0.1,
                    _ => 0.5,
                };

                let bubble_volume = 4.0 / 3.0 * std::f64::consts::PI * bubble_radius.powi(3);
                let transferred_amount = bubble_conc * bubble_volume * transfer_fraction;
                let liquid_concentration = transferred_amount / shell_volume;

                liquid_conc[[i, j, k]] += liquid_concentration;
            }
        }
    }

    /// Update liquid phase radical chemistry
    fn update_liquid_phase_chemistry(&mut self, dt: f64, grid_spacing: (f64, f64, f64)) {
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                for k in 0..self.shape.2 {
                    let mut local_conc = HashMap::new();
                    for (species, field) in &self.ros_concentrations.fields {
                        local_conc.insert(*species, field[[i, j, k]]);
                    }

                    self.radical_kinetics.ph = self.ph_field[[i, j, k]];

                    let rate_changes = self.radical_kinetics.calculate_rates(&local_conc);

                    for (species, rate) in rate_changes {
                        if let Some(field) = self.ros_concentrations.get_mut(species) {
                            field[[i, j, k]] = (field[[i, j, k]] + rate * dt).max(0.0);
                        }
                    }
                }
            }
        }

        let (dx, dy, dz) = grid_spacing;
        self.ros_concentrations.apply_diffusion(dx, dy, dz, dt);
        self.ros_concentrations.apply_decay(dt);
        self.ros_concentrations.update_total();
    }

    /// Calculate total sonochemical yield
    #[must_use]
    pub fn total_yield(&self) -> SonochemicalYield {
        let mut total = SonochemicalYield {
            oh_yield: 0.0,
            h2o2_yield: 0.0,
            total_ros_yield: 0.0,
            energy_efficiency: 0.0,
        };

        for yield_data in self.yields.values() {
            total.oh_yield += yield_data.oh_yield;
            total.h2o2_yield += yield_data.h2o2_yield;
            total.total_ros_yield += yield_data.total_ros_yield;
        }

        if !self.yields.is_empty() {
            total.energy_efficiency = self
                .yields
                .values()
                .map(|y| y.energy_efficiency)
                .sum::<f64>()
                / self.yields.len() as f64;
        }

        total
    }

    /// Get oxidative stress field
    #[must_use]
    pub fn oxidative_stress(&self) -> Array3<f64> {
        let stress_value = self.ros_concentrations.oxidative_stress_index();
        Array3::from_elem(self.shape, stress_value)
    }

    /// Update pH based on chemical reactions
    pub fn update_ph(&mut self, dt: f64) {
        if let (Some(h2o2), Some(oh)) = (
            self.ros_concentrations.get(ROSSpecies::HydrogenPeroxide),
            self.ros_concentrations.get(ROSSpecies::HydroxylRadical),
        ) {
            for ((i, j, k), ph) in self.ph_field.indexed_iter_mut() {
                let h2o2_effect = -0.1 * h2o2[[i, j, k]] / 1e-3;
                let oh_effect = 0.5 * oh[[i, j, k]] / 1e-6;
                *ph = (*ph + (h2o2_effect + oh_effect) * dt).clamp(2.0, 12.0);
            }
        }
    }
}
