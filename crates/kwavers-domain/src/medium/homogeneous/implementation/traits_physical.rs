use ndarray::Array3;

use kwavers_grid::Grid;
use crate::medium::{
    acoustic::AcousticProperties,
    bubble::{BubbleProperties, BubbleState},
    elastic::{ElasticArrayAccess, ElasticProperties},
    optical::MediumOpticalProperties,
    thermal::{ThermalField, ThermalProperties},
    viscous::ViscousProperties,
};

use super::HomogeneousMedium;

impl AcousticProperties for HomogeneousMedium {
    fn absorption_coefficient(
        &self,
        _x: f64,
        _y: f64,
        _z: f64,
        _grid: &Grid,
        frequency: f64,
    ) -> f64 {
        self.absorption_alpha * (frequency / self.reference_frequency).powf(self.absorption_power)
    }

    fn alpha_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.absorption_alpha
    }

    fn alpha_power(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.absorption_power
    }

    fn nonlinearity_parameter(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.nonlinearity
    }

    fn acoustic_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.thermal_conductivity / (self.density * self.specific_heat * self.sound_speed)
    }
}

impl ElasticProperties for HomogeneousMedium {
    fn lame_lambda(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.lame_lambda
    }

    fn lame_mu(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.lame_mu
    }
}

impl ElasticArrayAccess for HomogeneousMedium {
    fn lame_lambda_array(&self) -> Array3<f64> {
        Array3::from_elem(self.grid_shape, self.lame_lambda)
    }

    fn lame_mu_array(&self) -> Array3<f64> {
        Array3::from_elem(self.grid_shape, self.lame_mu)
    }

    fn shear_sound_speed_array(&self) -> Array3<f64> {
        // c_s = sqrt(μ / ρ)
        let shear_speed = if self.density > 0.0 {
            (self.lame_mu / self.density).sqrt()
        } else {
            0.0
        };
        Array3::from_elem(self.grid_shape, shear_speed)
    }
}

impl ThermalProperties for HomogeneousMedium {
    fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.specific_heat
    }

    fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.thermal_conductivity
    }

    fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.thermal_conductivity / (self.density * self.specific_heat)
    }

    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.thermal_expansion
    }
}

impl ThermalField for HomogeneousMedium {
    fn update_thermal_field(&mut self, temperature: &Array3<f64>) {
        self.temperature = temperature.clone();
    }

    fn thermal_field(&self) -> &Array3<f64> {
        &self.temperature
    }
}

impl MediumOpticalProperties for HomogeneousMedium {
    fn optical_absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.optical_absorption
    }

    fn optical_scattering_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.optical_scattering
    }
}

impl ViscousProperties for HomogeneousMedium {
    fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.viscosity
    }

    fn shear_viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.shear_viscosity
    }

    fn bulk_viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.bulk_viscosity
    }
}

impl BubbleProperties for HomogeneousMedium {
    fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.surface_tension
    }

    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.ambient_pressure
    }

    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.vapor_pressure
    }

    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.polytropic_index
    }

    fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.gas_diffusion
    }
}

impl BubbleState for HomogeneousMedium {
    fn bubble_radius(&self) -> &Array3<f64> {
        &self.bubble_radius
    }

    fn bubble_velocity(&self) -> &Array3<f64> {
        &self.bubble_velocity
    }

    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        self.bubble_radius = radius.clone();
        self.bubble_velocity = velocity.clone();
    }
}
