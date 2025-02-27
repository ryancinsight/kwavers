// physics/mechanics/viscosity/shear_viscosity.rs
use log::debug;
use crate::physics::mechanics::viscosity::ViscosityModel;

#[derive(Debug)]
pub struct ShearViscosityModel {
    base_viscosity: f64,    // Base viscosity at reference temperature (Pa·s)
    ref_temperature: f64,   // Reference temperature (K)
    temp_sensitivity: f64,  // Temperature sensitivity factor (1/K)
}

impl ShearViscosityModel {
    pub fn new(base_viscosity: f64, ref_temperature: f64, temp_sensitivity: f64) -> Self {
        debug!(
            "Initializing ShearViscosityModel: base = {:.6e} Pa·s, ref_temp = {:.2} K, sensitivity = {:.6e} 1/K",
            base_viscosity, ref_temperature, temp_sensitivity
        );
        Self {
            base_viscosity,
            ref_temperature,
            temp_sensitivity,
        }
    }
}

impl ViscosityModel for ShearViscosityModel {
    fn viscosity(&self, _x: f64, _y: f64, _z: f64, temperature: f64) -> f64 {
        // Simplified temperature-dependent viscosity (e.g., water-like behavior)
        let delta_t = temperature - self.ref_temperature;
        self.base_viscosity * (-self.temp_sensitivity * delta_t).exp()
    }
}