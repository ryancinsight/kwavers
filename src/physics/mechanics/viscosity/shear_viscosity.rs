// physics/mechanics/viscosity/shear_viscosity.rs
use crate::physics::mechanics::viscosity::ViscosityModel;
use log::debug;

#[derive(Debug, Debug))]
pub struct ShearViscosityModel {
    base_viscosity: f64,   // Base viscosity at reference temperature (Pa·s)
    ref_temperature: f64,  // Reference temperature (K)
    temp_sensitivity: f64, // Temperature sensitivity factor (1/K)
}

impl ShearViscosityModel {
    pub fn new(base_viscosity: f64, ref_temperature: f64, temp_sensitivity: f64) -> Self {
        debug!(
            "Initializing ShearViscosityModel: base = {:.6e} Pa·s, ref_temperature = {:.2} K, sensitivity = {:.6e} 1/K",
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
    fn viscosity(&self, x: f64, y: f64, z: f64, temperature: f64) -> f64 {
        // Temperature-dependent viscosity with potential for spatial variation
        // Currently uniform in space, but position parameters preserved for
        // future heterogeneous media support
        let _ = (x, y, z); // Acknowledge spatial parameters for future use
        let delta_t = temperature - self.ref_temperature;
        self.base_viscosity * (-self.temp_sensitivity * delta_t).exp()
    }
}
