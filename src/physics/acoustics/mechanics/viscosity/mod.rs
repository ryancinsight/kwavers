// physics/mechanics/viscosity/mod.rs
pub mod shear_viscosity;

pub use shear_viscosity::ShearViscosityModel;

pub trait ViscosityModel: Send + Sync {
    fn viscosity(&self, x: f64, y: f64, z: f64, temperature: f64) -> f64;
}
