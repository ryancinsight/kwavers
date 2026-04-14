//! PINN Electromagnetic Residuals
//!
//! Extracted into bounded contexts to comply with SRP and SoC.

pub mod constants;
pub mod scalar_wave;
pub mod sources;
pub mod statics;
pub mod te_mode;
pub mod tm_mode;

pub use constants::EPS_FD_F32;
pub use scalar_wave::wave_propagation_residual;
pub use sources::{compute_charge_density, compute_current_density_z};
pub use statics::{electrostatic_residual, magnetostatic_residual, quasi_static_residual};
pub use te_mode::{
    te_mode_ampere_x_residual, te_mode_ampere_y_residual, te_mode_faraday_residual,
    te_mode_gauss_residual,
};
pub use tm_mode::{
    tm_mode_ampere_z_residual, tm_mode_faraday_x_residual, tm_mode_faraday_y_residual,
};
