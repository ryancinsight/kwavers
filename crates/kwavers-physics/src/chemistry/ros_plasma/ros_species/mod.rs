//! Reactive Oxygen Species (ROS) definitions and concentration tracking
//!
//! Key ROS generated during sonoluminescence:
//! - Hydroxyl radical (•OH)
//! - Hydrogen peroxide (H₂O₂)
//! - Superoxide (O₂•⁻)
//! - Singlet oxygen (¹O₂)
//! - Ozone (O₃)

pub mod concentrations;
pub mod diffusion;
pub mod generation;
pub mod properties;
pub mod types;

#[cfg(test)]
mod tests;

pub use concentrations::ROSConcentrations;
pub use generation::calculate_ros_generation;
pub use types::ROSSpecies;
