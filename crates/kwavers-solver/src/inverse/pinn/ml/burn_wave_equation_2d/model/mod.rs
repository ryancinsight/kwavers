//! 2D PINN model components.

pub mod network;
pub mod wave_speed;

pub use network::BurnPINN2DWave;
pub use wave_speed::WaveSpeedFn;
