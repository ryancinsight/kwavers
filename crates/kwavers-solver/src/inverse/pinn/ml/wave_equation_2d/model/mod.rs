//! 2D PINN model components.

pub mod network;
pub mod wave_speed;

pub use network::PinnWave2D;
pub use wave_speed::WaveSpeedFn;
