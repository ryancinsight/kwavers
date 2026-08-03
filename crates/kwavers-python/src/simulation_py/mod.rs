mod checkpoint;
mod config;
pub(crate) mod gpu;
pub(crate) mod py_convert;
mod run;
mod solvers;
mod tests;
pub use gpu::GpuPstdSession;

/// Elastic velocity source bundle: (mask, ux_signal, uy_signal, uz_signal, mode).
mod simulation;
pub use simulation::Simulation;

pub(crate) type ElasticVelocitySource = Option<(
    leto::Array3<bool>,
    Option<leto::Array1<f64>>,
    Option<leto::Array1<f64>>,
    Option<leto::Array1<f64>>,
    String,
)>;
