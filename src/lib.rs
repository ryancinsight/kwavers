// src/lib.rs
pub mod boundary;
pub mod config;
pub mod grid;
pub mod log;
pub mod medium;
pub mod output;
pub mod physics;
pub mod plotting;
pub mod recorder;
pub mod sensor;
pub mod signal;
pub mod solver;
pub mod source;
pub mod time;
pub mod utils;
pub mod fft;

pub use fft::{Fft3d, Ifft3d};
pub use boundary::{pml::PMLBoundary, Boundary};
pub use config::{Config, OutputConfig, SimulationConfig, SourceConfig};
pub use grid::Grid;
pub use log::init_logging;
pub use medium::{
    absorption::power_law_absorption, heterogeneous::HeterogeneousMedium,
    homogeneous::HomogeneousMedium, Medium,
};
pub use output::{generate_summary, save_light_data, save_pressure_data};
pub use physics::{
    mechanics::cavitation::CavitationModel,
    mechanics::streaming::StreamingModel,
    mechanics::acoustic_wave::NonlinearWave,
    mechanics::viscosity::ViscosityModel,
    chemistry::ChemicalModel,
    optics::diffusion::LightDiffusion,
    thermodynamics::heat_transfer::ThermalModel,
    scattering::acoustic::AcousticScatteringModel,
    scattering::acoustic::{bubble_interactions, mie, rayleigh},
    heterogeneity::HeterogeneityModel,
};
pub use plotting::{plot_2d_slice, plot_positions, plot_simulation_outputs, plot_time_series};
pub use recorder::Recorder;
pub use sensor::Sensor;
pub use signal::{chirp::ChirpSignal, sine_wave::SineWave, sweep::SweepSignal, Signal};
pub use solver::{SimulationFields, Solver};
pub use source::{
    apodization::HanningApodization,
    linear_array::LinearArray,
    matrix_array::MatrixArray,
    Source,
};
pub use time::Time;
pub use utils::{derivative, fft_3d, ifft_3d, laplacian};