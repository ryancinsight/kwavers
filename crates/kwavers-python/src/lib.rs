//! # pykwavers: Python Bindings for kwavers
//!
//! This module provides Python bindings for the kwavers ultrasound simulation library
//! with an API compatible with k-Wave/k-wave-python for direct comparison and validation.
//!
//! ## Architecture
//!
//! Following Clean Architecture principles:
//! - **Presentation Layer**: Python API (this crate)
//! - **Domain Layer**: Core kwavers library
//! - **Dependency Direction**: Python → Rust (unidirectional)
//!
//! ## API Design
//!
//! The API mirrors k-Wave's structure for ease of comparison:
//! ```python
//! import pykwavers as kw
//!
//! # Create grid (similar to kWaveGrid)
//! grid = kw.Grid(nx=128, ny=128, nz=128, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
//!
//! # Define medium (similar to k-Wave medium struct)
//! medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
//!
//! # Create source
//! source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
//!
//! # Create sensor
//! sensor = kw.Sensor.point(position=[0.01, 0.01, 0.01])
//!
//! # Run simulation
//! sim = kw.Simulation(grid, medium, source, sensor)
//! result = sim.run(time_steps=1000, dt=1e-8)
//! ```
//!
//! ## Mathematical Specifications
//!
//! - Grid: Uniform Cartesian grid with spacing dx, dy, dz
//! - Medium: Acoustic properties (c, ρ, α, nonlinearity)
//! - Source: Pressure/velocity boundary conditions
//! - Sensor: Point/grid sampling with interpolation
//! - Simulation: FDTD/PSTD time-stepping with CPML boundaries
//!
//! ## References
//!
//! 1. Treeby & Cox (2010). "k-Wave: MATLAB toolbox for simulation and
//!    reconstruction of photoacoustic wave fields." J. Biomed. Opt. 15(2).
//! 2. kwavers architecture documentation (../kwavers/ARCHITECTURE.md)
//!
//! Author: Ryan Clanton (@ryancinsight)
//! Date: 2026-02-04
//! Sprint: 217 Session 9 - Python Integration via PyO3

use pyo3::prelude::*;

mod analytical_bindings;
mod breast_fwi_bindings;
mod pam_bindings;

// ============================================================================
// Utility Function Bindings
// ============================================================================

mod bubble_bindings;
mod fft_bindings;
mod field_surrogate_bindings;
mod imaging_bindings;
mod ritk_image;
mod sonogenetics_bindings;
mod theranostic_bindings;
mod thermal_bindings;
mod utils_bindings;

// ============================================================================
// Error Handling
// ============================================================================

/// Convert k-Wave absorption units dB/(MHz^y·cm) to Np/m at the given frequency.
///
/// This follows the standard scalar conversion
/// `alpha_np_m = alpha_db_cm * f_mhz^y * 100 / (20 / ln(10))`.
/// The helper remains local for the frequency-domain utility tests; the GPU
/// PSTD path uses the shared kwavers spectral conversion helper.
#[allow(dead_code)]
fn alpha_db_cm_to_np_m(alpha_db_cm: f64, frequency_mhz: f64, alpha_power: f64) -> f64 {
    let db_to_np = 20.0 / std::f64::consts::LN_10;
    alpha_db_cm * frequency_mhz.powf(alpha_power) * 100.0 / db_to_np
}

#[cfg(test)]
mod physics_unit_tests {
    use super::alpha_db_cm_to_np_m;

    #[test]
    fn test_alpha_db_cm_to_np_m_matches_scalar_reference_at_1mhz() {
        let alpha_db_cm = 0.75;
        let got = alpha_db_cm_to_np_m(alpha_db_cm, 1.0, 1.5);
        let expected = alpha_db_cm * 100.0 / (20.0 / std::f64::consts::LN_10);
        assert!(
            (got - expected).abs() < 1e-12,
            "conversion mismatch: got {got}, expected {expected}"
        );
    }

    #[test]
    fn test_alpha_db_cm_to_np_m_respects_power_law_frequency_scaling() {
        let alpha_db_cm = 0.5;
        let at_1mhz = alpha_db_cm_to_np_m(alpha_db_cm, 1.0, 1.5);
        let at_2mhz = alpha_db_cm_to_np_m(alpha_db_cm, 2.0, 1.5);
        let expected_ratio = 2.0_f64.powf(1.5);
        let got_ratio = at_2mhz / at_1mhz;
        assert!(
            (got_ratio - expected_ratio).abs() < 1e-12,
            "power-law scaling mismatch: got {got_ratio}, expected {expected_ratio}"
        );
    }
}

// ============================================================================
// Solver Type Enum
// ============================================================================

mod solver_type_bindings;
pub use solver_type_bindings::SolverType;

// ============================================================================
// Grid: Computational Domain
// ============================================================================

mod grid_py;
pub use grid_py::Grid;

// ============================================================================
// Medium: Acoustic Properties
// ============================================================================

mod medium_py;
pub use medium_py::Medium;

// ============================================================================
// Source: Acoustic Excitation
// ============================================================================

mod source_py;
pub use source_py::Source;

// ============================================================================
// KWaveArray: Custom Transducer Geometry
// ============================================================================

mod kwave_array_py;
pub use kwave_array_py::KWaveArray;

// ============================================================================
// TransducerArray2D: 2D Transducer Array Source
// ============================================================================

mod transducer_array_py;
pub use transducer_array_py::TransducerArray2D;

// ============================================================================
// Sensor: Field Sampling
// ============================================================================

mod sensor_py;
pub use sensor_py::Sensor;

// ============================================================================
// Simulation: Main Interface
// ============================================================================

mod simulation_py;
pub use simulation_py::{GpuPstdSession, Simulation};

// ============================================================================
// Simulation Result + internal run-result bundle
// ============================================================================

mod simulation_result_py;
pub(crate) use simulation_result_py::SimulationResult;

// ============================================================================
// Config Builders: PmlConfig, HelmholtzConfig, NonlinearConfig, ThermalConfig
// ============================================================================

mod config_builders;
pub use config_builders::{HelmholtzConfig, NonlinearConfig, PmlConfig, ThermalConfig};

// ============================================================================
// Module Definition
// ============================================================================

/// pykwavers: Python bindings for kwavers ultrasound simulation library.
///
/// This module provides a k-Wave-compatible API for acoustic wave simulation.
#[pymodule]
fn _pykwavers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Grid>()?;
    m.add_class::<Medium>()?;
    m.add_class::<Source>()?;
    m.add_class::<KWaveArray>()?;
    m.add_class::<TransducerArray2D>()?;
    m.add_class::<Sensor>()?;
    m.add_class::<Simulation>()?;
    m.add_class::<SimulationResult>()?;
    m.add_class::<SolverType>()?;
    m.add_class::<GpuPstdSession>()?;
    m.add_class::<PmlConfig>()?;
    m.add_class::<HelmholtzConfig>()?;
    m.add_class::<NonlinearConfig>()?;
    m.add_class::<ThermalConfig>()?;

    // Phase 22 bindings
    misc_bindings::register(m)?;

    // Register utility functions
    pam_bindings::register_pam(m)?;
    utils_bindings::register_utils(m)?;
    thermal_bindings::register_thermal(m)?;
    field_surrogate_bindings::register(m)?;
    imaging_bindings::register(m)?;
    theranostic_bindings::register(m)?;
    breast_fwi_bindings::register(m)?;
    fft_bindings::register(m)?;
    analytical_bindings::register_book(m)?;
    bubble_bindings::register_bubble(m)?;
    sonogenetics_bindings::register_sonogenetics(m)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Ryan Clanton PhD")?;

    Ok(())
}

// ============================================================================
// Phase 22 Wrappers: PID Controller, Registration, and Bubble Field
// ============================================================================

mod misc_bindings;
