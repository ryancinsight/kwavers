use crate::physics::optics::sonoluminescence::EmissionParameters;

/// Sonoluminescence coupling configuration
#[derive(Debug, Clone)]
pub struct SonoluminescenceCouplingConfig {
    /// Enable electromagnetic-sonoluminescence coupling
    pub enable_coupling: bool,
    /// Coupling efficiency (fraction of bubble energy converted to light)
    pub coupling_efficiency: f64,
    /// Emission parameters for sonoluminescence
    pub emission_params: EmissionParameters,
    /// Grid shape for emission field [nx, ny, nz]
    pub grid_shape: (usize, usize, usize),
    /// Grid spacing [dx, dy, dz] in meters
    pub grid_spacing: (f64, f64, f64),
    /// Enable spectral resolution
    pub spectral_resolution: bool,
    /// Wavelength range for spectral calculations [min, max] in meters
    pub wavelength_range: (f64, f64),
    /// Number of wavelength bins
    pub n_wavelengths: usize,
}

impl Default for SonoluminescenceCouplingConfig {
    fn default() -> Self {
        Self {
            enable_coupling: true,
            coupling_efficiency: 0.001,
            emission_params: EmissionParameters::default(),
            grid_shape: (50, 50, 50),
            grid_spacing: (2e-4, 2e-4, 2e-4),
            spectral_resolution: true,
            wavelength_range: (200e-9, 1000e-9),
            n_wavelengths: 50,
        }
    }
}

/// Sonoluminescence-electromagnetic coupling problem type
#[derive(Debug, Clone, PartialEq)]
pub enum SonoluminescenceCouplingType {
    /// Static emission: light sources fixed in time
    StaticEmission,
    /// Dynamic emission: time-varying light sources from bubble collapse
    DynamicEmission,
    /// Spectral coupling: full wavelength-dependent emission and propagation
    SpectralCoupling,
}
