//! Source factory for creating acoustic sources
//!
//! Follows Creator pattern for source instantiation using domain types.

use crate::core::error::{ConfigError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::signal::{Signal, SineWave, ToneBurst, WindowType};
use crate::domain::source::{
    basic::{PistonApodization, PistonConfig},
    wavefront::{
        BesselConfig, GaussianConfig, PlaneWaveConfig, SphericalConfig, SphericalWaveType,
    },
    BesselSource, EnvelopeType, GaussianSource, PistonSource, PlaneWaveSource, PointSource,
    PulseType, Source, SourceModel, SourceParameters, SphericalSource,
};
use std::f64::consts::PI;
use std::sync::Arc;

const DEFAULT_SOUND_SPEED: f64 = 1500.0; // m/s (Water)

/// Factory for creating sources
#[derive(Debug)]
pub struct SourceFactory;

impl SourceFactory {
    /// Create a source from configuration
    pub fn create_source(
        config: &SourceParameters,
        _grid: &Grid,
    ) -> KwaversResult<Box<dyn Source>> {
        config.validate()?;

        // Create signal
        let signal = Self::create_signal(
            &config.pulse,
            config.frequency,
            config.amplitude,
            config.phase,
            config.delay,
        )?;

        // Common conversions
        let position = (config.position[0], config.position[1], config.position[2]);
        let wavelength = DEFAULT_SOUND_SPEED / config.frequency;
        let direction = match config.focus {
            Some(f) => (f[0], f[1], f[2]),
            None => (0.0, 0.0, 1.0), // Default z-direction
        };

        match config.model {
            SourceModel::Point => Ok(Box::new(PointSource::new(position, signal))),
            SourceModel::PlaneWave => {
                let wave_config = PlaneWaveConfig {
                    direction,
                    wavelength,
                    phase: config.phase,
                    source_type: config.source_field,
                };
                Ok(Box::new(PlaneWaveSource::new(wave_config, signal)))
            }
            SourceModel::Gaussian => {
                let gauss_config = GaussianConfig {
                    focal_point: position,
                    waist_radius: config.radius,
                    wavelength,
                    direction,
                    source_type: config.source_field,
                    phase: config.phase,
                };
                Ok(Box::new(GaussianSource::new(gauss_config, signal)))
            }
            SourceModel::Bessel => {
                // Legacy: radial_wavenumber was passed as radius?
                // SourceParameters has 'radius'. For Bessel, legacy used radius as radial_wavenumber if provided, default 1000.
                let radial_wavenumber = if config.radius > 0.0 {
                    config.radius
                } else {
                    1000.0
                };

                // Axial calculation handled by factory or config?
                // Legacy factory calculated it manually.
                let axial_wavenumber =
                    ((2.0 * PI / wavelength).powi(2) - radial_wavenumber.powi(2)).sqrt();

                let bessel_config = BesselConfig {
                    center: position,
                    direction,
                    wavelength,
                    radial_wavenumber,
                    axial_wavenumber,
                    order: 0,
                    source_type: config.source_field,
                    phase: config.phase,
                };
                Ok(Box::new(BesselSource::new(bessel_config, signal)))
            }
            SourceModel::Spherical => {
                // Determine wave type from delay? Legacy: delay < 0 => Converging.
                // Standard SourceParameters has explicit fields usually, but legacy used delay sign.
                // Here we use delay for Signal delay.
                // But SourceParameters doesn't have WaveType field for Spherical.
                // We default to Diverging unless we add a field.
                // Legacy check:
                // let wave_type = if config.delay < 0.0 { Converging } else { Diverging };
                let wave_type = if config.delay < 0.0 {
                    SphericalWaveType::Converging
                } else {
                    SphericalWaveType::Diverging
                };

                let spherical_config = SphericalConfig {
                    center: position,
                    wavelength,
                    wave_type,
                    source_type: config.source_field,
                    phase: config.phase,
                    attenuation: 0.0,
                };
                Ok(Box::new(SphericalSource::new(spherical_config, signal)))
            }
            SourceModel::Piston => {
                let piston_config = PistonConfig {
                    center: position,
                    diameter: config.radius * 2.0,
                    normal: direction,
                    source_type: config.source_field,
                    apodization: PistonApodization::Uniform,
                };
                Ok(Box::new(PistonSource::new(piston_config, signal)))
            }
            // TODO: INCOMPLETE IMPLEMENTATION - Missing Source Models
            // The following source models are not yet implemented:
            //
            // 1. LinearArray: Linear array of transducer elements
            //    - Requires: Element positions, element size, steering angles
            //    - Physics: Array factor calculation, element directivity
            //    - Estimated effort: 8-10 hours
            //
            // 2. MatrixArray: 2D matrix array of elements
            //    - Requires: Element grid layout, steering in 2D
            //    - Physics: 2D array factor, elevation/azimuth control
            //    - Estimated effort: 10-12 hours
            //
            // 3. Focused: Focused transducer with geometric/electronic focusing
            //    - Requires: Focal point, F-number, aperture
            //    - Physics: Rayleigh-Sommerfeld diffraction, focal gain
            //    - Estimated effort: 6-8 hours
            //
            // 4. Custom: User-defined source pattern
            //    - Requires: Custom field calculation callback/function
            //    - Architecture: Trait-based extension point
            //    - Estimated effort: 4-6 hours
            //
            // See backlog.md item #7 for full specifications
            // Total effort: 28-36 hours for complete implementation
            _ => Err(ConfigError::InvalidValue {
                parameter: "model".to_string(),
                value: format!("{:?}", config.model),
                constraint: "Source model not currently supported by factory - TODO: Implement LinearArray, MatrixArray, Focused, Custom".to_string(),
            }
            .into()),
        }
    }

    fn create_signal(
        pulse: &crate::domain::source::config::PulseParameters,
        frequency: f64,
        amplitude: f64,
        phase: f64,
        delay: f64,
    ) -> KwaversResult<Arc<dyn Signal>> {
        match pulse.pulse_type {
            PulseType::ContinuousWave | PulseType::Sine => {
                Ok(Arc::new(SineWave::new(frequency, amplitude, phase)))
            }
            PulseType::ToneBurst => {
                let window = match pulse.envelope {
                    EnvelopeType::Hann | EnvelopeType::Hanning => WindowType::Hann,
                    EnvelopeType::Rectangular => WindowType::Rectangular,
                    EnvelopeType::Gaussian => WindowType::Gaussian,
                    EnvelopeType::Tukey => WindowType::Tukey { alpha: 0.5 },
                };

                // ToneBurst creation
                Ok(Arc::new(
                    ToneBurst::try_new(frequency, pulse.cycles, delay, amplitude)?
                        .with_window(window)
                        .with_phase(phase),
                ))
            }
            _ => Err(ConfigError::InvalidValue {
                parameter: "pulse_type".to_string(),
                value: format!("{:?}", pulse.pulse_type),
                constraint: "Pulse type not currently supported by factory".to_string(),
            }
            .into()),
        }
    }
}
