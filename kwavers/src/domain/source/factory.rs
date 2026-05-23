//! Source factory for creating acoustic sources
//!
//! Follows Creator pattern for source instantiation using domain types.

use crate::core::error::{ConfigError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::signal::{Signal, SignalWindowType, SineWave, ToneBurst};
use crate::domain::source::{
    basic::{LinearArray, MatrixArray, PistonApodization, PistonConfig},
    transducers::{
        apodization::RectangularApodization,
        focused::{BowlConfig, BowlTransducer, FocusedSource},
    },
    wavefront::{
        BesselConfig, GaussianConfig, PlaneWaveSourceConfig, SphericalConfig, SphericalWaveType,
    },
    BesselSource, DomainSourceParameters, EnvelopeType, GaussianSource, PistonSource,
    PlaneWaveSource, PointSource, PulseType, Source, SourceModel, SphericalSource,
};
use std::f64::consts::PI;
use std::sync::Arc;

use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

/// Factory for creating sources
#[derive(Debug)]
pub struct SourceFactory;

impl SourceFactory {
    /// Create a source from configuration
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn create_source(
        config: &DomainSourceParameters,
        grid: &Grid,
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
        let wavelength = SOUND_SPEED_WATER_SIM / config.frequency;
        let direction = match config.focus {
            Some(f) => (f[0], f[1], f[2]),
            None => (0.0, 0.0, 1.0), // Default z-direction
        };

        match config.model {
            SourceModel::Point => Ok(Box::new(PointSource::new(position, signal))),
            SourceModel::PlaneWave => {
                let wave_config = PlaneWaveSourceConfig {
                    direction,
                    wavelength,
                    phase: config.phase,
                    source_type: config.source_field,
                    injection_mode: crate::domain::source::InjectionMode::default(),
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
                // DomainSourceParameters has 'radius'. For Bessel, legacy used radius as radial_wavenumber if provided, default 1000.
                let radial_wavenumber = if config.radius > 0.0 {
                    config.radius
                } else {
                    1000.0
                };

                // Axial calculation handled by factory or config?
                // Legacy factory calculated it manually.
                let axial_wavenumber =
                    radial_wavenumber.mul_add(-radial_wavenumber, (2.0 * PI / wavelength).powi(2)).sqrt();

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
                // Standard DomainSourceParameters has explicit fields usually, but legacy used delay sign.
                // Here we use delay for Signal delay.
                // But DomainSourceParameters doesn't have WaveType field for Spherical.
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
            SourceModel::LinearArray => {
                let length = config.radius * 2.0;
                let num_elements = config.num_elements.unwrap_or(32);
                let mut array = LinearArray::new(
                    length,
                    num_elements,
                    position,
                    signal,
                    SOUND_SPEED_WATER_SIM,
                    config.frequency,
                    RectangularApodization,
                );
                if let Some(focus) = config.focus {
                    array.adjust_focus(focus[0], focus[1], focus[2], SOUND_SPEED_WATER_SIM);
                }
                Ok(Box::new(array))
            }
            SourceModel::MatrixArray => {
                let width = config.radius * 2.0;
                let height = config.radius * 2.0;
                let n_total = config.num_elements.unwrap_or(256);
                let n_side = (n_total as f64).sqrt().ceil() as usize;
                let mut array = MatrixArray::new(
                    width,
                    height,
                    n_side,
                    n_side,
                    position,
                    signal,
                    SOUND_SPEED_WATER_SIM,
                    config.frequency,
                    RectangularApodization,
                );
                if let Some(focus) = config.focus {
                    array.adjust_focus(focus[0], focus[1], focus[2], SOUND_SPEED_WATER_SIM);
                }
                Ok(Box::new(array))
            }
            SourceModel::Focused => {
                // Determine focus. Default to direction-based.
                let focus = match config.focus {
                    Some(f) => [f[0], f[1], f[2]],
                    None => {
                         // Extrapolate from center + direction * radius
                         [position.0, position.1, position.2 + 0.05]
                    }
                };

                // Approximate radius of curvature as distance to focus
                let r_curv = (focus[2] - position.2).mul_add(focus[2] - position.2, (focus[1] - position.1).mul_add(focus[1] - position.1, (focus[0] - position.0).powi(2)))
                .sqrt();

                // Avoid zero radius
                let r_curv = if r_curv < 1e-6 { 0.05 } else { r_curv };

                let bowl_config = BowlConfig {
                    radius_of_curvature: r_curv,
                    diameter: config.radius * 2.0,
                    center: config.position,
                    focus,
                    frequency: config.frequency,
                    amplitude: config.amplitude,
                    phase: config.phase,
                    element_size: None, // Auto-calculate
                    apply_directivity: true,
                };
                let transducer = match config.num_elements {
                    Some(element_count) => {
                        BowlTransducer::with_element_count(bowl_config, element_count)?
                    }
                    None => BowlTransducer::new(bowl_config)?,
                };
                Ok(Box::new(FocusedSource::from_transducer(
                    transducer, signal, grid,
                )))
            }
            SourceModel::Custom => Err(ConfigError::InvalidValue {
                parameter: "model".to_owned(),
                value: "Custom".to_owned(),
                constraint: "Custom source requires programmatic creation via Builder, not supported via config file.".to_owned(),
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
                    EnvelopeType::Hann | EnvelopeType::Hanning => SignalWindowType::Hann,
                    EnvelopeType::Rectangular => SignalWindowType::Rectangular,
                    EnvelopeType::Gaussian => SignalWindowType::Gaussian,
                    EnvelopeType::Tukey => SignalWindowType::Tukey { alpha: 0.5 },
                    EnvelopeType::Blackman => SignalWindowType::Blackman,
                    EnvelopeType::Hamming => SignalWindowType::Hamming,
                };

                // ToneBurst creation
                Ok(Arc::new(
                    ToneBurst::try_new(frequency, pulse.cycles, delay, amplitude)?
                        .with_window(window)
                        .with_phase(phase),
                ))
            }
            _ => Err(ConfigError::InvalidValue {
                parameter: "pulse_type".to_owned(),
                value: format!("{:?}", pulse.pulse_type),
                constraint: "Pulse type not currently supported by factory".to_owned(),
            }
            .into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::source::{DomainSourceParameters, SourceModel};

    #[test]
    fn focused_source_factory_honors_configured_element_count() {
        let mut grid = Grid::new(24, 24, 24, 0.004, 0.004, 0.004).unwrap();
        grid.origin = [-0.048, -0.048, -0.008];
        let element_count = 17;
        let config = DomainSourceParameters {
            model: SourceModel::Focused,
            position: [0.0, 0.0, 0.0],
            focus: Some([0.0, 0.0, 0.08]),
            radius: 0.02,
            frequency: 650.0e3,
            num_elements: Some(element_count),
            ..Default::default()
        };

        let source = SourceFactory::create_source(&config, &grid).unwrap();

        assert_eq!(source.positions().len(), element_count);
        assert_eq!(source.focal_point(), Some((0.0, 0.0, 0.08)));
    }
}
