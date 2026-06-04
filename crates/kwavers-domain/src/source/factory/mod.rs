//! Source factory for creating acoustic sources.
//!
//! Follows Creator pattern for source instantiation using domain types.

use kwavers_core::constants::numerical::TWO_PI;
mod focused;
mod signal;

#[cfg(test)]
mod tests;

use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::{ConfigError, KwaversResult};
use kwavers_grid::Grid;
use crate::source::{
    basic::{LinearArray, MatrixArray, PistonApodization, PistonConfig},
    transducers::{apodization::RectangularApodization, focused::FocusedSource},
    wavefront::{
        BesselConfig, GaussianConfig, PlaneWaveSourceConfig, SphericalConfig, SphericalWaveType,
    },
    BesselSource, DomainSourceParameters, GaussianSource, PistonSource, PlaneWaveSource,
    PointSource, Source, SourceModel, SphericalSource,
};

/// Factory for creating sources.
#[derive(Debug)]
pub struct SourceFactory;

impl SourceFactory {
    /// Create a source from configuration.
    ///
    /// # Errors
    /// Propagates any [`KwaversError`] returned by called functions.
    pub fn create_source(
        config: &DomainSourceParameters,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Source>> {
        config.validate()?;

        let signal_arc = signal::create_signal(
            &config.pulse,
            config.frequency,
            config.amplitude,
            config.phase,
            config.delay,
        )?;

        let position = (config.position[0], config.position[1], config.position[2]);
        let wavelength = SOUND_SPEED_WATER_SIM / config.frequency;
        let direction = match config.focus {
            Some(f) => (f[0], f[1], f[2]),
            None => (0.0, 0.0, 1.0),
        };

        match config.model {
            SourceModel::Point => Ok(Box::new(PointSource::new(position, signal_arc))),
            SourceModel::PlaneWave => {
                let wave_config = PlaneWaveSourceConfig {
                    direction,
                    wavelength,
                    phase: config.phase,
                    source_type: config.source_field,
                    injection_mode: crate::source::InjectionMode::default(),
                };
                Ok(Box::new(PlaneWaveSource::new(wave_config, signal_arc)))
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
                Ok(Box::new(GaussianSource::new(gauss_config, signal_arc)))
            }
            SourceModel::Bessel => {
                let radial_wavenumber = if config.radius > 0.0 {
                    config.radius
                } else {
                    1000.0
                };
                let axial_wavenumber = radial_wavenumber
                    .mul_add(-radial_wavenumber, (TWO_PI / wavelength).powi(2))
                    .sqrt();

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
                Ok(Box::new(BesselSource::new(bessel_config, signal_arc)))
            }
            SourceModel::Spherical => {
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
                Ok(Box::new(SphericalSource::new(spherical_config, signal_arc)))
            }
            SourceModel::Piston => {
                let piston_config = PistonConfig {
                    center: position,
                    diameter: config.radius * 2.0,
                    normal: direction,
                    source_type: config.source_field,
                    apodization: PistonApodization::Uniform,
                };
                Ok(Box::new(PistonSource::new(piston_config, signal_arc)))
            }
            SourceModel::LinearArray => {
                let length = config.radius * 2.0;
                let num_elements = config.num_elements.unwrap_or(32);
                let mut array = LinearArray::new(
                    length,
                    num_elements,
                    position,
                    signal_arc,
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
                    signal_arc,
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
                let transducer = focused::create_focused_bowl_transducer(config)?;
                Ok(Box::new(FocusedSource::from_transducer(
                    transducer,
                    signal_arc,
                    grid,
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
}
