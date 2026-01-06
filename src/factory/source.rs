//! Source factory for creating acoustic sources
//!
//! Follows Creator pattern for source instantiation

use crate::error::{ConfigError, KwaversResult};
use crate::grid::Grid;
use crate::physics::constants::{
    DEFAULT_ULTRASOUND_FREQUENCY, STANDARD_BEAM_WIDTH, STANDARD_PRESSURE_AMPLITUDE,
};
use crate::signal::{Signal, SineWave, ToneBurst, WindowType};
use crate::source::{PointSource, Source};
use std::sync::Arc;

/// Source configuration
#[derive(Debug, Clone)]
pub struct SourceConfig {
    pub source_type: String,
    pub position: (f64, f64, f64),
    pub amplitude: f64,
    pub frequency: f64,
    pub radius: Option<f64>,
    pub focus: Option<(f64, f64, f64)>,
    pub num_elements: Option<usize>,
    pub signal_type: String,
    pub phase: f64,
    pub delay: f64,
    pub cycles: Option<f64>,
    pub envelope: Option<String>,
}

impl SourceConfig {
    /// Validate source configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.amplitude <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "amplitude".to_string(),
                value: self.amplitude.to_string(),
                constraint: "Amplitude must be positive".to_string(),
            }
            .into());
        }

        if self.frequency <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "frequency".to_string(),
                value: self.frequency.to_string(),
                constraint: "Frequency must be positive".to_string(),
            }
            .into());
        }

        if self.delay < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "delay".to_string(),
                value: self.delay.to_string(),
                constraint: "Delay must be non-negative".to_string(),
            }
            .into());
        }

        if let Some(radius) = self.radius {
            if radius <= 0.0 {
                return Err(ConfigError::InvalidValue {
                    parameter: "radius".to_string(),
                    value: radius.to_string(),
                    constraint: "Radius must be positive".to_string(),
                }
                .into());
            }
        }

        match self.signal_type.as_str() {
            "sine" => {}
            "tone_burst" => {
                let cycles = self.cycles.ok_or_else(|| ConfigError::MissingParameter {
                    parameter: "cycles".to_string(),
                    section: "source".to_string(),
                })?;
                if cycles <= 0.0 {
                    return Err(ConfigError::InvalidValue {
                        parameter: "cycles".to_string(),
                        value: cycles.to_string(),
                        constraint: "Cycles must be positive".to_string(),
                    }
                    .into());
                }
            }
            _ => {
                return Err(ConfigError::InvalidValue {
                    parameter: "signal_type".to_string(),
                    value: self.signal_type.clone(),
                    constraint: "Unsupported signal_type (supported: sine, tone_burst)".to_string(),
                }
                .into());
            }
        }

        Ok(())
    }
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            source_type: "point".to_string(),
            position: (0.0, 0.0, 0.0),
            amplitude: STANDARD_PRESSURE_AMPLITUDE,
            frequency: DEFAULT_ULTRASOUND_FREQUENCY,
            radius: Some(STANDARD_BEAM_WIDTH),
            focus: None,
            num_elements: None,
            signal_type: "sine".to_string(),
            phase: 0.0,
            delay: 0.0,
            cycles: None,
            envelope: None,
        }
    }
}

/// Factory for creating sources
#[derive(Debug)]
pub struct SourceFactory;

impl SourceFactory {
    /// Create a source from configuration
    pub fn create_source(config: &SourceConfig, _grid: &Grid) -> KwaversResult<Box<dyn Source>> {
        config.validate()?;

        let signal: Arc<dyn Signal> = match config.signal_type.as_str() {
            "sine" => Arc::new(SineWave::new(
                config.frequency,
                config.amplitude,
                config.phase,
            )),
            "tone_burst" => {
                let cycles = config.cycles.ok_or_else(|| ConfigError::MissingParameter {
                    parameter: "cycles".to_string(),
                    section: "source".to_string(),
                })?;
                let window = match config.envelope.as_deref() {
                    None | Some("hann") => WindowType::Hann,
                    Some("rectangular") => WindowType::Rectangular,
                    Some("gaussian") => WindowType::Gaussian,
                    Some("tukey") => WindowType::Tukey { alpha: 0.5 },
                    Some(other) => {
                        return Err(ConfigError::InvalidValue {
                            parameter: "envelope".to_string(),
                            value: other.to_string(),
                            constraint: "Unsupported envelope (supported: hann, rectangular, gaussian, tukey)".to_string(),
                        }
                        .into());
                    }
                };

                Arc::new(
                    ToneBurst::try_new(config.frequency, cycles, config.delay, config.amplitude)?
                        .with_window(window)
                        .with_phase(config.phase),
                )
            }
            other => {
                return Err(ConfigError::InvalidValue {
                    parameter: "signal_type".to_string(),
                    value: other.to_string(),
                    constraint: "Unsupported signal_type (supported: sine, tone_burst)".to_string(),
                }
                .into());
            }
        };

        // Create appropriate source type based on config
        match config.source_type.as_str() {
            "point" => Ok(Box::new(PointSource::new(config.position, signal))),
            "plane_wave" => {
                let direction = match config.focus {
                    Some((dx, dy, dz)) => (dx, dy, dz),
                    None => (1.0, 0.0, 0.0), // Default direction
                };
                let wavelength = 1500.0 / config.frequency; // Default sound speed for water
                let config = crate::source::wavefront::PlaneWaveConfig {
                    direction,
                    wavelength,
                    phase: config.phase,
                    source_type: crate::source::SourceField::Pressure,
                };
                Ok(Box::new(crate::source::wavefront::PlaneWaveSource::new(
                    config, signal,
                )))
            }
            "piston" => {
                let radius = config.radius.unwrap_or(5.0e-3); // Default 5mm radius
                let config = crate::source::basic::PistonConfig {
                    center: config.position,
                    diameter: radius * 2.0,
                    normal: (0.0, 0.0, 1.0), // Default normal direction
                    source_type: crate::source::SourceField::Pressure,
                    apodization: crate::source::basic::PistonApodization::Uniform,
                };
                Ok(Box::new(crate::source::basic::PistonSource::new(
                    config, signal,
                )))
            }
            "gaussian" => {
                let wavelength = 1500.0 / config.frequency; // Default sound speed for water
                let config = crate::source::wavefront::GaussianConfig {
                    focal_point: config.position,
                    waist_radius: config.radius.unwrap_or(1.0e-3), // Default 1mm waist
                    wavelength,
                    direction: (0.0, 0.0, 1.0), // Default direction
                    source_type: crate::source::SourceField::Pressure,
                    phase: config.phase,
                };
                Ok(Box::new(crate::source::wavefront::GaussianSource::new(
                    config, signal,
                )))
            }
            "bessel" => {
                let wavelength = 1500.0 / config.frequency; // Default sound speed for water
                let radial_wavenumber = config.radius.unwrap_or(1000.0); // Default k_r
                let axial_wavenumber = ( (2.0 * std::f64::consts::PI / wavelength).powi(2) - radial_wavenumber.powi(2) ).sqrt();
                let config = crate::source::wavefront::BesselConfig {
                    center: config.position,
                    direction: (0.0, 0.0, 1.0), // Default direction
                    wavelength,
                    radial_wavenumber,
                    axial_wavenumber,
                    order: 0, // Zeroth-order by default
                    source_type: crate::source::SourceField::Pressure,
                    phase: config.phase,
                };
                Ok(Box::new(crate::source::wavefront::BesselSource::new(
                    config, signal,
                )))
            }
            "spherical" => {
                let wavelength = 1500.0 / config.frequency; // Default sound speed for water
                let wave_type = if config.delay < 0.0 {
                    crate::source::wavefront::SphericalWaveType::Converging
                } else {
                    crate::source::wavefront::SphericalWaveType::Diverging
                };
                let config = crate::source::wavefront::SphericalConfig {
                    center: config.position,
                    wavelength,
                    wave_type,
                    source_type: crate::source::SourceField::Pressure,
                    phase: config.phase,
                    attenuation: 0.0, // No attenuation by default
                };
                Ok(Box::new(crate::source::wavefront::SphericalSource::new(
                    config, signal,
                )))
            }
            other => Err(ConfigError::InvalidValue {
                parameter: "source_type".to_string(),
                value: other.to_string(),
                constraint: "Unsupported source_type (supported: point, plane_wave, piston, gaussian, bessel, spherical)"
                    .to_string(),
            }
            .into()),
        }
    }

    /// Create a point source at specified location
    #[must_use]
    pub fn create_point_source(
        x: f64,
        y: f64,
        z: f64,
        amplitude: f64,
        frequency: f64,
    ) -> Box<dyn Source> {
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(frequency, amplitude, 0.0));
        Box::new(PointSource::new((x, y, z), signal))
    }

    /// Create a plane wave source
    #[must_use]
    pub fn create_plane_wave_source(
        direction: (f64, f64, f64),
        wavelength: f64,
        amplitude: f64,
        frequency: f64,
    ) -> Box<dyn Source> {
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(frequency, amplitude, 0.0));
        let config = crate::source::wavefront::PlaneWaveConfig {
            direction,
            wavelength,
            phase: 0.0,
            source_type: crate::source::SourceField::Pressure,
        };
        Box::new(crate::source::wavefront::PlaneWaveSource::new(
            config, signal,
        ))
    }

    /// Create a piston source
    #[must_use]
    pub fn create_piston_source(
        center: (f64, f64, f64),
        diameter: f64,
        amplitude: f64,
        frequency: f64,
    ) -> Box<dyn Source> {
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(frequency, amplitude, 0.0));
        let config = crate::source::basic::PistonConfig {
            center,
            diameter,
            normal: (0.0, 0.0, 1.0),
            source_type: crate::source::SourceField::Pressure,
            apodization: crate::source::basic::PistonApodization::Uniform,
        };
        Box::new(crate::source::basic::PistonSource::new(config, signal))
    }

    /// Create a linear array source
    #[must_use]
    pub fn create_linear_array_source(
        length: f64,
        num_elements: usize,
        y_pos: f64,
        z_pos: f64,
        amplitude: f64,
        frequency: f64,
    ) -> Box<dyn Source> {
        let signal: Box<dyn Signal> = Box::new(SineWave::new(frequency, amplitude, 0.0));
        // Note: This is a simplified version - in practice, you'd need medium and grid
        // For factory convenience, we create a basic linear array
        Box::new(crate::source::basic::LinearArray::new(
            length,
            num_elements,
            y_pos,
            z_pos,
            signal,
            // Placeholder values - would need proper medium/grid in real usage
            &crate::medium::homogeneous::HomogeneousMedium::new(
                1500.0,
                1000.0,
                0.0,
                0.0,
                &crate::grid::Grid::new(1, 1, 1, 1.0, 1.0, 1.0).unwrap(),
            ),
            &crate::grid::Grid::new(1, 1, 1, 1.0, 1.0, 1.0).unwrap(),
            frequency,
            crate::source::transducers::apodization::RectangularApodization,
        ))
    }

    /// Create a Gaussian beam source
    #[must_use]
    pub fn create_gaussian_source(
        focal_point: (f64, f64, f64),
        waist_radius: f64,
        wavelength: f64,
        amplitude: f64,
        frequency: f64,
    ) -> Box<dyn Source> {
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(frequency, amplitude, 0.0));
        let config = crate::source::wavefront::GaussianConfig {
            focal_point,
            waist_radius,
            wavelength,
            direction: (0.0, 0.0, 1.0),
            source_type: crate::source::SourceField::Pressure,
            phase: 0.0,
        };
        Box::new(crate::source::wavefront::GaussianSource::new(
            config, signal,
        ))
    }

    /// Create a Bessel beam source
    #[must_use]
    pub fn create_bessel_source(
        center: (f64, f64, f64),
        wavelength: f64,
        radial_wavenumber: f64,
        amplitude: f64,
        frequency: f64,
    ) -> Box<dyn Source> {
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(frequency, amplitude, 0.0));
        let axial_wavenumber =
            ((2.0 * std::f64::consts::PI / wavelength).powi(2) - radial_wavenumber.powi(2)).sqrt();
        let config = crate::source::wavefront::BesselConfig {
            center,
            direction: (0.0, 0.0, 1.0),
            wavelength,
            radial_wavenumber,
            axial_wavenumber,
            order: 0,
            source_type: crate::source::SourceField::Pressure,
            phase: 0.0,
        };
        Box::new(crate::source::wavefront::BesselSource::new(config, signal))
    }

    /// Create a spherical wave source
    #[must_use]
    pub fn create_spherical_source(
        center: (f64, f64, f64),
        wavelength: f64,
        wave_type: crate::source::wavefront::SphericalWaveType,
        amplitude: f64,
        frequency: f64,
    ) -> Box<dyn Source> {
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(frequency, amplitude, 0.0));
        let config = crate::source::wavefront::SphericalConfig {
            center,
            wavelength,
            wave_type,
            source_type: crate::source::SourceField::Pressure,
            phase: 0.0,
            attenuation: 0.0,
        };
        Box::new(crate::source::wavefront::SphericalSource::new(
            config, signal,
        ))
    }
}
