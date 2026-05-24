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
        focused::{BowlAngularBounds, BowlConfig, BowlTransducer, FocusedSource},
    },
    wavefront::{
        BesselConfig, GaussianConfig, PlaneWaveSourceConfig, SphericalConfig, SphericalWaveType,
    },
    BesselSource, DomainSourceParameters, EnvelopeType, FocusedBowlAperture, GaussianSource,
    PistonSource, PlaneWaveSource, PointSource, PulseType, Source, SourceModel, SphericalSource,
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
                let transducer = Self::create_focused_bowl_transducer(config, bowl_config)?;
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

    fn create_focused_bowl_transducer(
        config: &DomainSourceParameters,
        bowl_config: BowlConfig,
    ) -> KwaversResult<BowlTransducer> {
        match config.focused_bowl_aperture {
            FocusedBowlAperture::Diameter => match config.num_elements {
                Some(element_count) => {
                    BowlTransducer::with_element_count(bowl_config, element_count)
                }
                None => BowlTransducer::new(bowl_config),
            },
            FocusedBowlAperture::Hemisphere => {
                let element_count = required_focused_bowl_element_count(config)?;
                BowlTransducer::with_angular_bounds(
                    bowl_config,
                    BowlAngularBounds::hemisphere(),
                    element_count,
                )
            }
            FocusedBowlAperture::PolarSpan { theta_max_rad } => {
                let element_count = required_focused_bowl_element_count(config)?;
                BowlTransducer::with_polar_span(bowl_config, theta_max_rad, element_count)
            }
            FocusedBowlAperture::PolarBounds {
                theta_min_rad,
                theta_max_rad,
            } => {
                let element_count = required_focused_bowl_element_count(config)?;
                BowlTransducer::with_polar_bounds(
                    bowl_config,
                    theta_min_rad,
                    theta_max_rad,
                    element_count,
                )
            }
            FocusedBowlAperture::AxisProjectionBounds {
                axis_projection_min,
                axis_projection_max,
            } => {
                let element_count = required_focused_bowl_element_count(config)?;
                BowlTransducer::with_axis_projection_bounds(
                    bowl_config,
                    axis_projection_min,
                    axis_projection_max,
                    element_count,
                )
            }
            FocusedBowlAperture::AxisReferencePolarBounds {
                radius_of_curvature_m,
                theta_min_rad,
                theta_max_rad,
            } => {
                let element_count = required_focused_bowl_element_count(config)?;
                let mut axis_config = BowlConfig::from_axis_reference_focus(
                    config.position,
                    bowl_config.focus,
                    radius_of_curvature_m,
                    config.frequency,
                    config.amplitude,
                )?;
                axis_config.phase = config.phase;
                BowlTransducer::with_polar_bounds(
                    axis_config,
                    theta_min_rad,
                    theta_max_rad,
                    element_count,
                )
            }
            FocusedBowlAperture::AxisReferenceHemisphere {
                radius_of_curvature_m,
            } => {
                let element_count = required_focused_bowl_element_count(config)?;
                let mut axis_config = BowlConfig::from_axis_reference_focus(
                    config.position,
                    bowl_config.focus,
                    radius_of_curvature_m,
                    config.frequency,
                    config.amplitude,
                )?;
                axis_config.phase = config.phase;
                BowlTransducer::with_angular_bounds(
                    axis_config,
                    BowlAngularBounds::hemisphere(),
                    element_count,
                )
            }
        }
    }
}

fn required_focused_bowl_element_count(config: &DomainSourceParameters) -> KwaversResult<usize> {
    config.num_elements.ok_or_else(|| {
        ConfigError::InvalidValue {
            parameter: "num_elements".to_owned(),
            value: "None".to_owned(),
            constraint: "Focused bowl angular aperture modes require a configured element count"
                .to_owned(),
        }
        .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::source::{DomainSourceParameters, FocusedBowlAperture, SourceModel};

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

    #[test]
    fn focused_source_factory_accepts_axis_projection_aperture() {
        let mut grid = Grid::new(40, 40, 28, 0.01, 0.01, 0.01).unwrap();
        grid.origin = [-0.20, -0.20, -0.08];
        let element_count = 19;
        let config = DomainSourceParameters {
            model: SourceModel::Focused,
            position: [0.0, 0.0, 0.16],
            focus: Some([0.0, 0.0, 0.0]),
            radius: 0.16,
            frequency: 650.0e3,
            num_elements: Some(element_count),
            focused_bowl_aperture: FocusedBowlAperture::AxisProjectionBounds {
                axis_projection_min: -0.20,
                axis_projection_max: 0.95,
            },
            ..Default::default()
        };

        let source = SourceFactory::create_source(&config, &grid).unwrap();
        let positions = source.positions();
        let min_projection = positions
            .iter()
            .map(|position| position.2 / 0.16)
            .fold(f64::INFINITY, f64::min);
        let max_projection = positions
            .iter()
            .map(|position| position.2 / 0.16)
            .fold(f64::NEG_INFINITY, f64::max);

        assert_eq!(positions.len(), element_count);
        assert!((min_projection + 0.20).abs() < 0.08);
        assert!((max_projection - 0.95).abs() < 0.08);
    }

    #[test]
    fn focused_source_factory_accepts_hemisphere_aperture() {
        let mut grid = Grid::new(48, 48, 32, 0.01, 0.01, 0.01).unwrap();
        grid.origin = [-0.24, -0.24, -0.04];
        let element_count = 31;
        let radius = 0.16_f64;
        let config = DomainSourceParameters {
            model: SourceModel::Focused,
            position: [0.0, 0.0, radius],
            focus: Some([0.0, 0.0, 0.0]),
            radius: 0.01,
            frequency: 650.0e3,
            num_elements: Some(element_count),
            focused_bowl_aperture: FocusedBowlAperture::Hemisphere,
            ..Default::default()
        };

        let source = SourceFactory::create_source(&config, &grid).unwrap();
        let positions = source.positions();
        let min_projection = positions
            .iter()
            .map(|position| position.2 / radius)
            .fold(f64::INFINITY, f64::min);
        let max_projection = positions
            .iter()
            .map(|position| position.2 / radius)
            .fold(f64::NEG_INFINITY, f64::max);

        assert_eq!(positions.len(), element_count);
        assert!(min_projection >= -1.0e-12);
        assert!(max_projection <= 1.0 + 1.0e-12);
    }

    #[test]
    fn angular_focused_source_factory_requires_element_count() {
        let grid = Grid::new(8, 8, 8, 0.01, 0.01, 0.01).unwrap();
        let config = DomainSourceParameters {
            model: SourceModel::Focused,
            position: [0.0, 0.0, 0.08],
            focus: Some([0.0, 0.0, 0.0]),
            radius: 0.08,
            frequency: 650.0e3,
            focused_bowl_aperture: FocusedBowlAperture::PolarSpan { theta_max_rad: 1.0 },
            ..Default::default()
        };

        let error = SourceFactory::create_source(&config, &grid).unwrap_err();
        assert!(
            format!("{error:?}").contains("num_elements"),
            "expected num_elements validation, got {error:?}"
        );
    }

    #[test]
    fn focused_source_factory_accepts_axis_reference_explicit_radius_aperture() {
        let mut grid = Grid::new(64, 64, 32, 0.008, 0.008, 0.008).unwrap();
        grid.origin = [-0.256, -0.256, -0.032];
        let axis_reference = [0.0, 0.0, 0.04];
        let focus = [0.0, 0.0, 0.0];
        let radius = 0.16_f64;
        let theta_min = 0.20_f64;
        let theta_max = 0.90_f64;
        let element_count = 23;
        let config = DomainSourceParameters {
            model: SourceModel::Focused,
            position: axis_reference,
            focus: Some(focus),
            radius: 0.01,
            frequency: 650.0e3,
            num_elements: Some(element_count),
            focused_bowl_aperture: FocusedBowlAperture::AxisReferencePolarBounds {
                radius_of_curvature_m: radius,
                theta_min_rad: theta_min,
                theta_max_rad: theta_max,
            },
            ..Default::default()
        };

        let source = SourceFactory::create_source(&config, &grid).unwrap();
        let axis_norm = ((focus[0] - axis_reference[0]).powi(2)
            + (focus[1] - axis_reference[1]).powi(2)
            + (focus[2] - axis_reference[2]).powi(2))
        .sqrt();
        let axis_unit = [
            (focus[0] - axis_reference[0]) / axis_norm,
            (focus[1] - axis_reference[1]) / axis_norm,
            (focus[2] - axis_reference[2]) / axis_norm,
        ];

        assert_eq!(source.positions().len(), element_count);
        assert_eq!(source.focal_point(), Some((focus[0], focus[1], focus[2])));
        for position in source.positions() {
            let vector_focus_to_element = [
                focus[0] - position.0,
                focus[1] - position.1,
                focus[2] - position.2,
            ];
            let distance = (vector_focus_to_element[0].powi(2)
                + vector_focus_to_element[1].powi(2)
                + vector_focus_to_element[2].powi(2))
            .sqrt();
            let axis_projection = axis_unit[0].mul_add(
                vector_focus_to_element[0],
                axis_unit[1].mul_add(
                    vector_focus_to_element[1],
                    axis_unit[2] * vector_focus_to_element[2],
                ),
            ) / radius;

            assert!((distance - radius).abs() < 1.0e-12);
            assert!(axis_projection <= theta_min.cos() + 1.0e-12);
            assert!(axis_projection >= theta_max.cos() - 1.0e-12);
        }
    }

    #[test]
    fn focused_source_factory_accepts_axis_reference_hemisphere_aperture() {
        let mut grid = Grid::new(64, 64, 40, 0.008, 0.008, 0.008).unwrap();
        grid.origin = [-0.256, -0.256, -0.064];
        let axis_reference = [0.0, 0.0, 0.04];
        let focus = [0.0, 0.0, 0.0];
        let radius = 0.16_f64;
        let element_count = 29;
        let config = DomainSourceParameters {
            model: SourceModel::Focused,
            position: axis_reference,
            focus: Some(focus),
            radius: 0.01,
            frequency: 650.0e3,
            num_elements: Some(element_count),
            focused_bowl_aperture: FocusedBowlAperture::AxisReferenceHemisphere {
                radius_of_curvature_m: radius,
            },
            ..Default::default()
        };

        let source = SourceFactory::create_source(&config, &grid).unwrap();
        let axis_norm = ((focus[0] - axis_reference[0]).powi(2)
            + (focus[1] - axis_reference[1]).powi(2)
            + (focus[2] - axis_reference[2]).powi(2))
        .sqrt();
        let axis_unit = [
            (focus[0] - axis_reference[0]) / axis_norm,
            (focus[1] - axis_reference[1]) / axis_norm,
            (focus[2] - axis_reference[2]) / axis_norm,
        ];

        assert_eq!(source.positions().len(), element_count);
        assert_eq!(source.focal_point(), Some((focus[0], focus[1], focus[2])));
        for position in source.positions() {
            let focus_to_element = [
                focus[0] - position.0,
                focus[1] - position.1,
                focus[2] - position.2,
            ];
            let distance = (focus_to_element[0].powi(2)
                + focus_to_element[1].powi(2)
                + focus_to_element[2].powi(2))
            .sqrt();
            let axis_projection = axis_unit[0].mul_add(
                focus_to_element[0],
                axis_unit[1].mul_add(focus_to_element[1], axis_unit[2] * focus_to_element[2]),
            ) / radius;

            assert!((distance - radius).abs() < 1.0e-12);
            assert!(axis_projection >= -1.0e-12);
            assert!(axis_projection <= 1.0 + 1.0e-12);
        }
    }
}
