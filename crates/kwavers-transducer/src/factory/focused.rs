//! Focused bowl transducer construction for SourceFactory.

use crate::transducers::focused::{BowlAngularBounds, BowlConfig, BowlTransducer};
use kwavers_core::error::{ConfigError, KwaversResult};
use kwavers_source::config::DEFAULT_FOCUSED_BOWL_FOCUS_OFFSET_M;
use kwavers_source::{DomainSourceParameters, FocusedBowlAperture};

pub(super) fn create_focused_bowl_transducer(
    config: &DomainSourceParameters,
) -> KwaversResult<BowlTransducer> {
    let focus = focused_bowl_focus(config);
    match config.focused_bowl_aperture {
        FocusedBowlAperture::Diameter => {
            let bowl_config = base_bowl_config(config, focus);
            match config.num_elements {
                Some(element_count) => {
                    BowlTransducer::with_element_count(bowl_config, element_count)
                }
                None => BowlTransducer::new(bowl_config),
            }
        }
        FocusedBowlAperture::Hemisphere => {
            let bowl_config = base_bowl_config(config, focus);
            let element_count = required_focused_bowl_element_count(config)?;
            BowlTransducer::with_angular_bounds(
                bowl_config,
                BowlAngularBounds::hemisphere(),
                element_count,
            )
        }
        FocusedBowlAperture::PolarSpan { theta_max_rad } => {
            let bowl_config = base_bowl_config(config, focus);
            let element_count = required_focused_bowl_element_count(config)?;
            BowlTransducer::with_polar_span(bowl_config, theta_max_rad, element_count)
        }
        FocusedBowlAperture::PolarBounds {
            theta_min_rad,
            theta_max_rad,
        } => {
            let bowl_config = base_bowl_config(config, focus);
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
            let bowl_config = base_bowl_config(config, focus);
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
            // Aperture chord: 2 R sin(theta_max).
            let aperture_diameter_m = 2.0 * radius_of_curvature_m * theta_max_rad.sin();
            let mut axis_config = BowlConfig::from_axis_reference_focus(
                config.position,
                focus,
                radius_of_curvature_m,
                aperture_diameter_m,
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
            // Full hemisphere: theta_max = pi/2, sin(pi/2) = 1, aperture = 2R.
            let aperture_diameter_m = 2.0 * radius_of_curvature_m;
            let mut axis_config = BowlConfig::from_axis_reference_focus(
                config.position,
                focus,
                radius_of_curvature_m,
                aperture_diameter_m,
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

fn focused_bowl_focus(config: &DomainSourceParameters) -> [f64; 3] {
    config.focus.unwrap_or([
        config.position[0],
        config.position[1],
        config.position[2] + DEFAULT_FOCUSED_BOWL_FOCUS_OFFSET_M,
    ])
}

fn base_bowl_config(config: &DomainSourceParameters, focus: [f64; 3]) -> BowlConfig {
    let mut bowl_config = BowlConfig::from_vertex_focus(
        config.position,
        focus,
        2.0 * config.radius,
        config.frequency,
        config.amplitude,
    );
    bowl_config.phase = config.phase;
    bowl_config
}

pub(super) fn required_focused_bowl_element_count(
    config: &DomainSourceParameters,
) -> KwaversResult<usize> {
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
