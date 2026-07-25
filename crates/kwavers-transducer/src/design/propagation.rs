//! Focused-array propagation over a synthesized aperture.
//!
//! The design module owns both the realized channel geometry and this coherent
//! pressure-envelope calculation so downstream crates do not reimplement array
//! propagation from copied pitch/channel scalars.

use aequitas::systems::si::quantities::{
    AcousticImpedance, ElectricCurrent, Frequency, Intensity, Length, Pressure,
    PressurePerElectricCurrent, Velocity,
};
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

use crate::transducers::physics::CartesianPosition;

use super::ArrayDesign;

const HALF_POWER_PRESSURE_RATIO: f64 = 0.5;
const WIDTH_SCAN_SAMPLES: usize = 160;
const WIDTH_BISECTION_STEPS: usize = 32;
const WIDTH_SEARCH_EXPANSIONS: usize = 8;

/// Inputs for focused propagation from a realized linear aperture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusedLinearArrayPropagationSpec {
    /// Synthesized array geometry and wiring.
    pub design: ArrayDesign,
    /// Array center in metres.
    pub center: CartesianPosition,
    /// Focus point in metres.
    pub focus: CartesianPosition,
    /// Drive frequency in hertz.
    pub frequency: Frequency,
    /// Medium sound speed in metres per second.
    pub sound_speed: Velocity,
    /// Peak current driven into each independent channel.
    pub per_channel_peak_current: ElectricCurrent,
    /// Peak pressure contribution per channel ampere at the focus.
    pub pressure_per_current: PressurePerElectricCurrent,
    /// Medium acoustic impedance in Rayl.
    pub acoustic_impedance: AcousticImpedance,
}

/// Focused propagation output derived from the realized channel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusedPressureMap {
    /// Coherent pressure magnitude at the requested focus.
    pub focal_pressure: Pressure,
    /// Mechanical Index at the focus.
    pub mechanical_index: f64,
    /// Spatial-peak pulse-average intensity at the focus.
    pub isppa: Intensity,
    /// Axial full width at the half-pressure contour.
    pub axial_extent: Length,
    /// Lateral full width at the half-pressure contour.
    pub lateral_extent: Length,
    /// True iff the realized steered pitch satisfies the spatial-Nyquist bound.
    pub grating_lobe_free: bool,
    /// True iff the requested focus lies beyond the realized-aperture Fraunhofer distance.
    pub in_far_field: bool,
}

/// Propagate a focused pressure envelope from a realized linear array.
///
/// The field is a coherent complex sum over the array's driven-channel
/// centroids. Each channel is phase-delayed to the requested focus, so the
/// focus is the constructive-interference target. Widths are measured from the
/// actual propagated envelope by finding the half-pressure crossings along the
/// lateral and axial axes through the focus.
///
/// # Errors
///
/// Returns `KwaversError::Config` when scalar inputs or the realized channel
/// geometry are non-physical.
pub fn propagate_focused_linear_array(
    spec: &FocusedLinearArrayPropagationSpec,
) -> KwaversResult<FocusedPressureMap> {
    validate_spec(spec)?;
    let scalar = ScalarPropagation::from_spec(spec);
    let channels = spec.design.channel_positions(scalar.center);
    let focal_pressure_pa = pressure_at(&channels, &scalar, scalar.focus);
    if !focal_pressure_pa.is_finite() || focal_pressure_pa <= 0.0 {
        return Err(invalid_value(
            "focal_pressure_pa",
            focal_pressure_pa,
            "finite and > 0 after coherent propagation",
        ));
    }

    let near_field = if scalar.wavelength > 0.0 {
        scalar.aperture_y * scalar.aperture_y / (4.0 * scalar.wavelength)
    } else {
        0.0
    };

    Ok(FocusedPressureMap {
        focal_pressure: Pressure::from_base(focal_pressure_pa),
        mechanical_index: mechanical_index(focal_pressure_pa, scalar.frequency),
        isppa: Intensity::from_base(acoustic_intensity_w_per_m2(
            focal_pressure_pa,
            scalar.impedance,
        )),
        axial_extent: Length::from_base(
            width_mm(&channels, &scalar, Axis::Axial, focal_pressure_pa)? * 1.0e-3,
        ),
        lateral_extent: Length::from_base(
            width_mm(&channels, &scalar, Axis::Lateral, focal_pressure_pa)? * 1.0e-3,
        ),
        grating_lobe_free: spec.design.grating_lobe_free,
        in_far_field: scalar.focus[2] >= near_field,
    })
}

fn validate_spec(spec: &FocusedLinearArrayPropagationSpec) -> KwaversResult<()> {
    let scalar = ScalarPropagation::from_spec(spec);
    validate_positive("frequency", scalar.frequency)?;
    validate_positive("sound_speed", scalar.sound_speed)?;
    validate_positive("per_channel_peak_current", scalar.current)?;
    validate_positive("pressure_per_current", scalar.pressure_per_current)?;
    validate_positive("acoustic_impedance", scalar.impedance)?;
    validate_point("center", scalar.center)?;
    validate_point("focus", scalar.focus)?;
    if spec.design.n_channels == 0 {
        return Err(invalid_value("n_channels", 0.0, "> 0"));
    }
    if spec.design.channel_positions(scalar.center).len() != spec.design.n_channels {
        return Err(KwaversError::Config(ConfigError::ValidationFailed {
            field: "channel_positions".to_owned(),
            value: spec
                .design
                .channel_positions(scalar.center)
                .len()
                .to_string(),
            constraint: format!("exactly {} driven channels", spec.design.n_channels),
        }));
    }
    Ok(())
}

struct ScalarPropagation {
    center: [f64; 3],
    focus: [f64; 3],
    frequency: f64,
    sound_speed: f64,
    current: f64,
    pressure_per_current: f64,
    impedance: f64,
    wavelength: f64,
    aperture_y: f64,
}

impl ScalarPropagation {
    fn from_spec(spec: &FocusedLinearArrayPropagationSpec) -> Self {
        Self {
            center: spec.center.into_base(),
            focus: spec.focus.into_base(),
            frequency: spec.frequency.into_base(),
            sound_speed: spec.sound_speed.into_base(),
            current: spec.per_channel_peak_current.into_base(),
            pressure_per_current: spec.pressure_per_current.into_base(),
            impedance: spec.acoustic_impedance.into_base(),
            wavelength: spec.design.wavelength.into_base(),
            aperture_y: spec.design.n_channels.saturating_sub(1) as f64
                * spec.design.pitch_y.into_base(),
        }
    }
}

fn validate_positive(parameter: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid_value(parameter, value, "finite and > 0"))
    }
}

fn validate_point(parameter: &str, point: [f64; 3]) -> KwaversResult<()> {
    if point.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: parameter.to_owned(),
            value: format!("{point:?}"),
            constraint: "all coordinates finite".to_owned(),
        }))
    }
}

fn invalid_value(parameter: &str, value: f64, constraint: &str) -> KwaversError {
    KwaversError::Config(ConfigError::InvalidValue {
        parameter: parameter.to_owned(),
        value: value.to_string(),
        constraint: constraint.to_owned(),
    })
}

fn pressure_at(channels: &[[f64; 3]], spec: &ScalarPropagation, point: [f64; 3]) -> f64 {
    let wavenumber = 2.0 * std::f64::consts::PI * spec.frequency / spec.sound_speed;
    let contribution_pa = spec.current * spec.pressure_per_current;
    let focus_distance_m = distance(spec.center, spec.focus).max(f64::MIN_POSITIVE);
    let mut real = 0.0;
    let mut imag = 0.0;
    for channel in channels {
        let focus_path_m = distance(*channel, spec.focus);
        let sample_path_m = distance(*channel, point).max(f64::MIN_POSITIVE);
        let phase = wavenumber * (sample_path_m - focus_path_m);
        let spread = focus_distance_m / sample_path_m;
        let amplitude = contribution_pa * spread;
        real += amplitude * phase.cos();
        imag += amplitude * phase.sin();
    }
    real.hypot(imag)
}

#[derive(Debug, Clone, Copy)]
enum Axis {
    Lateral,
    Axial,
}

fn width_mm(
    channels: &[[f64; 3]],
    spec: &ScalarPropagation,
    axis: Axis,
    focal_pressure_pa: f64,
) -> KwaversResult<f64> {
    let threshold = focal_pressure_pa * HALF_POWER_PRESSURE_RATIO;
    let search_extent_m = match axis {
        Axis::Lateral => (2.0 * spec.aperture_y).max(4.0 * spec.wavelength),
        Axis::Axial => (2.0 * spec.focus[2].abs()).max(8.0 * spec.wavelength),
    };
    let positive = half_width_m(channels, spec, axis, threshold, search_extent_m, 1.0)?;
    let negative = half_width_m(channels, spec, axis, threshold, search_extent_m, -1.0)?;
    Ok((positive + negative) * 1.0e3)
}

fn half_width_m(
    channels: &[[f64; 3]],
    spec: &ScalarPropagation,
    axis: Axis,
    threshold_pa: f64,
    search_extent_m: f64,
    direction: f64,
) -> KwaversResult<f64> {
    let mut extent = search_extent_m;
    let mut below = None;
    for _ in 0..=WIDTH_SEARCH_EXPANSIONS {
        for sample in 1..=WIDTH_SCAN_SAMPLES {
            let offset = extent * sample as f64 / WIDTH_SCAN_SAMPLES as f64;
            let point = offset_point(spec.focus, axis, direction * offset);
            if point[2] <= 0.0 {
                continue;
            }
            if pressure_at(channels, spec, point) <= threshold_pa {
                below = Some(offset);
                break;
            }
        }
        if below.is_some() {
            break;
        }
        extent *= 2.0;
    }
    let Some(mut hi) = below else {
        if matches!(axis, Axis::Axial) && direction < 0.0 {
            return Ok(spec.focus[2].max(0.0));
        }
        return Err(KwaversError::Config(ConfigError::ValidationFailed {
            field: "half_power_width".to_owned(),
            value: format!("{extent:.12e}"),
            constraint: "search extent reaches the half-pressure contour".to_owned(),
        }));
    };
    let mut lo = 0.0;
    for _ in 0..WIDTH_BISECTION_STEPS {
        let mid = 0.5 * (lo + hi);
        let point = offset_point(spec.focus, axis, direction * mid);
        if point[2] > 0.0 && pressure_at(channels, spec, point) > threshold_pa {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(hi)
}

fn offset_point(mut point: [f64; 3], axis: Axis, offset_m: f64) -> [f64; 3] {
    match axis {
        Axis::Lateral => point[1] += offset_m,
        Axis::Axial => point[2] += offset_m,
    }
    point
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt()
}

fn mechanical_index(pressure_pa: f64, frequency: f64) -> f64 {
    (pressure_pa / 1.0e6) / (frequency / 1.0e6).sqrt()
}

fn acoustic_intensity_w_per_m2(pressure_pa: f64, impedance: f64) -> f64 {
    pressure_pa * pressure_pa / (2.0 * impedance)
}

#[cfg(test)]
mod tests {
    use aequitas::systems::si::quantities::{
        AcousticImpedance, ElectricCurrent, Frequency, Length, PressurePerElectricCurrent, Velocity,
    };

    use super::*;
    use crate::{
        design_array, ApertureDesignSpec, CartesianPosition, ChannelWiring, DEFAULT_KERF_FRACTION,
    };

    fn spec() -> FocusedLinearArrayPropagationSpec {
        let design = design_array(&ApertureDesignSpec {
            aperture_x: Length::from_base(0.0),
            aperture_y: Length::from_base(96.0 * 0.25e-3),
            frequency: Frequency::from_base(500_000.0),
            sound_speed: Velocity::from_base(1540.0),
            max_pitch_fraction: 0.25e-3 / (1540.0 / 500_000.0),
            kerf_fraction: DEFAULT_KERF_FRACTION,
            wiring: ChannelWiring::ColumnsAsChannels,
        })
        .unwrap();
        FocusedLinearArrayPropagationSpec {
            design,
            center: CartesianPosition::from_base([0.0, 0.0, 0.0]).unwrap(),
            focus: CartesianPosition::from_base([0.0, 0.0, 0.010]).unwrap(),
            frequency: Frequency::from_base(500_000.0),
            sound_speed: Velocity::from_base(1540.0),
            per_channel_peak_current: ElectricCurrent::from_base(0.04),
            pressure_per_current: PressurePerElectricCurrent::from_base(9.375e6),
            acoustic_impedance: AcousticImpedance::from_base(1.48e6),
        }
    }

    #[test]
    fn focused_propagation_uses_all_realized_channels() {
        let spec = spec();
        let map = propagate_focused_linear_array(&spec).unwrap();
        let single_channel_pa =
            spec.per_channel_peak_current.into_base() * spec.pressure_per_current.into_base();
        let focus_distance_m = distance(spec.center.into_base(), spec.focus.into_base());
        let expected_focus_pa = single_channel_pa
            * spec
                .design
                .channel_positions(spec.center.into_base())
                .iter()
                .map(|channel| focus_distance_m / distance(*channel, spec.focus.into_base()))
                .sum::<f64>();
        assert!(
            (map.focal_pressure.into_base() - expected_focus_pa).abs()
                <= expected_focus_pa * 1.0e-12,
            "focused pressure must equal the coherent spherical-spreading sum"
        );
        assert!(map.focal_pressure.into_base() > single_channel_pa);
        assert!(map.mechanical_index > 0.0);
        assert!(map.isppa.into_base() > 0.0);
        assert!(map.lateral_extent.into_base() > 0.0);
        assert!(map.axial_extent > map.lateral_extent);
        assert!(map.grating_lobe_free);
    }

    #[test]
    fn propagation_rejects_nonfinite_focus() {
        let err = CartesianPosition::from_base([0.0, 0.0, f64::NAN]).unwrap_err();
        assert!(
            err.to_string().contains("position"),
            "invalid CartesianPosition must be rejected at construction"
        );
    }

    #[test]
    fn propagation_rejects_zero_drive_current() {
        let mut spec = spec();
        spec.per_channel_peak_current = ElectricCurrent::from_base(0.0);
        let err = propagate_focused_linear_array(&spec).unwrap_err();
        assert!(
            err.to_string().contains("per_channel_peak_current"),
            "error must name invalid current field: {err}"
        );
    }
}
