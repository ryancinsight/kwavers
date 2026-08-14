//! Focused-array propagation over a synthesized aperture.
//!
//! The design module owns both the realized channel geometry and this coherent
//! pressure-envelope calculation so downstream crates do not reimplement array
//! propagation from copied pitch/channel scalars.

#![cfg_attr(test, expect(clippy::unwrap_used, reason = "ratchet KWAVERS-UNWRAP-1"))]

use aequitas::systems::si::quantities::{
    AcousticImpedance, Dimensionless, ElectricCurrent, Frequency, Intensity, Length, Pressure,
    PressurePerElectricCurrent, Velocity,
};
use aequitas::systems::si::units::{
    Ampere, Hertz, Meter, MeterPerSecond, Pascal, Rayl, WattPerSquareMeter,
};
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

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
    /// Array center.
    pub center: [Length<f64>; 3],
    /// Focus point.
    pub focus: [Length<f64>; 3],
    /// Drive frequency.
    pub frequency: Frequency<f64>,
    /// Medium sound speed.
    pub sound_speed: Velocity<f64>,
    /// Peak current driven into each independent channel.
    pub per_channel_peak_current: ElectricCurrent<f64>,
    /// Peak pressure contribution per channel ampere at the focus.
    pub pressure_per_current: PressurePerElectricCurrent<f64>,
    /// Medium acoustic impedance in Rayl.
    pub acoustic_impedance: AcousticImpedance<f64>,
}

/// Focused propagation output derived from the realized channel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocusedPressureMap {
    /// Coherent pressure magnitude at the requested focus.
    pub focal_pressure: Pressure<f64>,
    /// Mechanical Index at the focus.
    pub mechanical_index: Dimensionless<f64>,
    /// Spatial-peak pulse-average intensity at the focus.
    pub isppa: Intensity<f64>,
    /// Axial full width at the half-pressure contour.
    pub axial_extent: Length<f64>,
    /// Lateral full width at the half-pressure contour.
    pub lateral_extent: Length<f64>,
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
    let focus_m = spec.focus.map(|coordinate| coordinate.in_unit::<Meter>());
    let channels = spec
        .design
        .channel_positions(spec.center)
        .into_iter()
        .map(|channel| channel.map(|coordinate| coordinate.in_unit::<Meter>()))
        .collect::<Vec<_>>();
    let focal_pressure_pa = pressure_at(&channels, spec, focus_m);
    if !focal_pressure_pa.is_finite() || focal_pressure_pa <= 0.0 {
        return Err(invalid_value(
            "focal_pressure",
            focal_pressure_pa,
            "finite and > 0 after coherent propagation",
        ));
    }

    let wavelength_m = spec.design.wavelength.in_unit::<Meter>();
    let aperture_y_m =
        (spec.design.n_channels.saturating_sub(1)) as f64 * spec.design.pitch_y.in_unit::<Meter>();
    let near_field_m = if wavelength_m > 0.0 {
        aperture_y_m * aperture_y_m / (4.0 * wavelength_m)
    } else {
        0.0
    };

    Ok(FocusedPressureMap {
        focal_pressure: Pressure::from_unit::<Pascal>(focal_pressure_pa),
        mechanical_index: Dimensionless::from_base(mechanical_index(
            Pressure::from_unit::<Pascal>(focal_pressure_pa),
            spec.frequency,
        )),
        isppa: acoustic_intensity(
            Pressure::from_unit::<Pascal>(focal_pressure_pa),
            spec.acoustic_impedance,
        ),
        axial_extent: Length::from_unit::<Meter>(width_m(
            &channels,
            spec,
            Axis::Axial,
            focal_pressure_pa,
        )?),
        lateral_extent: Length::from_unit::<Meter>(width_m(
            &channels,
            spec,
            Axis::Lateral,
            focal_pressure_pa,
        )?),
        grating_lobe_free: spec.design.grating_lobe_free,
        in_far_field: focus_m[2] >= near_field_m,
    })
}

fn validate_spec(spec: &FocusedLinearArrayPropagationSpec) -> KwaversResult<()> {
    validate_positive("frequency", spec.frequency.in_unit::<Hertz>())?;
    validate_positive("sound_speed", spec.sound_speed.in_unit::<MeterPerSecond>())?;
    validate_positive(
        "per_channel_peak_current",
        spec.per_channel_peak_current.in_unit::<Ampere>(),
    )?;
    validate_positive(
        "pressure_per_current",
        spec.pressure_per_current.into_base(),
    )?;
    validate_positive(
        "acoustic_impedance",
        spec.acoustic_impedance.in_unit::<Rayl>(),
    )?;
    validate_point("center", spec.center)?;
    validate_point("focus", spec.focus)?;
    if spec.design.n_channels == 0 {
        return Err(invalid_value("n_channels", 0.0, "> 0"));
    }
    if spec.design.channel_positions(spec.center).len() != spec.design.n_channels {
        return Err(KwaversError::Config(ConfigError::ValidationFailed {
            field: "channel_positions".to_owned(),
            value: spec.design.channel_positions(spec.center).len().to_string(),
            constraint: format!("exactly {} driven channels", spec.design.n_channels),
        }));
    }
    Ok(())
}

fn validate_positive(parameter: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid_value(parameter, value, "finite and > 0"))
    }
}

fn validate_point(parameter: &str, point: [Length<f64>; 3]) -> KwaversResult<()> {
    if point
        .iter()
        .all(|value| value.in_unit::<Meter>().is_finite())
    {
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

fn pressure_at(
    channels: &[[f64; 3]],
    spec: &FocusedLinearArrayPropagationSpec,
    point_m: [f64; 3],
) -> f64 {
    let frequency_hz = spec.frequency.in_unit::<Hertz>();
    let sound_speed_m_s = spec.sound_speed.in_unit::<MeterPerSecond>();
    let current_a = spec.per_channel_peak_current.in_unit::<Ampere>();
    let pressure_per_current = spec.pressure_per_current.into_base();
    let center_m = spec.center.map(|coordinate| coordinate.in_unit::<Meter>());
    let focus_m = spec.focus.map(|coordinate| coordinate.in_unit::<Meter>());
    let wavenumber = 2.0 * std::f64::consts::PI * frequency_hz / sound_speed_m_s;
    let contribution_pa = current_a * pressure_per_current;
    let focus_distance_m = distance(center_m, focus_m).max(f64::MIN_POSITIVE);
    let mut real = 0.0;
    let mut imag = 0.0;
    for channel in channels {
        let focus_path_m = distance(*channel, focus_m);
        let sample_path_m = distance(*channel, point_m).max(f64::MIN_POSITIVE);
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

fn width_m(
    channels: &[[f64; 3]],
    spec: &FocusedLinearArrayPropagationSpec,
    axis: Axis,
    focal_pressure_pa: f64,
) -> KwaversResult<f64> {
    let threshold = focal_pressure_pa * HALF_POWER_PRESSURE_RATIO;
    let search_extent_m = match axis {
        Axis::Lateral => (2.0 * spec.design.aperture_y().in_unit::<Meter>())
            .max(4.0 * spec.design.wavelength.in_unit::<Meter>()),
        Axis::Axial => (2.0 * spec.focus[2].in_unit::<Meter>().abs())
            .max(8.0 * spec.design.wavelength.in_unit::<Meter>()),
    };
    let focus_m = spec.focus.map(|coordinate| coordinate.in_unit::<Meter>());
    let positive = half_width_m(
        channels,
        spec,
        focus_m,
        axis,
        threshold,
        search_extent_m,
        1.0,
    )?;
    let negative = half_width_m(
        channels,
        spec,
        focus_m,
        axis,
        threshold,
        search_extent_m,
        -1.0,
    )?;
    Ok(positive + negative)
}

fn half_width_m(
    channels: &[[f64; 3]],
    spec: &FocusedLinearArrayPropagationSpec,
    focus_m: [f64; 3],
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
            let point = offset_point(focus_m, axis, direction * offset);
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
            return Ok(spec.focus[2].in_unit::<Meter>().max(0.0));
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
        let point = offset_point(focus_m, axis, direction * mid);
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

fn mechanical_index(pressure: Pressure<f64>, frequency: Frequency<f64>) -> f64 {
    let pressure_pa = pressure.in_unit::<Pascal>();
    let frequency_hz = frequency.in_unit::<Hertz>();
    (pressure_pa / 1.0e6) / (frequency_hz / 1.0e6).sqrt()
}

fn acoustic_intensity(
    pressure: Pressure<f64>,
    impedance: AcousticImpedance<f64>,
) -> Intensity<f64> {
    let pressure_pa = pressure.in_unit::<Pascal>();
    let impedance_rayl = impedance.in_unit::<Rayl>();
    Intensity::from_unit::<WattPerSquareMeter>(pressure_pa * pressure_pa / (2.0 * impedance_rayl))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{design_array, ApertureDesignSpec, ChannelWiring, DEFAULT_KERF_FRACTION};

    fn spec() -> FocusedLinearArrayPropagationSpec {
        let design = design_array(&ApertureDesignSpec {
            aperture_x: Length::from_unit::<Meter>(0.0),
            aperture_y: Length::from_unit::<Meter>(96.0 * 0.25e-3),
            frequency: Frequency::from_unit::<Hertz>(500_000.0),
            sound_speed: Velocity::from_unit::<MeterPerSecond>(1540.0),
            max_pitch_fraction: Dimensionless::from_base(0.25e-3 / (1540.0 / 500_000.0)),
            kerf_fraction: Dimensionless::from_base(DEFAULT_KERF_FRACTION),
            wiring: ChannelWiring::ColumnsAsChannels,
        })
        .unwrap();
        FocusedLinearArrayPropagationSpec {
            design,
            center: [
                Length::from_unit::<Meter>(0.0),
                Length::from_unit::<Meter>(0.0),
                Length::from_unit::<Meter>(0.0),
            ],
            focus: [
                Length::from_unit::<Meter>(0.0),
                Length::from_unit::<Meter>(0.0),
                Length::from_unit::<Meter>(0.010),
            ],
            frequency: Frequency::from_unit::<Hertz>(500_000.0),
            sound_speed: Velocity::from_unit::<MeterPerSecond>(1540.0),
            per_channel_peak_current: ElectricCurrent::from_unit::<Ampere>(0.04),
            pressure_per_current: PressurePerElectricCurrent::from_base(9.375e6),
            acoustic_impedance: AcousticImpedance::from_unit::<Rayl>(1.48e6),
        }
    }

    #[test]
    fn focused_propagation_uses_all_realized_channels() {
        let spec = spec();
        let map = propagate_focused_linear_array(&spec).unwrap();
        let single_channel_pa = spec.per_channel_peak_current.in_unit::<Ampere>()
            * spec.pressure_per_current.into_base();
        let center = spec.center.map(|coordinate| coordinate.in_unit::<Meter>());
        let focus = spec.focus.map(|coordinate| coordinate.in_unit::<Meter>());
        let focus_distance_m = distance(center, focus);
        let expected_focus_pa = single_channel_pa
            * spec
                .design
                .channel_positions(spec.center)
                .iter()
                .map(|channel| {
                    let channel = channel.map(|coordinate| coordinate.in_unit::<Meter>());
                    focus_distance_m / distance(channel, focus)
                })
                .sum::<f64>();
        assert!(
            (map.focal_pressure.in_unit::<Pascal>() - expected_focus_pa).abs()
                <= expected_focus_pa * 1.0e-12,
            "focused pressure must equal the coherent spherical-spreading sum"
        );
        assert!(map.focal_pressure.in_unit::<Pascal>() > single_channel_pa);
        assert!(map.mechanical_index.into_base() > 0.0);
        assert!(map.isppa.in_unit::<WattPerSquareMeter>() > 0.0);
        assert!(map.lateral_extent.in_unit::<Meter>() > 0.0);
        assert!(map.axial_extent.in_unit::<Meter>() > map.lateral_extent.in_unit::<Meter>());
        assert!(map.grating_lobe_free);
    }

    #[test]
    fn propagation_rejects_nonfinite_focus() {
        let mut spec = spec();
        spec.focus[2] = Length::from_unit::<Meter>(f64::NAN);
        let err = propagate_focused_linear_array(&spec).unwrap_err();
        assert!(
            err.to_string().contains("focus"),
            "error must name invalid focus field: {err}"
        );
    }

    #[test]
    fn propagation_rejects_zero_drive_current() {
        let mut spec = spec();
        spec.per_channel_peak_current = ElectricCurrent::from_unit::<Ampere>(0.0);
        let err = propagate_focused_linear_array(&spec).unwrap_err();
        assert!(
            err.to_string().contains("per_channel_peak_current"),
            "error must name invalid current field: {err}"
        );
    }
}
