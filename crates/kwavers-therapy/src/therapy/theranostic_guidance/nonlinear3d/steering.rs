//! Electronic steering calibration for abdominal nonlinear histotripsy solves.

use leto::Array3;

use super::super::AnatomyKind;
use super::encoding::SourceEncoding;
use super::forward::{forward_with_schedule, time_schedule, ForwardInput};
use super::types::{
    flat_index, ElectronicSteeringMetrics, GridIndex, Nonlinear3dAperture, Nonlinear3dConfig,
    Nonlinear3dVolume,
};

pub(crate) struct SteeringOutcome {
    pub(crate) aperture: Nonlinear3dAperture,
    pub(crate) metrics: ElectronicSteeringMetrics,
}

pub(crate) fn calibrate_electronic_steering(
    volume: &Nonlinear3dVolume,
    aperture: &Nonlinear3dAperture,
    config: &Nonlinear3dConfig,
) -> SteeringOutcome {
    let nominal_focus = volume.focus;
    if !matches!(volume.anatomy, AnatomyKind::Liver | AnatomyKind::Kidney) {
        return unchanged_outcome(aperture, nominal_focus);
    }

    let n = volume.body_mask.shape()[0];
    let cells = n * n * n;
    let background_speed = flatten(&volume.background_sound_speed_m_s);
    let density = flatten(&volume.density_kg_m3);
    let attenuation_alpha0 = flatten(&volume.attenuation_np_per_m_mhz);
    let attenuation_y = flatten(&volume.attenuation_power_law_y);
    let body = volume.body_mask.iter().copied().collect::<Vec<_>>();
    let inversion = volume.inversion_mask.iter().copied().collect::<Vec<_>>();
    let zero_beta = vec![0.0; cells];
    let schedule = time_schedule(&background_speed, n, volume.spacing_m, config);
    let target = volume.target_mask.iter().copied().collect::<Vec<_>>();
    let nominal = calibration_measurement(CalibrationInput {
        speed: &background_speed,
        density: &density,
        beta: &zero_beta,
        attenuation_alpha0: &attenuation_alpha0,
        attenuation_y: &attenuation_y,
        body: &body,
        target: &target,
        inversion: &inversion,
        n,
        spacing_m: volume.spacing_m,
        aperture,
        config,
        schedule,
    });
    let measured_error = [
        nominal_focus.x as isize - nominal.hotspot.x as isize,
        nominal_focus.y as isize - nominal.hotspot.y as isize,
        nominal_focus.z as isize - nominal.hotspot.z as isize,
    ];
    let mut candidates = vec![nominal_focus];
    for correction in [
        measured_error,
        [-measured_error[0], -measured_error[1], -measured_error[2]],
    ] {
        let candidate = constrained_focus(nominal_focus, correction, &body, n);
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }

    let mut best_focus = nominal_focus;
    let mut best = nominal;
    for focus in candidates.into_iter().skip(1) {
        let mut candidate_aperture = aperture.clone();
        candidate_aperture.focus = focus;
        let measurement = calibration_measurement(CalibrationInput {
            speed: &background_speed,
            density: &density,
            beta: &zero_beta,
            attenuation_alpha0: &attenuation_alpha0,
            attenuation_y: &attenuation_y,
            body: &body,
            target: &target,
            inversion: &inversion,
            n,
            spacing_m: volume.spacing_m,
            aperture: &candidate_aperture,
            config,
            schedule,
        });
        if measurement.better_than(&best, nominal_focus) {
            best_focus = focus;
            best = measurement;
        }
    }

    let correction = [
        best_focus.x as isize - nominal_focus.x as isize,
        best_focus.y as isize - nominal_focus.y as isize,
        best_focus.z as isize - nominal_focus.z as isize,
    ];
    let steering_applied = best_focus != nominal_focus;
    let mut steered = aperture.clone();
    steered.focus = best_focus;
    SteeringOutcome {
        aperture: steered,
        metrics: ElectronicSteeringMetrics {
            nominal_focus_index: grid_index_array(nominal_focus),
            calibration_hotspot_index: grid_index_array(best.hotspot),
            steering_focus_index: grid_index_array(best_focus),
            correction_grid_cells: correction,
            calibration_hotspot_distance_grid_cells: grid_distance(nominal_focus, best.hotspot),
            steering_applied,
        },
    }
}

struct CalibrationInput<'a> {
    speed: &'a [f64],
    density: &'a [f64],
    beta: &'a [f64],
    attenuation_alpha0: &'a [f64],
    attenuation_y: &'a [f64],
    body: &'a [bool],
    target: &'a [bool],
    inversion: &'a [bool],
    n: usize,
    spacing_m: f64,
    aperture: &'a Nonlinear3dAperture,
    config: &'a Nonlinear3dConfig,
    schedule: super::forward::TimeSchedule,
}

#[derive(Clone, Copy)]
struct CalibrationMeasurement {
    hotspot: GridIndex,
    target_peak_pa: f64,
    window_peak_pa: f64,
}

impl CalibrationMeasurement {
    fn better_than(self, other: &Self, nominal_focus: GridIndex) -> bool {
        let self_ratio = self.target_peak_pa / self.window_peak_pa.max(1.0);
        let other_ratio = other.target_peak_pa / other.window_peak_pa.max(1.0);
        self_ratio
            .total_cmp(&other_ratio)
            .then_with(|| {
                grid_distance(other.hotspot, nominal_focus)
                    .total_cmp(&grid_distance(self.hotspot, nominal_focus))
            })
            .then_with(|| self.target_peak_pa.total_cmp(&other.target_peak_pa))
            .is_gt()
    }
}

fn calibration_measurement(input: CalibrationInput<'_>) -> CalibrationMeasurement {
    let calibration = forward_with_schedule(ForwardInput {
        speed: input.speed,
        density: input.density,
        beta: input.beta,
        attenuation_np_per_m_mhz: Some(input.attenuation_alpha0),
        attenuation_power_law_y: Some(input.attenuation_y),
        source_body_mask: Some(input.body),
        n: input.n,
        spacing_m: input.spacing_m,
        aperture: input.aperture,
        config: input.config,
        schedule: input.schedule,
        encoding: SourceEncoding { index: 0, count: 1 },
        source_scale: 1.0,
        retain_history: false,
    });
    let hotspot = strongest_masked_index(&calibration.peak_pressure, input.inversion, input.n)
        .unwrap_or(input.aperture.focus);
    CalibrationMeasurement {
        hotspot,
        target_peak_pa: masked_peak(&calibration.peak_pressure, input.target),
        window_peak_pa: masked_peak(&calibration.peak_pressure, input.inversion),
    }
}

fn masked_peak(values: &[f64], mask: &[bool]) -> f64 {
    values
        .iter()
        .zip(mask.iter())
        .filter_map(|(value, active)| active.then_some(value.abs()))
        .filter(|value| value.is_finite())
        .fold(0.0, f64::max)
}

fn unchanged_outcome(aperture: &Nonlinear3dAperture, focus: GridIndex) -> SteeringOutcome {
    SteeringOutcome {
        aperture: aperture.clone(),
        metrics: ElectronicSteeringMetrics {
            nominal_focus_index: grid_index_array(focus),
            calibration_hotspot_index: grid_index_array(focus),
            steering_focus_index: grid_index_array(focus),
            correction_grid_cells: [0, 0, 0],
            calibration_hotspot_distance_grid_cells: 0.0,
            steering_applied: false,
        },
    }
}

fn strongest_masked_index(values: &[f64], mask: &[bool], n: usize) -> Option<GridIndex> {
    values
        .iter()
        .zip(mask.iter())
        .enumerate()
        .filter_map(|(cell, (value, active))| active.then_some((cell, value.abs())))
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(cell, _)| grid_index(cell, n))
}

fn constrained_focus(
    nominal_focus: GridIndex,
    correction: [isize; 3],
    body: &[bool],
    n: usize,
) -> GridIndex {
    let max_step = correction
        .iter()
        .map(|value| value.unsigned_abs())
        .max()
        .unwrap_or(0);
    for step in (0..=max_step).rev() {
        let candidate = shifted_focus(nominal_focus, correction, step, max_step, n);
        if body[flat_index(candidate, n)] {
            return candidate;
        }
    }
    nominal_focus
}

fn shifted_focus(
    nominal_focus: GridIndex,
    correction: [isize; 3],
    step: usize,
    max_step: usize,
    n: usize,
) -> GridIndex {
    if max_step == 0 {
        return nominal_focus;
    }
    GridIndex {
        x: shifted_axis(nominal_focus.x, correction[0], step, max_step, n),
        y: shifted_axis(nominal_focus.y, correction[1], step, max_step, n),
        z: shifted_axis(nominal_focus.z, correction[2], step, max_step, n),
    }
}

fn shifted_axis(axis: usize, correction: isize, step: usize, max_step: usize, n: usize) -> usize {
    let delta = correction * step as isize / max_step as isize;
    (axis as isize + delta).clamp(1, (n - 2) as isize) as usize
}

fn grid_index(cell: usize, n: usize) -> GridIndex {
    let n2 = n * n;
    GridIndex {
        x: cell / n2,
        y: (cell / n) % n,
        z: cell % n,
    }
}

fn grid_index_array(index: GridIndex) -> [usize; 3] {
    [index.x, index.y, index.z]
}

fn grid_distance(a: GridIndex, b: GridIndex) -> f64 {
    let dx = a.x as f64 - b.x as f64;
    let dy = a.y as f64 - b.y as f64;
    let dz = a.z as f64 - b.z as f64;
    dx.hypot(dy).hypot(dz)
}

fn flatten(values: &Array3<f64>) -> Vec<f64> {
    values.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrained_focus_applies_opposite_measured_hotspot_offset_inside_body() {
        let n = 9;
        let body = vec![true; n * n * n];
        let target = GridIndex { x: 4, y: 4, z: 4 };
        let corrected = constrained_focus(target, [1, -2, 0], &body, n);

        assert_eq!(corrected, GridIndex { x: 5, y: 2, z: 4 });
    }

    #[test]
    fn constrained_focus_reduces_correction_until_inside_body() {
        let n = 9;
        let mut body = vec![false; n * n * n];
        let target = GridIndex { x: 4, y: 4, z: 4 };
        body[flat_index(target, n)] = true;
        body[flat_index(GridIndex { x: 5, y: 4, z: 4 }, n)] = true;

        let corrected = constrained_focus(target, [3, 0, 0], &body, n);

        assert_eq!(corrected, GridIndex { x: 5, y: 4, z: 4 });
    }
}
