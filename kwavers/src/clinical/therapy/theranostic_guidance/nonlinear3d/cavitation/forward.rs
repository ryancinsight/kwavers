//! Rayleigh-Plesset cavitation forward map.
//!
//! Each voxel inside the planned treatment inversion window receives the local
//! Westervelt peak pressure as the acoustic forcing amplitude in the
//! Rayleigh-Plesset ODE. The source density is the maximum period-doubled radius
//! response normalised by the peak pressure inside that source support. Voxels
//! outside that treatment window, including skin/source boundary cells, emit no
//! passive cavitation source even if their source-injection pressure is high.

use ndarray::Array3;
use rayon::prelude::*;

use crate::physics::acoustics::bubble_dynamics::{
    BubbleParameters, BubbleState, RayleighPlessetSolver,
};

#[cfg(test)]
use super::super::types::GridIndex;
use super::super::types::{Nonlinear3dConfig, Nonlinear3dVolume};

pub(super) fn cavitation_source(
    volume: &Nonlinear3dVolume,
    peak_pressure: &Array3<f64>,
    config: &Nonlinear3dConfig,
) -> Array3<f64> {
    let dim = peak_pressure.dim();
    if let (Some(pressures), Some(source_mask)) = (
        peak_pressure.as_slice_memory_order(),
        volume.inversion_mask.as_slice_memory_order(),
    ) {
        let max_pressure = pressures
            .par_iter()
            .zip(source_mask.par_iter())
            .filter_map(|(&pressure, &active)| active.then_some(pressure))
            .reduce(|| 0.0, f64::max)
            .max(1.0);
        let values = pressures
            .par_iter()
            .zip(source_mask.par_iter())
            .map(|(&pressure, &active)| cavitation_value(active, pressure, max_pressure, config))
            .collect::<Vec<_>>();
        return Array3::from_shape_vec(dim, values)
            .expect("contiguous cavitation source length must match grid shape");
    }

    let max_pressure = peak_pressure
        .iter()
        .zip(volume.inversion_mask.iter())
        .filter_map(|(&pressure, &active)| active.then_some(pressure))
        .fold(0.0, f64::max)
        .max(1.0);
    Array3::from_shape_fn(dim, |idx| {
        cavitation_value(
            volume.inversion_mask[idx],
            peak_pressure[idx],
            max_pressure,
            config,
        )
    })
}

fn cavitation_value(
    active_source_voxel: bool,
    pressure_pa: f64,
    max_pressure_pa: f64,
    config: &Nonlinear3dConfig,
) -> f64 {
    if !active_source_voxel {
        return 0.0;
    }
    if mechanical_index(pressure_pa, config.frequency_hz) < config.inertial_mi_threshold {
        return 0.0;
    }
    let response = rayleigh_plesset_subharmonic_response(pressure_pa, config);
    response * (pressure_pa / max_pressure_pa).clamp(0.0, 1.0)
}

fn mechanical_index(pressure_pa: f64, frequency_hz: f64) -> f64 {
    let frequency_mhz = frequency_hz * 1.0e-6;
    if pressure_pa <= 0.0 || frequency_mhz <= 0.0 {
        return 0.0;
    }
    pressure_pa * 1.0e-6 / frequency_mhz.sqrt()
}

fn rayleigh_plesset_subharmonic_response(pressure_pa: f64, config: &Nonlinear3dConfig) -> f64 {
    if pressure_pa <= 0.0 {
        return 0.0;
    }
    let mut params = BubbleParameters {
        r0: config.bubble_radius_m,
        driving_frequency: config.frequency_hz,
        driving_amplitude: pressure_pa,
        use_thermal_effects: false,
        use_mass_transfer: false,
        use_compressibility: false,
        ..BubbleParameters::default()
    };
    params.initial_gas_pressure = params.p0;
    let solver = RayleighPlessetSolver::new(params.clone());
    let mut state = BubbleState::at_equilibrium(&params);
    let steps_per_period = config.bubble_time_steps_per_period;
    let total_steps = (config.cycles.ceil() as usize + 2) * steps_per_period;
    let dt = 1.0 / (config.frequency_hz * steps_per_period as f64);
    let mut period_lag = PeriodLagRadiusBuffer::new(steps_per_period, state.radius);
    let mut max_subharmonic: f64 = 0.0;
    let mut max_compression: f64 = 1.0;
    for step in 0..total_steps {
        state = rk4_step(
            &solver,
            &state,
            pressure_pa,
            step as f64 * dt,
            dt,
            params.r0,
        );
        state.update_compression(params.r0);
        max_compression = max_compression.max(state.compression_ratio);
        if let Some(previous_period) = period_lag.push(state.radius) {
            max_subharmonic =
                max_subharmonic.max((state.radius - previous_period).abs() / params.r0);
        }
        if state.radius <= 0.05 * params.r0 {
            max_subharmonic = max_subharmonic.max(max_compression);
            break;
        }
    }
    max_subharmonic.max((max_compression - 1.0).max(0.0))
}

/// Fixed-period radius history for the Rayleigh-Plesset subharmonic map.
///
/// The period-doubling observable is `max_t |R(t) - R(t - T)| / R0`. A full
/// radius history is unnecessary because the recurrence only reads the sample
/// one drive period behind the current RK4 state. This ring buffer stores
/// exactly one period of radii and is algebraically equivalent to indexing a
/// full history at `n - steps_per_period`.
struct PeriodLagRadiusBuffer {
    values: Vec<f64>,
    cursor: usize,
    fill_count: usize,
}

impl PeriodLagRadiusBuffer {
    fn new(steps_per_period: usize, initial_radius: f64) -> Self {
        debug_assert!(steps_per_period > 0);
        Self {
            values: vec![initial_radius; steps_per_period],
            cursor: 0,
            fill_count: 0,
        }
    }

    fn push(&mut self, radius: f64) -> Option<f64> {
        let previous_period = self.values[self.cursor];
        self.values[self.cursor] = radius;
        self.cursor += 1;
        if self.cursor == self.values.len() {
            self.cursor = 0;
        }
        if self.fill_count + 1 < self.values.len() {
            self.fill_count += 1;
            None
        } else {
            self.fill_count = self.values.len();
            Some(previous_period)
        }
    }
}

fn rk4_step(
    solver: &RayleighPlessetSolver,
    state: &BubbleState,
    pressure_pa: f64,
    t: f64,
    dt: f64,
    r0: f64,
) -> BubbleState {
    let k1 = derivative(solver, state, pressure_pa, t);
    let s2 = shifted_state(state, k1, 0.5 * dt, r0);
    let k2 = derivative(solver, &s2, pressure_pa, t + 0.5 * dt);
    let s3 = shifted_state(state, k2, 0.5 * dt, r0);
    let k3 = derivative(solver, &s3, pressure_pa, t + 0.5 * dt);
    let s4 = shifted_state(state, k3, dt, r0);
    let k4 = derivative(solver, &s4, pressure_pa, t + dt);
    let mut out = state.clone();
    out.radius = (state.radius + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0).max(0.05 * r0);
    out.wall_velocity = state.wall_velocity + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
    out.wall_acceleration = solver.calculate_acceleration(&out, pressure_pa, t + dt);
    out
}

fn derivative(
    solver: &RayleighPlessetSolver,
    state: &BubbleState,
    pressure_pa: f64,
    t: f64,
) -> (f64, f64) {
    (
        state.wall_velocity,
        solver.calculate_acceleration(state, pressure_pa, t),
    )
}

fn shifted_state(state: &BubbleState, derivative: (f64, f64), dt: f64, r0: f64) -> BubbleState {
    let mut shifted = state.clone();
    shifted.radius = (state.radius + dt * derivative.0).max(0.05 * r0);
    shifted.wall_velocity = state.wall_velocity + dt * derivative.1;
    shifted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clinical::therapy::theranostic_guidance::AnatomyKind;

    #[test]
    fn period_lag_buffer_matches_full_history_indices() {
        let mut buffer = PeriodLagRadiusBuffer::new(3, 10.0);
        let samples = [11.0, 12.0, 13.0, 14.0, 15.0];
        let lagged = samples
            .into_iter()
            .filter_map(|sample| buffer.push(sample))
            .collect::<Vec<_>>();

        assert_eq!(lagged, vec![10.0, 11.0, 12.0]);
    }

    #[test]
    fn rayleigh_plesset_ring_history_matches_full_history_reference() {
        let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
        config.bubble_time_steps_per_period = 24;
        config.cycles = 2.0;
        config.frequency_hz = 500_000.0;

        let pressure = 2.0e6;
        let optimized = rayleigh_plesset_subharmonic_response(pressure, &config);
        let reference = reference_response_with_full_history(pressure, &config);

        assert!(
            (optimized - reference).abs() <= 1.0e-12 * reference.max(1.0),
            "ring-buffer response {optimized} must match full-history response {reference}"
        );
    }

    #[test]
    fn rayleigh_plesset_zero_pressure_has_zero_subharmonic_response() {
        let config = Nonlinear3dConfig::new(AnatomyKind::Kidney);
        let response = rayleigh_plesset_subharmonic_response(0.0, &config);
        assert_eq!(response, 0.0);
    }

    #[test]
    fn cavitation_value_rejects_subthreshold_mechanical_index() {
        let mut config = Nonlinear3dConfig::new(AnatomyKind::Brain);
        config.frequency_hz = 650_000.0;
        config.inertial_mi_threshold = 1.9;
        let subthreshold_pressure = 1.5e5;

        let value = cavitation_value(true, subthreshold_pressure, subthreshold_pressure, &config);

        assert_eq!(value, 0.0);
        assert!(mechanical_index(subthreshold_pressure, config.frequency_hz) < 1.9);
    }

    #[test]
    fn cavitation_value_keeps_suprathreshold_rayleigh_plesset_source() {
        let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
        config.frequency_hz = 500_000.0;
        config.inertial_mi_threshold = 1.9;
        config.bubble_time_steps_per_period = 24;
        config.cycles = 2.0;
        let pressure = 2.0e6;

        let value = cavitation_value(true, pressure, pressure, &config);

        assert!(mechanical_index(pressure, config.frequency_hz) >= 1.9);
        assert!(value > 0.0);
    }

    #[test]
    fn cavitation_source_rejects_suprathreshold_pressure_outside_treatment_window() {
        let mut config = Nonlinear3dConfig::new(AnatomyKind::Kidney);
        config.frequency_hz = 500_000.0;
        config.inertial_mi_threshold = 1.9;
        config.bubble_time_steps_per_period = 24;
        config.cycles = 2.0;
        let n = 5;
        let center = (2, 2, 2);
        let boundary = (1, 1, 1);
        let volume = treatment_window_fixture(n, center);
        let mut pressure = Array3::<f64>::zeros((n, n, n));
        pressure[boundary] = 8.0e6;
        pressure[center] = 2.0e6;

        let source = cavitation_source(&volume, &pressure, &config);
        let expected = cavitation_value(true, pressure[center], pressure[center], &config);

        assert_eq!(source[boundary], 0.0);
        assert!(
            (source[center] - expected).abs() <= 1.0e-12 * expected.max(1.0),
            "target-window source {} must be normalized by active pressure {}, not outside pressure {}",
            source[center],
            pressure[center],
            pressure[boundary]
        );
    }

    fn treatment_window_fixture(n: usize, target: (usize, usize, usize)) -> Nonlinear3dVolume {
        let shape = (n, n, n);
        let body_mask = Array3::<bool>::from_elem(shape, true);
        let mut target_mask = Array3::<bool>::from_elem(shape, false);
        target_mask[target] = true;
        let inversion_mask = target_mask.clone();
        Nonlinear3dVolume {
            anatomy: AnatomyKind::Kidney,
            ct_hu: Array3::<f64>::zeros(shape),
            label: Array3::<i16>::zeros(shape),
            body_mask,
            target_mask,
            inversion_mask,
            density_kg_m3: Array3::<f64>::from_elem(shape, 1000.0),
            background_beta: Array3::<f64>::from_elem(shape, 4.0),
            true_beta: Array3::<f64>::from_elem(shape, 4.0),
            background_sound_speed_m_s: Array3::<f64>::from_elem(shape, 1500.0),
            true_sound_speed_m_s: Array3::<f64>::from_elem(shape, 1500.0),
            attenuation_np_per_m_mhz: Array3::<f64>::zeros(shape),
            attenuation_power_law_y: Array3::<f64>::from_elem(shape, 1.05),
            spacing_m: 1.0e-3,
            source_dimensions: [n, n, n],
            source_spacing_m: [1.0e-3; 3],
            crop_bounds_index: [0, n, 0, n, 0, n],
            aperture_direction: None,
            aperture_skin: None,
            focus: GridIndex {
                x: target.0,
                y: target.1,
                z: target.2,
            },
        }
    }

    fn reference_response_with_full_history(pressure_pa: f64, config: &Nonlinear3dConfig) -> f64 {
        let mut params = BubbleParameters {
            r0: config.bubble_radius_m,
            driving_frequency: config.frequency_hz,
            driving_amplitude: pressure_pa,
            use_thermal_effects: false,
            use_mass_transfer: false,
            use_compressibility: false,
            ..BubbleParameters::default()
        };
        params.initial_gas_pressure = params.p0;
        let solver = RayleighPlessetSolver::new(params.clone());
        let mut state = BubbleState::at_equilibrium(&params);
        let steps_per_period = config.bubble_time_steps_per_period;
        let total_steps = (config.cycles.ceil() as usize + 2) * steps_per_period;
        let dt = 1.0 / (config.frequency_hz * steps_per_period as f64);
        let mut radii = Vec::with_capacity(total_steps + 1);
        radii.push(state.radius);
        let mut max_subharmonic: f64 = 0.0;
        let mut max_compression: f64 = 1.0;
        for step in 0..total_steps {
            state = rk4_step(
                &solver,
                &state,
                pressure_pa,
                step as f64 * dt,
                dt,
                params.r0,
            );
            state.update_compression(params.r0);
            max_compression = max_compression.max(state.compression_ratio);
            radii.push(state.radius);
            if radii.len() > steps_per_period {
                let previous_period = radii[radii.len() - 1 - steps_per_period];
                max_subharmonic =
                    max_subharmonic.max((state.radius - previous_period).abs() / params.r0);
            }
            if state.radius <= 0.05 * params.r0 {
                max_subharmonic = max_subharmonic.max(max_compression);
                break;
            }
        }
        max_subharmonic.max((max_compression - 1.0).max(0.0))
    }
}
