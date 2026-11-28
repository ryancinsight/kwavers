use super::coupling::{coupling_field, grid_gradient, history_row, pair_geometry};
use super::pass::{integrate_pass, PassHistories};
use super::{CoupledBubble, CoupledCloudConfig, BAND_1P5_HZ, COUPLED_VALIDATED_FREQUENCY_HZ};
use crate::acoustics::bubble_dynamics::adaptive_integration::AdaptiveBubbleModel;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array2;
use std::f64::consts::PI;

/// 1.5f₀ band power of one bubble's steady-segment monopole
/// pressure at 1 mm, `p_sc = ρ/(4π·1e-3)·V̈` (Hann window, direct
/// DFT over the band bins — the diagnostic of WS-B §2, used for
/// solver tolerances, not as a received-field model).
#[must_use]
pub fn segment_band_power(
    vddot_row: &[f64],
    dt: f64,
    segment_start: usize,
    density_kg_m3: f64,
) -> f64 {
    if vddot_row.len() <= segment_start + 2 {
        return 0.0;
    }
    let seg = &vddot_row[segment_start..];
    let n = seg.len();
    let scale = density_kg_m3 / (4.0 * PI * 1e-3);
    let duration = n as f64 * dt;
    let mut power = 0.0;
    let mut bin = (BAND_1P5_HZ.0 * duration).floor() as usize;
    while (bin as f64) / duration <= BAND_1P5_HZ.1 {
        let f = bin as f64 / duration;
        if f >= BAND_1P5_HZ.0 {
            let (mut re, mut im) = (0.0, 0.0);
            for (k, &v) in seg.iter().enumerate() {
                let hann = 0.5 * (1.0 - (2.0 * PI * k as f64 / (n - 1) as f64).cos());
                let phase = -2.0 * PI * bin as f64 * k as f64 / n as f64;
                let s = scale * v * hann / n as f64;
                re += s * phase.cos();
                im += s * phase.sin();
            }
            power += re * re + im * im;
        }
        bin += 1;
    }
    power
}

/// Outcome of a coupled solve.
#[derive(Debug, Clone)]
pub struct CoupledSolution {
    /// Radius histories [m], one row per bubble, on the grid.
    pub radius: Array2<f64>,
    /// Wall-velocity histories [m/s], per bubble.
    pub velocity: Array2<f64>,
    /// Volume-acceleration histories `V̈` [m³/s²], per bubble
    /// (the final pass's output; the coupling source term).
    pub volume_accel: Array2<f64>,
    /// Per-pass change of the cloud's total 1.5f₀ segment band
    /// power [dB] between successive pass outputs (index 0 is the
    /// change from independent driving to pass 1's input history
    /// blend — see [`solve_coupled_cloud`]). The Picard residual:
    /// it tends to 0 at the fixed point.
    pub pass_band_power_change_db: Vec<f64>,
}

/// Solve the coupled cloud by damped Picard iteration.
///
/// `models` holds one bubble model per bubble (each model's
/// parameters carry that bubble's `r0` and medium); `incident` is
/// the incident instantaneous-pressure history [Pa] on the same
/// grid (`config.n_cycles * config.steps_per_cycle + 1` samples),
/// spatially uniform over the cloud (the focal-region convention
/// of WS-B — propagation phase across the cloud is not part of the
/// validated configuration). `max_passes` counts coupling passes
/// after the independent pass 0.
///
/// The history fed forward is damped with
/// `config.damping_theta`; the returned histories are the final
/// pass's own output.
///
/// # Errors
/// [`KwaversError::InvalidInput`] on shape mismatches;
/// propagates KM integration errors (e.g. wall Mach > 0.95).
pub fn solve_coupled_cloud<M: AdaptiveBubbleModel>(
    models: &[M],
    bubbles: &[CoupledBubble],
    incident: &[f64],
    config: &CoupledCloudConfig,
    max_passes: usize,
) -> KwaversResult<CoupledSolution> {
    let nb = bubbles.len();
    if models.len() != nb || nb == 0 {
        return Err(KwaversError::InvalidInput(
            "coupled_driving: one model per bubble, at least one bubble".to_owned(),
        ));
    }
    let nsteps = config.n_cycles * config.steps_per_cycle;
    if incident.len() != nsteps + 1 {
        return Err(KwaversError::InvalidInput(format!(
            "coupled_driving: incident history has {} samples, grid needs {}",
            incident.len(),
            nsteps + 1
        )));
    }
    if !(0.0..=1.0).contains(&config.damping_theta) {
        return Err(KwaversError::InvalidInput(
            "coupled_driving: damping theta must lie in [0, 1]".to_owned(),
        ));
    }
    let dt = 1.0 / (COUPLED_VALIDATED_FREQUENCY_HZ * config.steps_per_cycle as f64);
    let positions: Vec<[f64; 3]> = bubbles.iter().map(|b| b.position_m).collect();
    let (weights, delays) = pair_geometry(
        &positions,
        config.density_kg_m3,
        config.propagation_sound_speed_m_s,
    );
    let seg_start = config.segment_start_cycle * config.steps_per_cycle;
    let band_power = |vddot: &Array2<f64>| -> f64 {
        (0..nb)
            .map(|i| segment_band_power(history_row(vddot, i), dt, seg_start, config.density_kg_m3))
            .sum()
    };

    // Pass 0: independent driving.
    let zero_hist = Array2::zeros([nb, nsteps + 1]);
    let run_pass = |input_hist: &Array2<f64>| {
        let coup = coupling_field(input_hist, &weights, &delays, dt);
        let mut drive_p = Array2::zeros([nb, nsteps + 1]);
        for i in 0..nb {
            for k in 0..=nsteps {
                drive_p[[i, k]] = incident[k] + coup[[i, k]];
            }
        }
        let drive_dp = grid_gradient(&drive_p, dt);
        integrate_pass(models, bubbles, &drive_p, &drive_dp, dt)
    };

    let PassHistories {
        mut radius,
        mut velocity,
        volume_accel: mut vddot,
    } = run_pass(&zero_hist)?;
    let mut input_hist = vddot.clone();
    let mut prev_power = band_power(&vddot);
    let mut changes = Vec::new();
    for _ in 0..max_passes {
        let next = run_pass(&input_hist)?;
        let power = band_power(&next.volume_accel);
        changes.push(10.0 * (power / prev_power).log10());
        prev_power = power;
        // Damped history for the next pass (exact: the coupling
        // field is linear in V̈).
        let theta = config.damping_theta;
        for i in 0..nb {
            for k in 0..=nsteps {
                input_hist[[i, k]] =
                    (1.0 - theta) * input_hist[[i, k]] + theta * next.volume_accel[[i, k]];
            }
        }
        radius = next.radius;
        velocity = next.velocity;
        vddot = next.volume_accel;
    }
    Ok(CoupledSolution {
        radius,
        velocity,
        volume_accel: vddot,
        pass_band_power_change_db: changes,
    })
}
