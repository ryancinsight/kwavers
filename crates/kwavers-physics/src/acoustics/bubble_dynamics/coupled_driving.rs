//! Foldy–Lax self-consistent coupled bubble driving
//! (Build 15 / WS-B, KW-PHYS-013).
//!
//! Every other cloud path in this programme drove each bubble with
//! the incident field alone and summed the received field linearly.
//! This module removes that approximation: bubble `i` is driven by
//!
//! ```text
//! p_i(t) = p_inc(t) + Σ_{j≠i} (ρ / 4π d_ij) · V̈_j(t − d_ij / c)
//! ```
//!
//! with `V = 4πR³/3` the bubble volume, `d_ij` the pair distance and
//! `c` the propagation sound speed (Foldy–Lax leading order:
//! monopole coupling only — no translation/Bjerknes drift, no
//! coalescence, no thermal coupling). Each bubble's radial dynamics
//! are integrated with the existing Keller–Miksis model
//! ([`super::keller_miksis::KellerMiksisModel`], via the
//! [`AdaptiveBubbleModel`] contract — the KM equation itself is
//! **not** duplicated here). The drive contract is the KM contract:
//! the *instantaneous* far-field pressure `p∞(t) − p0` and its time
//! derivative (Keller & Miksis 1980, Eq. 3.4); the total drive is
//! precomputed on the integration grid per Picard pass and its
//! derivative taken by centred differences of the total, exactly as
//! in the Python reference (`kwavers-work/b15/km_coupled.py`).
//!
//! # Picard iteration and damping
//!
//! Pass 0 is independent driving (coupling field zero). Each later
//! pass precomputes the coupling field from the previous history
//! and re-integrates every bubble. **Plain Picard oscillates** on
//! strongly coupled close pairs (WS-B diagnostic at 0.123 MPa:
//! segment band power +13.4 dB then −8.4 dB on successive passes),
//! so the history fed to the next pass is damped,
//! `V̈ ← (1 − θ)·V̈_prev + θ·V̈_new` with
//! [`PICARD_DAMPING_THETA`] `θ = 0.5`. Damping changes only the
//! path to the fixed point, never the fixed point itself: the
//! coupling field is linear in the `V̈` history, so history
//! blending is exact. With θ = 0.5 the diagnostic run converged
//! monotonically (−6.6 / −2.1 / −0.53 / −0.14 dB per pass).
//!
//! # Validation (Build 15 / WS-B, Python reference)
//!
//! * **Dilute limit:** at density ×0.008 (cloud radius ×5) the
//!   coupled solver reproduces independent driving — 1.5f₀ band
//!   power **−0.11 dB**, centre drive +0.18 dB, iteration converged
//!   to 0.007 dB by pass 3. (On the Kuji reduction *statistic* the
//!   same physical difference reads +1.24 dB at fc 3.0 MHz — the
//!   statistic divides by a near-floor 3P image and amplifies
//!   sub-0.2 dB emission changes; the 0.11 dB emission figure is
//!   the solver tolerance.)
//! * **KM cross-check:** the reference RHS (transcribed from this
//!   crate's `keller_miksis/equation.rs`) matches pykwavers
//!   `solve_keller_miksis` 1.5f₀ band power within **2.7 %** at
//!   0.123 / 0.268 MPa and **3.5 %** at 0.729 MPa
//!   (R(t) correlation ≥ 0.998); at 1.273 MPa the band max of the
//!   spiky collapse spectrum is trajectory-chaotic and only
//!   Rmax (within 3.5 %) is cited.
//!
//! # Measured effect size (WS-B, formation density ≈ 32 bubbles/mm³)
//!
//! Coupling does **not** close the Kuji cavitation-side gap. At
//! fc 3.0 MHz the cavitation-reduction change (coupled −
//! independent) is **+0.19 / +0.83 / +2.44 / −1.94 dB** at
//! 0.123 / 0.268 / 0.729 / 1.273 MPa — **+0.83 dB of the 10.5 dB
//! gap (≈ 8 %) at 0.26 MPa**, sign-reversing at the top PNP, and
//! PNP-dependent (4.4 dB spread) where the paper is nearly flat.
//! The raw bubble *emission* rises more (+2.0 to +6.6 dB in the
//! 1.5f₀ band) — a rare close-pair coherent-enhancement effect
//! (at 24 µm separation a neighbour's radiated field is comparable
//! to the incident field) — but most of it does not survive the
//! 3P/1P reduction statistic. Across a 0.25×–4× density bracket
//! the metric effect is not monotone and never exceeds ≈ +1 dB
//! at 0.268 MPa / fc 3.0 MHz.
//!
//! # No shadowing — and a warning about extinction
//!
//! At imaging conditions there is **no shadowing**: the total
//! drive at the cloud centre is *enhanced*, **+0.9 to +2.6 dB RMS**
//! relative to the incident tone — neighbours' fields arrive there
//! substantially in phase with the incident wave.
//!
//! **Warning:** the extinction coefficient α of
//! [`super::cloud_attenuation`] (KW-PHYS-006, Xia 2020 energy
//! method) is an *incoherent energy* quantity — each bubble does
//! remove energy (`W > 0`), and at imaging conditions α evaluates
//! to 8–68 Np/m (`αL ≤ 0.08` over a cloud radius). It must **not**
//! be imported as a *local drive reduction*: the coherent field
//! sum at a point can have the opposite sign (the centre-drive
//! measurement implies an effective α of −80 to −246 Np/m, a
//! gain). α is a propagation/energy quantity, not a local-drive
//! quantity.
//!
//! # Regime
//!
//! **Validated at imaging conditions only:** 2.45 MHz,
//! PNP 0.12–1.27 MPa, R0 ≈ 2–2.8 µm, densities to
//! ≈ 128 bubbles/mm³ (the 4× bracket point), void fraction
//! ≈ 2.9e-6 at the formation density. **NOT validated at trigger
//! pressures** (10–20 MPa): nothing here has been checked against
//! a coupled benchmark in the formation regime, and the KM solves
//! themselves leave their validated envelope there. Use
//! [`coupled_regime_reasons`] to check a proposed operating point
//! against the validated envelope. Monopole coupling also requires
//! pair distances ≫ R0 (the WS-B draw's minimum pair distance was
//! 24 µm ≈ 8.6 R0).

use super::acoustic_pressure::InstantaneousPressure;
use super::adaptive_integration::AdaptiveBubbleModel;
use super::bubble_state::{BubbleParameters, BubbleState};
use aequitas::systems::si::quantities::PressureRate;
use kwavers_core::error::{KwaversError, KwaversResult};
use std::f64::consts::PI;

/// Picard damping factor θ (WS-B): the history fed to the next
/// pass is `(1 − θ)·previous + θ·new`. θ = 0.5 converges
/// monotonically where plain Picard (θ = 1) oscillates; the fixed
/// point is θ-independent because the coupling field is linear in
/// the `V̈` history.
pub const PICARD_DAMPING_THETA: f64 = 0.5;

/// Validated drive frequency [Hz] (Kuji imaging, WS-B).
pub const COUPLED_VALIDATED_FREQUENCY_HZ: f64 = 2.45e6;

/// Validated incident PNP range [Pa] (WS-B sweep).
pub const COUPLED_VALIDATED_PNP_PA: (f64, f64) = (0.12e6, 1.27e6);

/// Highest density [bubbles/mm³] in the WS-B bracket (4× the
/// ≈ 32 bubbles/mm³ formation density).
pub const COUPLED_VALIDATED_MAX_DENSITY_PER_MM3: f64 = 128.0;

/// 1.5f₀ diagnostic band [Hz] at 2.45 MHz (prototype convention).
pub const BAND_1P5_HZ: (f64, f64) = (3.425e6, 3.925e6);

/// One bubble of the coupled cloud.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledBubble {
    /// Equilibrium radius `R0` [m].
    pub radius0_m: f64,
    /// Position [m] (3D; coupling distances are the full 3D
    /// distances, as in WS-B).
    pub position_m: [f64; 3],
}

/// Numerical configuration of the coupled solve. Defaults are the
/// WS-B production conventions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledCloudConfig {
    /// Liquid density `ρ` [kg/m³] in the monopole coupling weight
    /// `ρ/(4πd)` (WS-B: 998).
    pub density_kg_m3: f64,
    /// Propagation sound speed [m/s] for the retardation delays
    /// `d/c` (WS-B: 1540, the prototype imaging convention — the
    /// KM-internal sound speed stays each model's own `c_liquid`).
    pub propagation_sound_speed_m_s: f64,
    /// Picard damping θ (see [`PICARD_DAMPING_THETA`]).
    pub damping_theta: f64,
    /// Integration steps per drive cycle (the WS-B SPC ladder:
    /// 1000 at ≤ 0.27 MPa, 4000 at 0.73 MPa, 8000 at 1.27 MPa).
    pub steps_per_cycle: usize,
    /// Total drive cycles integrated (WS-B: 12).
    pub n_cycles: usize,
    /// First cycle of the steady segment used by the band
    /// diagnostic (WS-B: cycles 6–12).
    pub segment_start_cycle: usize,
}

impl Default for CoupledCloudConfig {
    fn default() -> Self {
        Self {
            density_kg_m3: 998.0,
            propagation_sound_speed_m_s: 1540.0,
            damping_theta: PICARD_DAMPING_THETA,
            steps_per_cycle: 1000,
            n_cycles: 12,
            segment_start_cycle: 6,
        }
    }
}

/// Reasons a proposed operating point lies outside the validated
/// envelope of this component (empty = inside). Checked against
/// the WS-B validation: imaging frequency and PNPs, bracketed
/// density. Trigger pressures are outside by construction.
#[must_use]
pub fn coupled_regime_reasons(
    frequency_hz: f64,
    pnp_pa: f64,
    density_per_mm3: f64,
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if (frequency_hz - COUPLED_VALIDATED_FREQUENCY_HZ).abs() > 0.01 * COUPLED_VALIDATED_FREQUENCY_HZ
    {
        reasons.push("drive frequency outside the validated 2.45 MHz imaging point");
    }
    if pnp_pa < COUPLED_VALIDATED_PNP_PA.0 || pnp_pa > COUPLED_VALIDATED_PNP_PA.1 {
        reasons.push("PNP outside the validated 0.12–1.27 MPa imaging range (not validated at trigger pressures)");
    }
    if density_per_mm3 > COUPLED_VALIDATED_MAX_DENSITY_PER_MM3 {
        reasons.push("density above the validated 128 bubbles/mm³ bracket");
    }
    reasons
}

/// Pair geometry: coupling weights `W[i][j] = ρ/(4π d_ij)`
/// (diagonal 0) and retardation delays `d_ij / c` [s].
#[must_use]
pub fn pair_geometry(
    positions: &[[f64; 3]],
    density_kg_m3: f64,
    propagation_sound_speed_m_s: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = positions.len();
    let mut weights = vec![vec![0.0; n]; n];
    let mut delays = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = positions[i][0] - positions[j][0];
            let dy = positions[i][1] - positions[j][1];
            let dz = positions[i][2] - positions[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > 0.0 {
                weights[i][j] = density_kg_m3 / (4.0 * PI * d);
                delays[i][j] = d / propagation_sound_speed_m_s;
            }
        }
    }
    (weights, delays)
}

/// Linear interpolation of one history row at time `tq` (zero
/// outside the history), the Python reference's `interp_rows`.
fn interp_row(row: &[f64], dt: f64, tq: f64) -> f64 {
    if row.len() < 2 || tq < 0.0 {
        return 0.0;
    }
    let x = tq / dt;
    let i0 = x.floor();
    if i0 < 0.0 || i0 as usize >= row.len() - 1 {
        return 0.0;
    }
    let i0 = i0 as usize;
    let frac = x - i0 as f64;
    row[i0] * (1.0 - frac) + row[i0 + 1] * frac
}

/// Coupling field on the grid: `coup[i][k] = Σ_j W[i][j] ·
/// V̈_j(t_k − delay_ij)`.
#[must_use]
pub fn coupling_field(
    vddot_prev: &[Vec<f64>],
    weights: &[Vec<f64>],
    delays: &[Vec<f64>],
    dt: f64,
) -> Vec<Vec<f64>> {
    let n = weights.len();
    let nsteps = if n == 0 { 0 } else { vddot_prev[0].len() };
    let mut coup = vec![vec![0.0; nsteps]; n];
    for i in 0..n {
        for (k, slot) in coup[i].iter_mut().enumerate() {
            let t = k as f64 * dt;
            let mut sum = 0.0;
            for j in 0..n {
                if weights[i][j] != 0.0 {
                    sum += weights[i][j] * interp_row(&vddot_prev[j], dt, t - delays[i][j]);
                }
            }
            *slot = sum;
        }
    }
    coup
}

/// Centred-difference derivative of each row on the grid
/// (one-sided at the ends) — `np.gradient` semantics, as in the
/// Python reference.
fn grid_gradient(rows: &[Vec<f64>], dt: f64) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            let n = row.len();
            let mut out = vec![0.0; n];
            if n < 2 {
                return out;
            }
            out[0] = (row[1] - row[0]) / dt;
            out[n - 1] = (row[n - 1] - row[n - 2]) / dt;
            for k in 1..n - 1 {
                out[k] = (row[k + 1] - row[k - 1]) / (2.0 * dt);
            }
            out
        })
        .collect()
}

/// Linear interpolation of a grid row at fractional index `x`.
fn interp_grid(row: &[f64], x: f64) -> f64 {
    if row.is_empty() {
        return 0.0;
    }
    let i0 = x.floor().max(0.0) as usize;
    if i0 >= row.len() - 1 {
        return row[row.len() - 1];
    }
    let frac = x - i0 as f64;
    row[i0] * (1.0 - frac) + row[i0 + 1] * frac
}

/// One bubble's KM acceleration at a trial state under the grid
/// drive, through the existing model (no KM duplication).
fn accel<M: AdaptiveBubbleModel>(
    model: &M,
    params: &BubbleParameters,
    radius: f64,
    velocity: f64,
    p_ac: f64,
    dp_dt: f64,
    t: f64,
) -> KwaversResult<f64> {
    let mut state = BubbleState::new(params);
    state.radius = radius;
    state.wall_velocity = velocity;
    model.calculate_acceleration(
        &mut state,
        InstantaneousPressure::from_pascals(p_ac),
        PressureRate::from_base(dp_dt),
        t,
    )
}

/// Integrate one Picard pass for the whole cloud under the given
/// per-bubble total-drive grids. RK4 in the Python reference's
/// stage arrangement. Returns `(R, V, V̈)` histories per bubble.
#[allow(clippy::type_complexity)]
fn integrate_pass<M: AdaptiveBubbleModel>(
    models: &[M],
    bubbles: &[CoupledBubble],
    drive_p: &[Vec<f64>],
    drive_dp: &[Vec<f64>],
    dt: f64,
) -> KwaversResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let nb = bubbles.len();
    let nsteps = drive_p[0].len() - 1;
    let mut radius = vec![vec![0.0; nsteps + 1]; nb];
    let mut velocity = vec![vec![0.0; nsteps + 1]; nb];
    for (i, bubble) in bubbles.iter().enumerate() {
        let model = &models[i];
        let params = model.params();
        radius[i][0] = bubble.radius0_m;
        for k in 0..nsteps {
            let rk = radius[i][k];
            let vk = velocity[i][k];
            let drive = |x: f64| (interp_grid(&drive_p[i], x), interp_grid(&drive_dp[i], x));
            let (p, dp) = drive(k as f64);
            let a1 = accel(model, params, rk, vk, p, dp, k as f64 * dt)?;
            let (p, dp) = drive(k as f64 + 0.5);
            let a2 = accel(
                model,
                params,
                rk + 0.5 * dt * vk,
                vk + 0.5 * dt * a1,
                p,
                dp,
                (k as f64 + 0.5) * dt,
            )?;
            let a3 = accel(
                model,
                params,
                rk + 0.5 * dt * (vk + 0.5 * dt * a1),
                vk + 0.5 * dt * a2,
                p,
                dp,
                (k as f64 + 0.5) * dt,
            )?;
            let (p, dp) = drive((k + 1) as f64);
            let a4 = accel(
                model,
                params,
                rk + dt * (vk + 0.5 * dt * a2),
                vk + dt * a3,
                p,
                dp,
                (k + 1) as f64 * dt,
            )?;
            velocity[i][k + 1] = vk + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
            radius[i][k + 1] = rk + dt * vk + dt * dt / 6.0 * (a1 + a2 + a3);
        }
    }
    // Volume acceleration on the grid from the grid drive:
    // V̈ = 4π(2RV̇² + R²R̈).
    let mut vddot = vec![vec![0.0; nsteps + 1]; nb];
    for i in 0..nb {
        let model = &models[i];
        let params = model.params();
        for k in 0..=nsteps {
            let a = accel(
                model,
                params,
                radius[i][k],
                velocity[i][k],
                drive_p[i][k],
                drive_dp[i][k],
                k as f64 * dt,
            )?;
            let r = radius[i][k];
            let v = velocity[i][k];
            vddot[i][k] = 4.0 * PI * (2.0 * r * v * v + r * r * a);
        }
    }
    Ok((radius, velocity, vddot))
}

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
    /// Radius histories [m], per bubble, on the grid.
    pub radius: Vec<Vec<f64>>,
    /// Wall-velocity histories [m/s], per bubble.
    pub velocity: Vec<Vec<f64>>,
    /// Volume-acceleration histories `V̈` [m³/s²], per bubble
    /// (the final pass's output; the coupling source term).
    pub volume_accel: Vec<Vec<f64>>,
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
    let band_power = |vddot: &[Vec<f64>]| -> f64 {
        vddot
            .iter()
            .map(|row| segment_band_power(row, dt, seg_start, config.density_kg_m3))
            .sum()
    };

    // Pass 0: independent driving.
    let zero_hist = vec![vec![0.0; nsteps + 1]; nb];
    let run_pass = |input_hist: &[Vec<f64>]| {
        let coup = coupling_field(input_hist, &weights, &delays, dt);
        let mut drive_p = vec![vec![0.0; nsteps + 1]; nb];
        for i in 0..nb {
            for k in 0..=nsteps {
                drive_p[i][k] = incident[k] + coup[i][k];
            }
        }
        let drive_dp = grid_gradient(&drive_p, dt);
        integrate_pass(models, bubbles, &drive_p, &drive_dp, dt)
    };

    let (mut radius, mut velocity, mut vddot) = run_pass(&zero_hist)?;
    let mut input_hist = vddot.clone();
    let mut prev_power = band_power(&vddot);
    let mut changes = Vec::new();
    for _ in 0..max_passes {
        let (r, v, vd) = run_pass(&input_hist)?;
        let power = band_power(&vd);
        changes.push(10.0 * (power / prev_power).log10());
        prev_power = power;
        // Damped history for the next pass (exact: the coupling
        // field is linear in V̈).
        let theta = config.damping_theta;
        for i in 0..nb {
            for k in 0..=nsteps {
                input_hist[i][k] = (1.0 - theta) * input_hist[i][k] + theta * vd[i][k];
            }
        }
        radius = r;
        velocity = v;
        vddot = vd;
    }
    Ok(CoupledSolution {
        radius,
        velocity,
        volume_accel: vddot,
        pass_band_power_change_db: changes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;

    /// WS-B medium (km_coupled.py constants): ρ = 998, σ = 0.0725,
    /// μ = 1e-3, pv = 2.34e3, c_KM = 1481, p0 = 1.013e5, γ = 1.4
    /// (air), thermal/mass-transfer off (the Python closure).
    fn imaging_params(r0: f64) -> BubbleParameters {
        let mut p = BubbleParameters::default();
        p.r0 = r0;
        p.p0 = 1.013e5;
        p.rho_liquid = 998.0;
        p.c_liquid = 1481.0;
        p.mu_liquid = 1.0e-3;
        p.sigma = 0.0725;
        p.pv = 2.34e3;
        p.use_thermal_effects = false;
        p.use_mass_transfer = false;
        p.driving_frequency = COUPLED_VALIDATED_FREQUENCY_HZ;
        p
    }

    fn tone_history(n: usize, dt: f64, amp: f64) -> Vec<f64> {
        let omega = 2.0 * PI * COUPLED_VALIDATED_FREQUENCY_HZ;
        (0..=n)
            .map(|k| amp * (omega * k as f64 * dt).sin())
            .collect()
    }

    fn test_config() -> CoupledCloudConfig {
        CoupledCloudConfig {
            steps_per_cycle: 500,
            ..CoupledCloudConfig::default()
        }
    }

    fn cloud_band_power_db(sol: &CoupledSolution, config: &CoupledCloudConfig) -> f64 {
        let dt = 1.0 / (COUPLED_VALIDATED_FREQUENCY_HZ * config.steps_per_cycle as f64);
        let seg = config.segment_start_cycle * config.steps_per_cycle;
        let total: f64 = sol
            .volume_accel
            .iter()
            .map(|row| segment_band_power(row, dt, seg, config.density_kg_m3))
            .sum();
        10.0 * total.log10()
    }

    #[test]
    fn pair_geometry_weights_and_delays() {
        let pos = [[0.0, 0.0, 0.0], [1.0e-3, 0.0, 0.0]];
        let (w, d) = pair_geometry(&pos, 998.0, 1540.0);
        assert_eq!(w[0][0], 0.0);
        assert_eq!(d[1][1], 0.0);
        let expected_w = 998.0 / (4.0 * PI * 1.0e-3);
        assert!((w[0][1] - expected_w).abs() / expected_w < 1e-12);
        assert!((d[0][1] - 1.0e-3 / 1540.0).abs() < 1e-15);
        assert_eq!(w[0][1], w[1][0]);
    }

    /// WS-B dilute-limit validation, self-consistent form: at
    /// 5× the cloud radius (density ×0.008 of the bracket scale)
    /// the coupled solve must recover independent driving with
    /// the 1.5f₀ band power within ≈ 0.2 dB (Python: −0.11 dB).
    #[test]
    fn dilute_limit_recovers_independent_driving() {
        let config = test_config();
        let n = config.n_cycles * config.steps_per_cycle;
        let dt = 1.0 / (COUPLED_VALIDATED_FREQUENCY_HZ * config.steps_per_cycle as f64);
        let incident = tone_history(n, dt, 0.268e6);
        // Three bubbles ≥ 7 mm apart (cf. WS-B dilute radius ×5).
        let bubbles = vec![
            CoupledBubble {
                radius0_m: 2.5e-6,
                position_m: [0.0, 0.0, 0.0],
            },
            CoupledBubble {
                radius0_m: 2.5e-6,
                position_m: [7.0e-3, 1.0e-3, 0.0],
            },
            CoupledBubble {
                radius0_m: 2.5e-6,
                position_m: [1.0e-3, 7.0e-3, 2.0e-3],
            },
        ];
        let models: Vec<KellerMiksisModel> = bubbles
            .iter()
            .map(|b| KellerMiksisModel::new(imaging_params(b.radius0_m)))
            .collect();
        let independent = solve_coupled_cloud(&models, &bubbles, &incident, &config, 0).unwrap();
        let coupled = solve_coupled_cloud(&models, &bubbles, &incident, &config, 3).unwrap();
        let delta_db =
            cloud_band_power_db(&coupled, &config) - cloud_band_power_db(&independent, &config);
        assert!(
            delta_db.abs() < 0.2,
            "dilute coupled−independent band power {delta_db} dB (WS-B: −0.11 dB)"
        );
    }

    /// Two identical bubbles at mirror positions under a spatially
    /// uniform incident tone must follow identical trajectories.
    #[test]
    fn two_identical_bubbles_are_symmetric() {
        let config = test_config();
        let n = config.n_cycles * config.steps_per_cycle;
        let dt = 1.0 / (COUPLED_VALIDATED_FREQUENCY_HZ * config.steps_per_cycle as f64);
        let incident = tone_history(n, dt, 0.268e6);
        let bubbles = vec![
            CoupledBubble {
                radius0_m: 2.5e-6,
                position_m: [-25.0e-6, 0.0, 0.0],
            },
            CoupledBubble {
                radius0_m: 2.5e-6,
                position_m: [25.0e-6, 0.0, 0.0],
            },
        ];
        let models: Vec<KellerMiksisModel> = bubbles
            .iter()
            .map(|b| KellerMiksisModel::new(imaging_params(b.radius0_m)))
            .collect();
        let sol = solve_coupled_cloud(&models, &bubbles, &incident, &config, 2).unwrap();
        let max_diff = sol.radius[0]
            .iter()
            .zip(sol.radius[1].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-12, "symmetry broken: max ΔR {max_diff} m");
        // And the coupling must actually be active at 50 µm:
        // coupled trajectories differ from independent ones.
        let independent = solve_coupled_cloud(&models, &bubbles, &incident, &config, 0).unwrap();
        let effect = sol.radius[0]
            .iter()
            .zip(independent.radius[0].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(effect > 1e-9, "coupling inactive at 50 µm: ΔR {effect} m");
    }

    /// Damped Picard converges to a fixed point on a close pair
    /// (the configuration where plain Picard oscillates, WS-B):
    /// successive band-power changes shrink and the last is small.
    #[test]
    fn damped_picard_converges_to_fixed_point() {
        let config = test_config();
        let n = config.n_cycles * config.steps_per_cycle;
        let dt = 1.0 / (COUPLED_VALIDATED_FREQUENCY_HZ * config.steps_per_cycle as f64);
        let incident = tone_history(n, dt, 0.123e6);
        let bubbles = vec![
            CoupledBubble {
                radius0_m: 2.5e-6,
                position_m: [0.0, 0.0, 0.0],
            },
            CoupledBubble {
                radius0_m: 2.5e-6,
                position_m: [50.0e-6, 0.0, 0.0],
            },
        ];
        let models: Vec<KellerMiksisModel> = bubbles
            .iter()
            .map(|b| KellerMiksisModel::new(imaging_params(b.radius0_m)))
            .collect();
        let sol = solve_coupled_cloud(&models, &bubbles, &incident, &config, 5).unwrap();
        let changes = &sol.pass_band_power_change_db;
        assert_eq!(changes.len(), 5);
        let first = changes[0].abs();
        let last = changes[changes.len() - 1].abs();
        assert!(
            last < first && last < 0.5,
            "Picard not converging: changes {changes:?} dB"
        );
    }

    #[test]
    fn regime_guard_flags_trigger_pressures() {
        assert!(coupled_regime_reasons(2.45e6, 0.26e6, 32.0).is_empty());
        assert!(coupled_regime_reasons(2.45e6, 1.27e6, 128.0).is_empty());
        // Trigger pressures: outside, by construction.
        assert!(!coupled_regime_reasons(2.45e6, 20.0e6, 32.0).is_empty());
        assert!(!coupled_regime_reasons(1.0e6, 0.26e6, 32.0).is_empty());
        assert!(!coupled_regime_reasons(2.45e6, 0.26e6, 500.0).is_empty());
    }
}
