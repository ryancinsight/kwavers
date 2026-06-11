//! Sonogenetics physics for book chapter ch18.
//!
//! Covers: Hill activation probability, acoustic radiation force,
//! acoustic streaming velocity, and ISPTA calculation.

// ─── Hill activation ──────────────────────────────────────────────────────────

/// Hill equation activation probability for mechanosensitive channels.
///
/// Models the probability of channel activation as a function of acoustic
/// pressure amplitude:
/// ```text
/// p(P) = P^n / (P^n + P_thresh^n)   ∈ [0, 1]
/// ```
///
/// # Arguments
/// * `pressure_arr` – acoustic pressure amplitudes [Pa]
/// * `p_threshold_pa` – half-activation threshold pressure [Pa]
/// * `hill_n` – Hill coefficient (cooperativity exponent)
///
/// # Reference
/// Ibsen et al. (2015), *Nat. Commun.* 6, 8264.
#[must_use]
pub fn hill_activation_probability(
    pressure_arr: &[f64],
    p_threshold_pa: f64,
    hill_n: f64,
) -> Vec<f64> {
    if !positive_finite(p_threshold_pa) || !positive_finite(hill_n) {
        return vec![0.0; pressure_arr.len()];
    }

    let pt_n = p_threshold_pa.powf(hill_n);
    if !positive_finite(pt_n) {
        return vec![0.0; pressure_arr.len()];
    }

    pressure_arr
        .iter()
        .map(|&p| {
            if !p.is_finite() {
                return 0.0;
            }

            let pn = p.abs().powf(hill_n);
            let denominator = pn + pt_n;
            if pn.is_finite() && positive_finite(denominator) {
                pn / denominator
            } else {
                0.0
            }
        })
        .collect()
}

// ─── Acoustic radiation force ─────────────────────────────────────────────────

/// Acoustic radiation force density from a travelling plane wave.
///
/// ```text
/// F = 2·α·I / c   [N/m³]
/// ```
/// where I = intensity [W/m²] at each spatial position, α is the absorption
/// coefficient [Np/m], and c is sound speed.
///
/// # Arguments
/// * `intensity_w_m2` – intensity profile [W/m²]
/// * `alpha_np_m` – absorption coefficient [Np/m]
/// * `c` – sound speed [m/s]
///
/// # Reference
/// Nyborg (1965), *Physical Acoustics* Vol. 2, ch. 11.
#[must_use]
#[inline]
pub fn radiation_force_1d(intensity_w_m2: &[f64], alpha_np_m: f64, c: f64) -> Vec<f64> {
    if !nonnegative_finite(alpha_np_m) || !positive_finite(c) {
        return vec![0.0; intensity_w_m2.len()];
    }

    let scale = 2.0 * alpha_np_m / c;
    intensity_w_m2
        .iter()
        .map(|&i| {
            if nonnegative_finite(i) {
                scale * i
            } else {
                0.0
            }
        })
        .collect()
}

// ─── Acoustic streaming ───────────────────────────────────────────────────────

/// Acoustic streaming (Eckart) velocity in an absorbing fluid column.
///
/// ```text
/// u_s = α·I·L² / (2·μ·c)   [m/s]
/// ```
/// This is the Eckart (1948) approximation for a cylindrical fluid column of
/// length L driven by a Gaussian beam with intensity I.
///
/// # Arguments
/// * `i_w_m2` – beam intensity [W/m²]
/// * `mu_pa_s` – dynamic viscosity [Pa·s]
/// * `alpha_np_m` – absorption coefficient [Np/m]
/// * `c` – sound speed [m/s]
/// * `l_m` – streaming path length [m]
///
/// # Reference
/// Eckart (1948), *Phys. Rev.* 73, 68.
#[must_use]
#[inline]
pub fn acoustic_streaming_velocity(
    i_w_m2: f64,
    mu_pa_s: f64,
    alpha_np_m: f64,
    c: f64,
    l_m: f64,
) -> f64 {
    if !nonnegative_finite(i_w_m2)
        || !positive_finite(mu_pa_s)
        || !nonnegative_finite(alpha_np_m)
        || !positive_finite(c)
        || !nonnegative_finite(l_m)
    {
        return 0.0;
    }

    alpha_np_m * i_w_m2 * l_m * l_m / (2.0 * mu_pa_s * c)
}

// ─── Safety metric ────────────────────────────────────────────────────────────

/// Spatial-peak time-average intensity (ISPTA) from a pressure waveform.
///
/// ```text
/// ISPTA = (1/T) · ∫ p²(t) dt / (ρ·c)   [W/m²]  →  converted to W/cm²
/// ```
/// Integrated by the rectangle rule.
///
/// # Arguments
/// * `p_pa` – pressure waveform at the spatial peak [Pa]
/// * `dt_s` – time step [s]
/// * `rho` – density [kg/m³]
/// * `c` – sound speed [m/s]
///
/// Returns ISPTA in W/cm².
///
/// # Reference
/// NCRP Report 74 (1983), §4.
#[must_use]
pub fn ispta_w_cm2(p_pa: &[f64], dt_s: f64, rho: f64, c: f64) -> f64 {
    if p_pa.is_empty() || !positive_finite(dt_s) || !positive_finite(rho) || !positive_finite(c) {
        return 0.0;
    }

    let n = p_pa.len() as f64;
    let total_time = n * dt_s;
    let integral: f64 = p_pa
        .iter()
        .map(|&p| if p.is_finite() { p * p * dt_s } else { 0.0 })
        .sum();
    let ispta_w_m2 = integral / (rho * c * total_time);
    if ispta_w_m2.is_finite() {
        ispta_w_m2 * 1e-4 // convert W/m² → W/cm²
    } else {
        0.0
    }
}

#[inline]
fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[inline]
fn nonnegative_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

// ─── Acoustic gene-expression kinetics ──────────────────────────────────────────

/// State of the two-stage gene-expression cascade: transcript (mRNA) and
/// expressed protein levels (arbitrary concentration units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpressionState {
    /// Transcript (mRNA) level.
    pub mrna: f64,
    /// Expressed-protein level.
    pub protein: f64,
}

impl ExpressionState {
    /// The quiescent state (no transcript, no protein).
    #[must_use]
    pub fn zero() -> Self {
        Self {
            mrna: 0.0,
            protein: 0.0,
        }
    }
}

/// Linear two-stage **gene-expression kinetics** for acoustic gene therapy
/// (§17.13.2) — the central-dogma / PK–PD cascade driven by the acoustic
/// channel-activation level `a(t) ∈ [0, 1]` (e.g. [`hill_activation_probability`]):
///
/// ```text
/// dm/dt = β·a − δ_m·m     (transcription ∝ activation; mRNA decays at δ_m)
/// dp/dt = κ·m − δ_p·p     (translation ∝ mRNA; protein decays at δ_p)
/// ```
///
/// This couples the (already-modelled) acoustic activation to downstream
/// expression: feed an activation time series (the field-driven open
/// probability over a pulse train) through [`integrate`](Self::integrate). The
/// model is the standard linear two-compartment cascade (Alon 2006), so its
/// steady state and transient are closed-form — used here for verification.
#[derive(Debug, Clone, Copy)]
pub struct GeneExpressionKinetics {
    transcription_rate: f64, // β
    mrna_decay: f64,         // δ_m
    translation_rate: f64,   // κ
    protein_decay: f64,      // δ_p
}

impl GeneExpressionKinetics {
    /// Construct from the four positive rate constants
    /// `(β, δ_m, κ, δ_p)`. Returns `None` unless all are finite and positive.
    #[must_use]
    pub fn new(
        transcription_rate: f64,
        mrna_decay: f64,
        translation_rate: f64,
        protein_decay: f64,
    ) -> Option<Self> {
        if positive_finite(transcription_rate)
            && positive_finite(mrna_decay)
            && positive_finite(translation_rate)
            && positive_finite(protein_decay)
        {
            Some(Self {
                transcription_rate,
                mrna_decay,
                translation_rate,
                protein_decay,
            })
        } else {
            None
        }
    }

    /// Right-hand side `(dm/dt, dp/dt)` at `state` for activation `a`.
    #[inline]
    #[must_use]
    fn derivative(&self, state: ExpressionState, activation: f64) -> (f64, f64) {
        let dm = self
            .transcription_rate
            .mul_add(activation, -self.mrna_decay * state.mrna);
        let dp = self
            .translation_rate
            .mul_add(state.mrna, -self.protein_decay * state.protein);
        (dm, dp)
    }

    /// Advance the state by one RK4 step `dt` under a constant activation `a`.
    #[must_use]
    pub fn step_rk4(&self, state: ExpressionState, activation: f64, dt: f64) -> ExpressionState {
        let add = |s: ExpressionState, k: (f64, f64), h: f64| ExpressionState {
            mrna: k.0.mul_add(h, s.mrna),
            protein: k.1.mul_add(h, s.protein),
        };
        let k1 = self.derivative(state, activation);
        let k2 = self.derivative(add(state, k1, 0.5 * dt), activation);
        let k3 = self.derivative(add(state, k2, 0.5 * dt), activation);
        let k4 = self.derivative(add(state, k3, dt), activation);
        ExpressionState {
            mrna: (dt / 6.0).mul_add(k4.0 + 2.0 * (k2.0 + k3.0) + k1.0, state.mrna),
            protein: (dt / 6.0).mul_add(k4.1 + 2.0 * (k2.1 + k3.1) + k1.1, state.protein),
        }
    }

    /// Closed-form steady state under sustained activation `a`:
    /// `m_ss = β·a/δ_m`, `p_ss = κ·m_ss/δ_p = κβa/(δ_m δ_p)`.
    #[must_use]
    pub fn steady_state(&self, activation: f64) -> ExpressionState {
        let mrna = self.transcription_rate * activation / self.mrna_decay;
        let protein = self.translation_rate * mrna / self.protein_decay;
        ExpressionState { mrna, protein }
    }

    /// Integrate the cascade from the quiescent state over an activation time
    /// series (one sample per step `dt`), returning the per-step state. Drive
    /// this with the field-derived activation (e.g. the open probability over a
    /// neuromodulation pulse train). Non-finite activations are treated as `0`.
    #[must_use]
    pub fn integrate(&self, activation: &[f64], dt: f64) -> Vec<ExpressionState> {
        let mut state = ExpressionState::zero();
        let mut out = Vec::with_capacity(activation.len());
        for &a in activation {
            let a = if a.is_finite() { a } else { 0.0 };
            state = self.step_rk4(state, a, dt);
            out.push(state);
        }
        out
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

    #[test]
    fn hill_at_threshold_is_half() {
        let p = hill_activation_probability(&[100.0], 100.0, 2.0);
        assert!((p[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn hill_zero_pressure_is_zero() {
        let p = hill_activation_probability(&[0.0], 100.0, 2.0);
        assert!((p[0]).abs() < 1e-15);
    }

    #[test]
    fn hill_saturates_at_high_pressure() {
        let p = hill_activation_probability(&[1e10], 100.0, 2.0);
        assert!((p[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn hill_rejects_invalid_domains_and_samples() {
        let invalid_threshold = hill_activation_probability(&[100.0], 0.0, 2.0);
        assert_eq!(invalid_threshold, vec![0.0]);

        let invalid_hill = hill_activation_probability(&[100.0], 100.0, 0.0);
        assert_eq!(invalid_hill, vec![0.0]);

        let nonfinite_pressure = hill_activation_probability(&[f64::NAN, 100.0], 100.0, 2.0);
        assert_eq!(nonfinite_pressure[0], 0.0);
        assert!((nonfinite_pressure[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn radiation_force_proportional_to_intensity() {
        let f = radiation_force_1d(&[1.0, 2.0, 4.0], 1.0, SOUND_SPEED_WATER_SIM);
        assert!((f[1] / f[0] - 2.0).abs() < 1e-10);
        assert!((f[2] / f[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn radiation_force_rejects_negative_or_nonfinite_domains() {
        assert_eq!(
            radiation_force_1d(&[1.0, 2.0], -1.0, SOUND_SPEED_WATER_SIM),
            vec![0.0, 0.0]
        );
        assert_eq!(radiation_force_1d(&[1.0], 1.0, 0.0), vec![0.0]);

        let f = radiation_force_1d(&[1.0, -2.0, f64::INFINITY], 1.0, SOUND_SPEED_WATER_SIM);
        assert!((f[0] - 2.0 / SOUND_SPEED_WATER_SIM).abs() < 1e-15);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 0.0);
    }

    #[test]
    fn streaming_velocity_positive() {
        let v = acoustic_streaming_velocity(
            DENSITY_WATER_NOMINAL,
            1e-3,
            0.5,
            SOUND_SPEED_WATER_SIM,
            0.05,
        );
        assert!(v > 0.0);
    }

    #[test]
    fn streaming_velocity_rejects_invalid_domains() {
        assert_eq!(
            acoustic_streaming_velocity(-1.0, 1e-3, 0.5, SOUND_SPEED_WATER_SIM, 0.05),
            0.0
        );
        assert_eq!(
            acoustic_streaming_velocity(1.0, 0.0, 0.5, SOUND_SPEED_WATER_SIM, 0.05),
            0.0
        );
        assert_eq!(
            acoustic_streaming_velocity(1.0, 1e-3, -0.5, SOUND_SPEED_WATER_SIM, 0.05),
            0.0
        );
        assert_eq!(acoustic_streaming_velocity(1.0, 1e-3, 0.5, 0.0, 0.05), 0.0);
        assert_eq!(
            acoustic_streaming_velocity(1.0, 1e-3, 0.5, SOUND_SPEED_WATER_SIM, f64::NAN),
            0.0
        );
    }

    #[test]
    fn ispta_constant_pressure() {
        // Constant pressure p0 → ISPTA = p0²/(rho*c) in W/m² → W/cm²
        let p0 = 1e5_f64;
        let rho = DENSITY_WATER_NOMINAL;
        let c = SOUND_SPEED_WATER_SIM;
        let n = 1000;
        let dt = 1e-7_f64;
        let p = vec![p0; n];
        let ispta = ispta_w_cm2(&p, dt, rho, c);
        let expected = p0 * p0 / (rho * c) * 1e-4;
        assert!(
            (ispta - expected).abs() / expected < 1e-10,
            "got={} expected={}",
            ispta,
            expected
        );
    }

    #[test]
    fn ispta_rejects_empty_invalid_and_nonfinite_domains() {
        assert_eq!(
            ispta_w_cm2(&[], 1e-7, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM),
            0.0
        );
        assert_eq!(
            ispta_w_cm2(&[1.0], 0.0, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM),
            0.0
        );
        assert_eq!(ispta_w_cm2(&[1.0], 1e-7, 0.0, SOUND_SPEED_WATER_SIM), 0.0);
        assert_eq!(
            ispta_w_cm2(&[1.0], 1e-7, DENSITY_WATER_NOMINAL, -SOUND_SPEED_WATER_SIM),
            0.0
        );

        let ispta = ispta_w_cm2(&[1.0, f64::NAN, 1.0], 1.0, 1.0, 1.0);
        assert!((ispta - (2.0 / 3.0) * 1e-4).abs() < 1e-16);
    }

    // ── Gene-expression kinetics ────────────────────────────────────────────

    fn kinetics() -> GeneExpressionKinetics {
        // β=2.0, δ_m=0.5, κ=1.5, δ_p=0.1 (mRNA fast, protein slow — typical).
        GeneExpressionKinetics::new(2.0, 0.5, 1.5, 0.1).unwrap()
    }

    #[test]
    fn gene_expression_rejects_invalid_rates() {
        assert!(GeneExpressionKinetics::new(0.0, 0.5, 1.5, 0.1).is_none());
        assert!(GeneExpressionKinetics::new(2.0, -0.5, 1.5, 0.1).is_none());
        assert!(GeneExpressionKinetics::new(2.0, 0.5, 1.5, f64::NAN).is_none());
    }

    #[test]
    fn gene_expression_integrates_to_closed_form_steady_state() {
        let k = kinetics();
        let a = 0.8; // sustained activation
                     // Drive long enough to reach steady state (protein τ = 1/δ_p = 10 s).
        let dt = 0.01;
        let n = (200.0 / dt) as usize; // 200 s ≫ 10 s
        let series = vec![a; n];
        let traj = k.integrate(&series, dt);
        let last = *traj.last().unwrap();

        let ss = k.steady_state(a);
        // Closed form: m_ss = βa/δ_m = 2·0.8/0.5 = 3.2; p_ss = κ m_ss/δ_p = 1.5·3.2/0.1 = 48.
        assert!((ss.mrna - 3.2).abs() < 1e-12);
        assert!((ss.protein - 48.0).abs() < 1e-12);
        assert!(
            (last.mrna - ss.mrna).abs() < 1e-4 * ss.mrna
                && (last.protein - ss.protein).abs() < 1e-4 * ss.protein,
            "integrated state {last:?} must reach steady state {ss:?}"
        );
    }

    #[test]
    fn gene_expression_transcript_matches_analytic_rise() {
        // Under constant activation from rest, m(t) = m_ss·(1 − e^{−δ_m t}).
        let k = kinetics();
        let a = 1.0;
        let dt = 0.005;
        let n = 2000; // 10 s
        let traj = k.integrate(&vec![a; n], dt);
        let m_ss = k.steady_state(a).mrna; // 2·1/0.5 = 4.0
        for (i, st) in traj.iter().enumerate() {
            let t = (i + 1) as f64 * dt;
            let analytic = m_ss * (1.0 - (-k_mrna_decay() * t).exp());
            assert!(
                (st.mrna - analytic).abs() < 1e-3 * m_ss,
                "mRNA at t={t}: {} vs analytic {analytic}",
                st.mrna
            );
        }
    }

    fn k_mrna_decay() -> f64 {
        0.5
    }

    #[test]
    fn gene_expression_is_linear_in_activation_and_decays_to_zero() {
        let k = kinetics();
        // Linearity: doubling sustained activation doubles steady protein.
        let p1 = k.steady_state(0.3).protein;
        let p2 = k.steady_state(0.6).protein;
        assert!((p2 / p1 - 2.0).abs() < 1e-12);

        // After a pulse then silence, expression decays back toward zero.
        let dt = 0.05;
        let on = vec![1.0; 400]; // 20 s on
        let off = vec![0.0; 4000]; // 200 s off
        let mut series = on;
        series.extend(off);
        let traj = k.integrate(&series, dt);
        let peak = traj[399].protein;
        let end = traj.last().unwrap().protein;
        assert!(
            end < 0.05 * peak,
            "expression must wash out: {end} vs peak {peak}"
        );
    }
}
