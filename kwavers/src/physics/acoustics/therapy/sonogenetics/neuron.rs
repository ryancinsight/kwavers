//! Leaky integrate-and-fire (LIF) neuron model driven by mechanosensitive ion current.
//!
//! # Model
//!
//! The membrane equation (Lapicque 1907; Abbott 1999):
//!
//!   C_m · dV/dt = −G_leak · (V − E_leak) + I_ion(t)
//!
//! Forward-Euler discretisation:
//!
//!   V[n+1] = V[n] + dt/C_m · (−G_leak · (V[n] − E_leak) + I_ion[n])
//!
//! Spike condition: if V[n+1] ≥ V_thresh → emit spike, reset V ← V_reset.
//! Refractory period: for `refractory_s` seconds after a spike, V is clamped
//! to V_reset and no further spikes can occur.
//!
//! # Canonical mammalian neuron parameters (Koch 1999)
//!
//! | Parameter       | Symbol  | Value    |
//! |-----------------|---------|----------|
//! | Capacitance     | C_m     | 100 pF   |
//! | Leak conductance| G_leak  | 10 nS    |
//! | Time constant   | τ_m     | 10 ms    |
//! | Leak reversal   | E_leak  | −65 mV   |
//! | Spike threshold | V_thresh| −55 mV   |
//! | Reset voltage   | V_reset | −65 mV   |
//! | Refractory      | τ_ref   | 2 ms     |
//!
//! # References
//!
//! - Lapicque, L. (1907). Recherches quantitatives sur l'excitation électrique des nerfs.
//!   *J. Physiol. Pathol. Gen.*, 9, 620-635.
//! - Abbott, L.F. (1999). Lapicque's introduction of the integrate-and-fire model neuron.
//!   *Brain Research Bulletin*, 50(5-6), 303-304.
//! - Koch, C. (1999). *Biophysics of Computation*. Oxford University Press.

use crate::core::error::{KwaversError, KwaversResult, ValidationError};

// ─────────────────────────────────────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Leaky integrate-and-fire neuron parameters.
///
/// All values in SI units (F, S, V, s).
#[derive(Debug, Clone)]
pub struct LifParams {
    /// Membrane capacitance C_m [F].
    pub capacitance_f: f64,
    /// Leak conductance G_leak [S].
    pub leak_conductance_s: f64,
    /// Leak reversal potential E_leak [V].
    pub leak_reversal_v: f64,
    /// Spike threshold V_thresh [V].
    pub threshold_v: f64,
    /// Reset voltage V_reset [V].
    pub reset_v: f64,
    /// Absolute refractory period [s].
    pub refractory_s: f64,
}

impl Default for LifParams {
    /// Canonical mammalian neuron soma parameters (Koch 1999 Table 1.1).
    ///
    /// τ_m = C_m / G_leak = 100e-12 / 10e-9 = 10 ms.
    fn default() -> Self {
        Self {
            capacitance_f: 100.0e-12,    // 100 pF
            leak_conductance_s: 10.0e-9, // 10 nS → τ_m = 10 ms
            leak_reversal_v: -65.0e-3,   // −65 mV
            threshold_v: -55.0e-3,       // −55 mV (10 mV above rest)
            reset_v: -65.0e-3,           // −65 mV
            refractory_s: 2.0e-3,        // 2 ms
        }
    }
}

impl LifParams {
    /// Membrane time constant τ_m = C_m / G_leak [s].
    #[inline]
    #[must_use]
    pub fn time_constant_s(&self) -> f64 {
        self.capacitance_f / self.leak_conductance_s
    }

    /// Returns `true` if all parameters are physically consistent.
    ///
    /// Invariants:
    /// - C_m > 0
    /// - G_leak > 0
    /// - V_reset < V_thresh (threshold above reset)
    /// - refractory ≥ 0
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.capacitance_f > 0.0
            && self.leak_conductance_s > 0.0
            && self.reset_v < self.threshold_v
            && self.refractory_s >= 0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Neuron
// ─────────────────────────────────────────────────────────────────────────────

/// Single leaky integrate-and-fire neuron driven by an external current source.
///
/// Implements forward-Euler integration of the LIF membrane equation.
/// Spike times are recorded in chronological order. Voltage is clamped at
/// `V_reset` throughout the absolute refractory period.
#[derive(Debug, Clone)]
pub struct LifNeuron {
    params: LifParams,
    /// Current membrane voltage [V].
    membrane_voltage_v: f64,
    /// Time elapsed since last spike [s]. Starts at `+∞` to allow immediate spiking.
    time_since_spike_s: f64,
    /// Chronological spike times [s].
    spike_times_s: Vec<f64>,
}

impl LifNeuron {
    /// Construct a new LIF neuron with membrane voltage initialised to `E_leak`.
    ///
    /// # Panics (debug)
    ///
    /// Panics in debug builds if `params.is_valid()` is false.
    #[must_use]
    pub fn new(params: LifParams) -> Self {
        debug_assert!(params.is_valid(), "LifParams failed validity check");
        let v0 = params.leak_reversal_v;
        Self {
            params,
            membrane_voltage_v: v0,
            time_since_spike_s: f64::INFINITY,
            spike_times_s: Vec::new(),
        }
    }

    /// Advance the neuron by one time step using forward-Euler integration.
    ///
    /// # Arguments
    ///
    /// - `i_ion`  — total injected ion current [A]
    /// - `dt`     — time step [s]; must be > 0
    /// - `t_now`  — current simulation time [s]
    ///
    /// # Returns
    ///
    /// `Ok(true)` if a spike was emitted during this step; `Ok(false)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `dt ≤ 0`.
    ///
    /// # Correctness invariants
    ///
    /// - During the refractory period, V is clamped to `V_reset`; no spike is counted.
    /// - The spike is registered at `t_now` (the current simulation time, not the predicted
    ///   threshold-crossing time, consistent with forward-Euler convention).
    ///
    /// # Stability note
    ///
    /// Forward Euler is first-order. For accuracy, `dt ≪ τ_m = C_m / G_leak`. The
    /// canonical τ_m = 10 ms; dt = 0.1 ms gives a relative integration error O(dt/τ_m) = 1%.
    pub fn step(&mut self, i_ion: f64, dt: f64, t_now: f64) -> KwaversResult<bool> {
        if dt <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "dt".to_string(),
                value: dt,
                reason: "time step must be strictly positive".to_string(),
            }));
        }
        self.time_since_spike_s += dt;

        // Clamp during absolute refractory period — do not integrate.
        if self.time_since_spike_s < self.params.refractory_s {
            self.membrane_voltage_v = self.params.reset_v;
            return Ok(false);
        }

        // LIF forward-Euler:  V[n+1] = V[n] + dt/C_m · (-G_leak·(V[n] - E_leak) + I_ion)
        let p = &self.params;
        let dv = (dt / p.capacitance_f)
            * (-p.leak_conductance_s * (self.membrane_voltage_v - p.leak_reversal_v) + i_ion);
        self.membrane_voltage_v += dv;

        // Spike detection and reset.
        if self.membrane_voltage_v >= self.params.threshold_v {
            self.membrane_voltage_v = self.params.reset_v;
            self.time_since_spike_s = 0.0;
            self.spike_times_s.push(t_now);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Current membrane voltage [V].
    #[inline]
    #[must_use]
    pub fn membrane_voltage(&self) -> f64 {
        self.membrane_voltage_v
    }

    /// Recorded spike times in chronological order [s].
    #[inline]
    #[must_use]
    pub fn spike_times(&self) -> &[f64] {
        &self.spike_times_s
    }

    /// Total number of spikes emitted.
    #[inline]
    #[must_use]
    pub fn spike_count(&self) -> usize {
        self.spike_times_s.len()
    }

    /// Mean firing rate over a window of `duration_s` seconds [Hz].
    ///
    /// Returns 0.0 if `duration_s ≤ 0` or no spikes have been recorded.
    #[must_use]
    pub fn mean_firing_rate(&self, duration_s: f64) -> f64 {
        if duration_s <= 0.0 || self.spike_times_s.is_empty() {
            0.0
        } else {
            self.spike_times_s.len() as f64 / duration_s
        }
    }

    /// Membrane time constant τ_m = C_m / G_leak [s].
    #[inline]
    #[must_use]
    pub fn time_constant_s(&self) -> f64 {
        self.params.time_constant_s()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Sub-threshold constant current must not produce spikes.
    ///
    /// Steady-state voltage for constant current I:
    ///   V_ss = E_leak + I / G_leak
    /// Require: V_ss < V_thresh
    ///   I < G_leak · (V_thresh − E_leak) = 10e-9 × 10e-3 = 100 pA
    #[test]
    fn test_subthreshold_no_spike() {
        let params = LifParams::default();
        let mut neuron = LifNeuron::new(params.clone());
        // I = 50 pA < 100 pA threshold
        let i_ion = 50.0e-12_f64;
        let dt = 0.1e-3_f64; // 0.1 ms
        let n_steps = 500; // 50 ms >> τ_m = 10 ms
        let mut t = 0.0;
        for _ in 0..n_steps {
            let spiked = neuron.step(i_ion, dt, t).unwrap();
            assert!(
                !spiked,
                "no spike expected for sub-threshold current at t={t:.3e}"
            );
            t += dt;
        }
        // Steady-state: V = E_leak + I/G_leak
        let v_ss = params.leak_reversal_v + i_ion / params.leak_conductance_s;
        assert!(
            v_ss < params.threshold_v,
            "steady-state voltage {v_ss:.4e} must be below threshold {:.4e}",
            params.threshold_v
        );
        assert_relative_eq!(neuron.membrane_voltage(), v_ss, max_relative = 1e-3);
        assert_eq!(neuron.spike_count(), 0);
    }

    /// Supra-threshold constant current must produce spikes.
    ///
    /// I = 200 pA > 100 pA threshold → repetitive firing.
    #[test]
    fn test_suprathreshold_produces_spikes() {
        let params = LifParams::default();
        let mut neuron = LifNeuron::new(params);
        let i_ion = 200.0e-12_f64; // 2× threshold current
        let dt = 0.05e-3_f64; // 0.05 ms (well below τ_m = 10 ms)
        let duration = 100.0e-3_f64;
        let n_steps = (duration / dt) as usize;
        let mut t = 0.0;
        for _ in 0..n_steps {
            let _ = neuron.step(i_ion, dt, t).unwrap();
            t += dt;
        }
        assert!(
            neuron.spike_count() >= 3,
            "expected ≥3 spikes for I=200 pA over 100 ms, got {}",
            neuron.spike_count()
        );
    }

    /// After a spike, voltage must be at V_reset.
    ///
    /// Check: step until the first spike occurs, then immediately inspect the
    /// membrane voltage — it must equal V_reset.  Integration must not continue
    /// past this point because subsequent steps will re-charge the membrane
    /// (I = 1 nA >> threshold current).
    #[test]
    fn test_refractory_clamp() {
        let params = LifParams::default();
        let mut neuron = LifNeuron::new(params.clone());
        let i_large = 1.0e-9_f64; // 1 nA — drives to threshold within a few steps
        let dt = 0.1e-3_f64;
        let mut t = 0.0;
        let mut spiked_once = false;
        for _ in 0..200 {
            let spiked = neuron.step(i_large, dt, t).unwrap();
            t += dt;
            if spiked {
                spiked_once = true;
                // Immediately after the spike: voltage must be V_reset.
                assert_relative_eq!(
                    neuron.membrane_voltage(),
                    params.reset_v,
                    max_relative = 1e-9
                );
                break;
            }
        }
        assert!(
            spiked_once,
            "should have spiked with I = 1 nA over 200 steps"
        );
    }

    /// Zero time step returns an error.
    #[test]
    fn test_zero_dt_is_error() {
        let mut neuron = LifNeuron::new(LifParams::default());
        assert!(neuron.step(0.0, 0.0, 0.0).is_err());
        assert!(neuron.step(0.0, -1e-6, 0.0).is_err());
    }

    /// Mean firing rate is spike_count / duration.
    #[test]
    fn test_mean_firing_rate() {
        let params = LifParams::default();
        let mut neuron = LifNeuron::new(params);
        let i_ion = 200.0e-12_f64;
        let dt = 0.05e-3_f64;
        let duration = 100.0e-3_f64;
        let n_steps = (duration / dt) as usize;
        let mut t = 0.0;
        for _ in 0..n_steps {
            let _ = neuron.step(i_ion, dt, t).unwrap();
            t += dt;
        }
        let rate = neuron.mean_firing_rate(duration);
        let expected = neuron.spike_count() as f64 / duration;
        assert_relative_eq!(rate, expected, max_relative = 1e-12);
        // Zero or negative duration returns 0
        assert_eq!(neuron.mean_firing_rate(0.0), 0.0);
        assert_eq!(neuron.mean_firing_rate(-1.0), 0.0);
    }

    /// Membrane time constant equals C_m / G_leak.
    #[test]
    fn test_time_constant() {
        let params = LifParams::default();
        let tau = params.time_constant_s();
        assert_relative_eq!(tau, 10.0e-3, max_relative = 1e-12);
    }

    /// LifParams validity check.
    #[test]
    fn test_params_validity() {
        let valid = LifParams::default();
        assert!(valid.is_valid());
        let mut bad = LifParams::default();
        bad.capacitance_f = 0.0;
        assert!(!bad.is_valid());
        let mut bad2 = LifParams::default();
        bad2.threshold_v = bad2.reset_v - 1e-3; // threshold below reset
        assert!(!bad2.is_valid());
    }
}
