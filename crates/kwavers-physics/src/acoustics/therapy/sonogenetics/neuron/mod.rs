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
//!   `V[n+1] = V[n] + dt/C_m · (−G_leak · (V[n] − E_leak) + I_ion[n])`
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

#[cfg(test)]
mod tests;

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};

/// Leaky integrate-and-fire neuron parameters (SI units: F, S, V, s).
#[derive(Debug, Clone)]
pub struct LifParams {
    /// Membrane capacitance C_m (F).
    pub capacitance_f: f64,
    /// Leak conductance G_leak (S).
    pub leak_conductance_s: f64,
    /// Leak reversal potential E_leak (V).
    pub leak_reversal_v: f64,
    /// Spike threshold V_thresh (V).
    pub threshold_v: f64,
    /// Reset voltage V_reset (V).
    pub reset_v: f64,
    /// Absolute refractory period (s).
    pub refractory_s: f64,
}

impl Default for LifParams {
    /// Canonical mammalian neuron soma parameters (Koch 1999 Table 1.1).
    ///
    /// τ_m = C_m / G_leak = 100e-12 / 10e-9 = 10 ms.
    fn default() -> Self {
        Self {
            capacitance_f: 100.0e-12,
            leak_conductance_s: 10.0e-9,
            leak_reversal_v: -65.0e-3,
            threshold_v: -55.0e-3,
            reset_v: -65.0e-3,
            refractory_s: 2.0e-3,
        }
    }
}

impl LifParams {
    /// Membrane time constant τ_m = C_m / G_leak (s).
    #[inline]
    #[must_use]
    pub fn time_constant_s(&self) -> f64 {
        self.capacitance_f / self.leak_conductance_s
    }

    /// Returns `true` if all parameters are physically consistent.
    ///
    /// Invariants: C_m > 0, G_leak > 0, V_reset < V_thresh, refractory ≥ 0.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.capacitance_f > 0.0
            && self.leak_conductance_s > 0.0
            && self.reset_v < self.threshold_v
            && self.refractory_s >= 0.0
    }
}

/// Single leaky integrate-and-fire neuron driven by an external current source.
///
/// Implements forward-Euler integration of the LIF membrane equation.
/// Spike times are recorded in chronological order.
#[derive(Debug, Clone)]
pub struct LifNeuron {
    params: LifParams,
    /// Current membrane voltage (V).
    membrane_voltage_v: f64,
    /// Time elapsed since last spike (s). Starts at `+∞` to allow immediate spiking.
    time_since_spike_s: f64,
    /// Chronological spike times (s).
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
    /// Returns `Ok(true)` if a spike was emitted; `Ok(false)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `dt ≤ 0`.
    pub fn step(&mut self, i_ion: f64, dt: f64, t_now: f64) -> KwaversResult<bool> {
        if dt <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "dt".to_owned(),
                value: dt,
                reason: "time step must be strictly positive".to_owned(),
            }));
        }
        self.time_since_spike_s += dt;

        if self.time_since_spike_s < self.params.refractory_s {
            self.membrane_voltage_v = self.params.reset_v;
            return Ok(false);
        }

        let p = &self.params;
        let dv = (dt / p.capacitance_f)
            * (-p.leak_conductance_s).mul_add(self.membrane_voltage_v - p.leak_reversal_v, i_ion);
        self.membrane_voltage_v += dv;

        if self.membrane_voltage_v >= self.params.threshold_v {
            self.membrane_voltage_v = self.params.reset_v;
            self.time_since_spike_s = 0.0;
            self.spike_times_s.push(t_now);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Current membrane voltage (V).
    #[inline]
    #[must_use]
    pub fn membrane_voltage(&self) -> f64 {
        self.membrane_voltage_v
    }

    /// Recorded spike times in chronological order (s).
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

    /// Mean firing rate over `duration_s` seconds (Hz). Returns 0.0 for invalid input.
    #[must_use]
    pub fn mean_firing_rate(&self, duration_s: f64) -> f64 {
        if duration_s <= 0.0 || self.spike_times_s.is_empty() {
            0.0
        } else {
            self.spike_times_s.len() as f64 / duration_s
        }
    }

    /// Membrane time constant τ_m = C_m / G_leak (s).
    #[inline]
    #[must_use]
    pub fn time_constant_s(&self) -> f64 {
        self.params.time_constant_s()
    }
}
