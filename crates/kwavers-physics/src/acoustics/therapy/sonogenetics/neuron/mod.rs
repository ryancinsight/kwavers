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

/// Parameters for the two-state temperature-coupled neural response model
/// (Yoo et al. 2022, *Nat. Neurosci.* 25, 1557): thermal drive → calcium influx →
/// membrane depolarisation → logistic activation probability.
#[derive(Debug, Clone)]
pub struct ThermalNeuralParams {
    /// Membrane time constant τ_v [s].
    pub tau_v_s: f64,
    /// Calcium→voltage coupling k_ca [mV/µM].
    pub k_ca_mv_per_um: f64,
    /// Calcium time constant τ_ca [s].
    pub tau_ca_s: f64,
    /// Temperature→calcium coupling α_T [µM/°C].
    pub alpha_t_um_per_c: f64,
    /// Resting membrane potential [mV].
    pub v_rest_mv: f64,
    /// Resting cytosolic calcium [µM].
    pub ca_rest_um: f64,
    /// Reference temperature [°C].
    pub t_ref_c: f64,
    /// Activation-sigmoid midpoint [mV].
    pub v_thresh_mv: f64,
    /// Activation-sigmoid slope [mV].
    pub v_slope_mv: f64,
}

impl Default for ThermalNeuralParams {
    /// Thermal-coupling regime for tFUS at 0.5 MHz (Yoo 2022 Suppl. Table 1).
    fn default() -> Self {
        Self {
            tau_v_s: 0.010,
            k_ca_mv_per_um: 0.05,
            tau_ca_s: 0.100,
            alpha_t_um_per_c: 0.002,
            v_rest_mv: -70.0,
            ca_rest_um: 0.1,
            t_ref_c: 37.0,
            v_thresh_mv: -59.0,
            v_slope_mv: 2.8,
        }
    }
}

/// Integrate the two-state temperature-coupled neural response (Yoo et al. 2022) by
/// RK4:
/// ```text
/// d[Ca]/dt = −([Ca] − Ca_rest)/τ_ca + α_T·(T(t) − T_ref)
/// dV/dt    = −(V − V_rest)/τ_v + k_ca·[Ca]
/// P_act(t) = 1 / (1 + exp(−(V − V_thresh)/V_slope))
/// ```
/// The thermal drive `temperature_c` is sampled every `dt_thermal_s` and linearly
/// interpolated onto the neural grid; an empty/length-1 slice is treated as isothermal
/// at `params.t_ref_c`. Returns `(time_s, voltage_mv, activation_probability)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn yoo_thermal_neural_response(
    t_end_s: f64,
    dt_s: f64,
    v0_mv: f64,
    ca0_um: f64,
    temperature_c: &[f64],
    dt_thermal_s: f64,
    params: &ThermalNeuralParams,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let p = params;
    if !(t_end_s > 0.0 && dt_s > 0.0) {
        return (vec![0.0], vec![v0_mv], vec![0.0]);
    }
    let n_out = ((t_end_s / dt_s).ceil() as usize) + 1;
    let mut time = vec![0.0_f64; n_out];
    let mut voltage = vec![0.0_f64; n_out];
    let mut response = vec![0.0_f64; n_out];

    let default_t = [p.t_ref_c];
    let thermal: &[f64] = if temperature_c.is_empty() {
        &default_t
    } else {
        temperature_c
    };
    let dt_th = dt_thermal_s.max(1e-30);
    let temp_at = |t: f64| -> f64 {
        if thermal.len() == 1 {
            return thermal[0];
        }
        let idx_f = t / dt_th;
        let lo = (idx_f.floor() as usize).min(thermal.len() - 1);
        let hi = (lo + 1).min(thermal.len() - 1);
        if lo == hi {
            thermal[lo]
        } else {
            let frac = idx_f - lo as f64;
            thermal[lo] * (1.0 - frac) + thermal[hi] * frac
        }
    };
    let rhs = |t: f64, ca: f64, v: f64| -> (f64, f64) {
        let temp = temp_at(t);
        let dca = -(ca - p.ca_rest_um) / p.tau_ca_s + p.alpha_t_um_per_c * (temp - p.t_ref_c);
        let dv = -(v - p.v_rest_mv) / p.tau_v_s + p.k_ca_mv_per_um * ca;
        (dca, dv)
    };
    let sigmoid = |vm: f64| 1.0 / (1.0 + (-(vm - p.v_thresh_mv) / p.v_slope_mv).exp());

    let (mut ca, mut v) = (ca0_um, v0_mv);
    voltage[0] = v;
    response[0] = sigmoid(v);
    for i in 1..n_out {
        let t = (i - 1) as f64 * dt_s;
        let (dca1, dv1) = rhs(t, ca, v);
        let (dca2, dv2) = rhs(t + 0.5 * dt_s, ca + 0.5 * dt_s * dca1, v + 0.5 * dt_s * dv1);
        let (dca3, dv3) = rhs(t + 0.5 * dt_s, ca + 0.5 * dt_s * dca2, v + 0.5 * dt_s * dv2);
        let (dca4, dv4) = rhs(t + dt_s, ca + dt_s * dca3, v + dt_s * dv3);
        ca += dt_s / 6.0 * (dca1 + 2.0 * dca2 + 2.0 * dca3 + dca4);
        v += dt_s / 6.0 * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4);
        time[i] = t + dt_s;
        voltage[i] = v;
        response[i] = sigmoid(v);
    }
    (time, voltage, response)
}

#[cfg(test)]
mod yoo_tests {
    use super::*;

    #[test]
    fn isothermal_holds_near_rest() {
        let p = ThermalNeuralParams::default();
        let (_t, v, resp) =
            yoo_thermal_neural_response(0.2, 1e-4, p.v_rest_mv, p.ca_rest_um, &[], 0.01, &p);
        // Isothermal at T_ref → no thermal calcium drive → membrane holds essentially
        // at rest (only the negligible resting-calcium offset τ_v·k_ca·Ca_rest) and the
        // activation probability stays flat at its resting value.
        assert!(
            (v.last().unwrap() - p.v_rest_mv).abs() < 1e-2,
            "V drifted: {}",
            v.last().unwrap()
        );
        assert!(
            (resp.last().unwrap() - resp[0]).abs() < 1e-3,
            "activation not flat"
        );
    }

    #[test]
    fn heating_depolarises_and_raises_activation() {
        let p = ThermalNeuralParams::default();
        let temp = vec![42.0_f64; 200]; // +5 °C above reference
        let (_t, v, resp) =
            yoo_thermal_neural_response(1.0, 1e-4, p.v_rest_mv, p.ca_rest_um, &temp, 0.005, &p);
        assert!(
            *v.last().unwrap() > p.v_rest_mv,
            "heating should depolarise"
        );
        assert!(
            *resp.last().unwrap() > resp[0],
            "activation probability should rise"
        );
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
