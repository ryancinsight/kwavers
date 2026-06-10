//! Conductance-based membrane abstraction for the NICE / SONIC integrators.
//!
//! A [`Membrane`] supplies the ionic current and gating kinetics of a
//! point-neuron membrane so that the displacement-current coupling
//! ([`super::nice`]) and its cycle-averaged reduction ([`super::sonic`]) are
//! written once and monomorphised per neuron model. The gating state is a fixed
//! four-element array `[m, h, n, p]` (Na activation, Na inactivation, K
//! delayed-rectifier activation, slow non-inactivating K / M-current); models
//! that lack the M-current simply leave `p` inert (`g_M = 0`).
//!
//! Two implementors are provided:
//! - [`super::hodgkin_huxley::HhParams`] — the classic squid axon (Na, K, leak;
//!   `p` unused), validated against the 1952 reference.
//! - [`super::cortical::CorticalNeuron`] — the Pospischil et al. (2008) cortical
//!   neuron (Na, Kd, M-current, leak; regular- and fast-spiking presets) that the
//!   NICE model of Plaksin et al. (2014) actually uses.
//!
//! # References
//!
//! - Hodgkin, A.L. & Huxley, A.F. (1952). *J. Physiol.* 117(4), 500-544.
//! - Pospischil, M. et al. (2008). Minimal Hodgkin–Huxley type models for
//!   different classes of cortical and thalamic neurons. *Biol. Cybern.* 99,
//!   427-441.

/// Gating state `[m, h, n, p]` shared by all [`Membrane`] models.
pub type Gates = [f64; 4];

/// A conductance-based point-neuron membrane.
///
/// All quantities use the module's electrophysiology units: `V` [mV], `t` [ms],
/// `C_m` [µF/cm²], `g` [mS/cm²], current density [µA/cm²].
pub trait Membrane {
    /// Resting gating state at membrane potential `v_rest_mv` (each gate at its
    /// voltage-dependent steady state).
    fn resting_gates(&self, v_rest_mv: f64) -> Gates;

    /// Total ionic current density [µA/cm²] at gating state `g` and potential
    /// `v_mv` (sum of all voltage-gated and leak currents).
    fn ionic_current(&self, g: &Gates, v_mv: f64) -> f64;

    /// Gate time-derivatives `d[m,h,n,p]/dt` [1/ms] at state `g`, potential `v_mv`.
    fn gate_rates(&self, g: &Gates, v_mv: f64) -> Gates;

    /// Baseline (resting) specific membrane capacitance C_m0 [µF/cm²].
    fn cm0_uf_cm2(&self) -> f64;

    /// Physical-consistency predicate for the membrane parameters.
    fn is_membrane_valid(&self) -> bool;
}

/// `base + w·delta` with the four gates clamped to `[0, 1]` (RK4 stage helper).
#[inline]
#[must_use]
pub fn axpy_gates(base: &Gates, delta: &Gates, w: f64) -> Gates {
    [
        (base[0] + w * delta[0]).clamp(0.0, 1.0),
        (base[1] + w * delta[1]).clamp(0.0, 1.0),
        (base[2] + w * delta[2]).clamp(0.0, 1.0),
        (base[3] + w * delta[3]).clamp(0.0, 1.0),
    ]
}

/// Weighted RK4 combination `base + (dt/6)·(k1 + 2k2 + 2k3 + k4)`, gates clamped.
#[inline]
#[must_use]
pub fn rk4_combine_gates(base: &Gates, k1: &Gates, k2: &Gates, k3: &Gates, k4: &Gates, dt: f64) -> Gates {
    let mut out = *base;
    for i in 0..4 {
        out[i] = (base[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])).clamp(0.0, 1.0);
    }
    out
}
