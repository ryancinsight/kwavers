//! Threshold and dose–response analysis for ultrasonic neuromodulation.
//!
//! Reproduces the strength–frequency and strength–duration relationships of the
//! NICE model (Plaksin et al. 2014, Fig. 3): the acoustic pressure/intensity
//! threshold to evoke an action potential **rises with carrier frequency** and
//! **falls with pulse duration**. The threshold is found by bisection on the
//! pressure amplitude, building a [`CapacitanceSource`] at each trial pressure
//! and running the carrier-resolved NICE coupling ([`simulate_nice`]).
//!
//! The search is source-agnostic: the caller supplies a closure mapping a
//! pressure amplitude to a source, so either the quasi-static
//! ([`super::bls::BilayerSonophoreQuasistatic`]) or the full transient
//! ([`super::bls::BilayerSonophoreDynamic`]) bilayer-sonophore source can be used.
//!
//! # References
//!
//! - Plaksin, M., Shoham, S. & Kimmel, E. (2014). *Phys. Rev. X* 4, 011004,
//!   Fig. 3 (threshold intensity vs frequency and duration).

use super::intramembrane_cavitation::CapacitanceSource;
use super::membrane::Membrane;
use super::nice::{simulate_nice, NiceConfig};

/// Fixed (pressure-independent) parameters of a threshold search.
#[derive(Debug, Clone)]
pub struct ThresholdQuery<M: Membrane + Clone> {
    /// Membrane model (cloned per trial pressure).
    pub membrane: M,
    /// Resting membrane potential / initial condition `mV`.
    pub v_rest_mv: f64,
    /// Constant bias current density [µA/cm²].
    pub i_bias_ua_cm2: f64,
    /// Carrier-resolved integration step `ms`.
    pub dt_ms: f64,
    /// Sonication onset `ms`.
    pub onset_ms: f64,
    /// Sonication offset `ms`.
    pub offset_ms: f64,
    /// Total simulated duration `ms` (> offset to capture the post-stimulus AP).
    pub t_end_ms: f64,
}

impl<M: Membrane + Clone> ThresholdQuery<M> {
    /// Whether an acoustic source produced by `make_source(pressure_pa)` evokes at
    /// least one action potential.
    #[must_use]
    pub fn fires<C, F>(&self, pressure_pa: f64, make_source: &F) -> bool
    where
        C: CapacitanceSource,
        F: Fn(f64) -> C,
    {
        let cfg = NiceConfig {
            membrane: self.membrane.clone(),
            v_rest_mv: self.v_rest_mv,
            source: make_source(pressure_pa),
            i_bias_ua_cm2: self.i_bias_ua_cm2,
            dt_ms: self.dt_ms,
            onset_ms: self.onset_ms,
            offset_ms: self.offset_ms,
            t_end_ms: self.t_end_ms,
        };
        cfg.is_valid() && simulate_nice(&cfg).spike_count() >= 1
    }

    /// Minimum acoustic pressure amplitude `Pa` that evokes an AP, found by
    /// bisection on `[p_lo, p_hi]` for `n_iter` halvings (monotonic
    /// dose–response is assumed). Returns `None` if `p_hi` does not fire (no
    /// threshold within range) or if `p_lo` already fires (threshold below range).
    #[must_use]
    pub fn threshold_pressure_pa<C, F>(
        &self,
        p_lo: f64,
        p_hi: f64,
        n_iter: usize,
        make_source: &F,
    ) -> Option<f64>
    where
        C: CapacitanceSource,
        F: Fn(f64) -> C,
    {
        if !self.fires(p_hi, make_source) || self.fires(p_lo, make_source) {
            return None;
        }
        let (mut lo, mut hi) = (p_lo, p_hi); // fires(lo)=false, fires(hi)=true
        for _ in 0..n_iter {
            let mid = 0.5 * (lo + hi);
            if self.fires(mid, make_source) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }
}
