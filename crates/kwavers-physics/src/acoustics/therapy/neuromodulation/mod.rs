//! Ultrasonic neuromodulation via the electrical (capacitive) pathway.
//!
//! This module complements [`crate::acoustics::therapy::sonogenetics`] (which
//! covers the *mechanosensitive-channel* pathway — Blackmore et al. 2019
//! mechanism (ii)) by providing the *membrane-capacitance / intramembrane-
//! cavitation* pathway — Blackmore mechanism (i) — built on a genuine
//! Hodgkin–Huxley conductance neuron.
//!
//! # Mechanism coverage
//!
//! | Mechanism (Blackmore 2019)              | kwavers module                          |
//! |-----------------------------------------|-----------------------------------------|
//! | (i)  Membrane capacitance / cavitation  | this module ([`nice`])                  |
//! | (ii) Mechanosensitive channels          | [`crate::acoustics::therapy::sonogenetics`] |
//! | Thermal pathway                         | `sonogenetics::yoo_thermal_neural_response` |
//! | Acoustic radiation force / streaming    | `sonogenetics::VolumetricArfField`      |
//!
//! # Pipeline
//!
//! ```text
//!  acoustic pressure p(x,t) / leaflet deflection Z(t)
//!       │
//!       ▼  capacitance source (CapacitanceSource trait)
//!  C_m(t), dC_m/dt
//!       │  NICE coupling (Plaksin Eq. 1): displacement current −V·dC_m/dt
//!       ▼
//!  Hodgkin–Huxley membrane  →  excitability shift / action potentials
//! ```
//!
//! # Two capacitance sources
//!
//! - [`BilayerSonophore`] — the **grounded** intramembrane-cavitation source: the
//!   exact curved-dome capacitance geometry `C_m(Z)` of Plaksin et al. (2014,
//!   Eq. 8) driven by a leaflet deflection. Its asymmetric waveform reproduces
//!   the NICE mechanism: membrane *hyperpolarisation during* sonication, net
//!   *charge accumulation*, and a *post-stimulus action potential* with the
//!   characteristic pulse-duration dependence (the requirement for long pulses).
//! - [`CapacitanceModulation`] — a simpler symmetric sinusoid `C_m0(1+ε·sinωt)`
//!   whose net cycle-averaged effect is a depth-dependent (hyperpolarising)
//!   excitability shift; useful as an analytic baseline (see [`nice`] for the
//!   sign analysis).
//!
//! [`simulate_nice`] is generic over [`CapacitanceSource`]; the Hodgkin–Huxley
//! neuron is validated against the canonical 1952 squid-axon reference and can be
//! driven directly by any external current via [`simulate_hh`].
//!
//! # Validation, accuracy and limitations (honest critique)
//!
//! **Exactly reproduced & value-semantically tested:** the Hodgkin–Huxley neuron
//! (1952 squid-axon reference — rest gating, AP overshoot, monotone f–I); the
//! Pospischil RS/FS cortical neuron parameters/kinetics; the displacement-current
//! coupling (Plaksin Eq. 1); the curved-dome capacitance `C_m(Z)` (Eq. 8); the
//! full bilayer-sonophore pressure set (Eq. 3–8) with the rest-gap solver that
//! recovers Δ ≈ 1.26 nm from the charge balance; and the SONIC cycle-averaging
//! (differential-tested against the carrier-resolved path within 1 ms).
//!
//! **Approximations (documented):**
//! - *Deflection accuracy.* The transient [`BilayerSonophoreDynamic`] integrates
//!   the full leaflet Rayleigh–Plesset ODE (Eq. 2) and reproduces Plaksin Fig. 1
//!   to ≈ 10–11 nm peak at 500 kPa / 0.5 MHz vs the published ≈ 12 nm (~10 % low),
//!   a consequence of the carrier step (`dt_max = T/200`), the recorded-cycle
//!   interpolation, and the `P_M(Z)` lookup table.
//! - *Multi-scale charge separation.* A source's one-cycle `C_m(t)` is computed at
//!   the resting charge `Q_m0` and reused during the slow membrane evolution; the
//!   electrical (Maxwell-stress) pressure therefore does not track the
//!   accumulating `Q_m` within the carrier cycle. This is the standard NICE/SONIC
//!   timescale separation, valid while `Q_m` drifts slowly versus the carrier.
//! - *`P_M(Z)` lookup.* Exact quadrature is tabulated over the smooth deflection
//!   range and interpolated (O(1)); the steep steric-wall band and out-of-range
//!   `Z` fall back to the exact quadrature, so accuracy is grid-controlled.
//! - *Steric-wall edge.* At over-driven amplitudes the leaflet reaches `Z_min`,
//!   where the explicit integrator treats contact inelastically (`Z = Z_min`,
//!   inward velocity dropped). The validated regime (cortical rest −71.9 mV,
//!   ≤ 500 kPa; squid rest ≤ ~300 kPa) does not reach the wall.
//!
//! **Scope (completeness).** Neuron models: squid HH and Pospischil RS/FS cortical
//! (thalamic classes and explicit temperature-dependent gating are out of scope;
//! the thermal pathway is the separate `yoo_thermal_neural_response`). Capacitance
//! sources span analytic, kinematic, quasi-static and full-transient bilayer
//! sonophores plus the cycle-averaged SONIC reduction.
//!
//! # References
//!
//! - Hodgkin, A.L. & Huxley, A.F. (1952). *J. Physiol.* 117(4), 500-544.
//! - Krasovitski, B. et al. (2011). *PNAS* 108(8), 3258-3263.
//! - Plaksin, M., Shoham, S. & Kimmel, E. (2014). *Phys. Rev. X* 4, 011004.
//! - Lemaire, T. et al. (2019). *J. Neural Eng.* 16, 046007 (SONIC).
//! - Blackmore, J. et al. (2019). *Ultrasound Med. Biol.* 45(7), 1509-1536.
//! - Manuel, T.J. et al. (2020). Ultrasound neuromodulation depends on pulse
//!   repetition frequency and can modulate inhibitory effects of TTX.
//!   *Sci. Rep.* 10, 15347.

pub mod analysis;
pub mod bls;
pub mod cortical;
pub mod hodgkin_huxley;
pub mod intramembrane_cavitation;
pub mod membrane;
pub mod nice;
pub mod protocol;
pub mod sonic;

#[cfg(test)]
mod tests;

pub use analysis::ThresholdQuery;
pub use bls::{
    bls_capacitance, quasistatic_deflection, rest_gap, BilayerSonophore, BilayerSonophoreDynamic,
    BilayerSonophoreQuasistatic, LEAFLET_GAP_M, SONOPHORE_RADIUS_M,
};
pub use cortical::CorticalNeuron;
pub use hodgkin_huxley::{simulate_hh, HhParams, HhState, HhTrace, SPIKE_THRESHOLD_MV};
pub use intramembrane_cavitation::{
    modulation_depth_from_pressure, CapacitanceModulation, CapacitanceSource, PhaseCycle,
    BILAYER_AREA_MODULUS_N_M,
};
pub use membrane::{Gates, Membrane};
pub use nice::{simulate_nice, NiceConfig};
pub use protocol::{
    itrusst_assess, tissue_dosimetry, ItrusstAssessment, PulseTrainDosimetry, PulseTrainProtocol,
};
pub use sonic::{simulate_sonic, SonicConfig};
