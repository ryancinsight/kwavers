//! Frequency-swept (chirp) cavitation control.
//!
//! Two coupled mechanisms by which sweeping the drive frequency improves a
//! pulsed histotripsy exposure — and why the benefit is large for millisecond
//! pulses and negligible for microsecond pulses:
//!
//! 1. **Intra-pulse enhancement** ([`engagement`]). Tissue nuclei span a size
//!    distribution; each size has a Minnaert resonance. A single tone engages
//!    only the resonant size band, whereas a chirp sweeps its resonance across
//!    the band and engages a strictly larger population. The pulse must be many
//!    carrier cycles long for the sweep to traverse the band and for bubbles to
//!    ring up — so a ~single-cycle µs pulse realizes no advantage, while a ms
//!    pulse realizes the full enhancement.
//!
//! 2. **Inter-pulse dissipation** ([`clearance`]). A low-amplitude clearing
//!    sweep fragments persistent residual bubbles; the smaller daughters
//!    dissolve much faster (Epstein–Plesset `τ ∝ R²`), lowering the residual
//!    void fraction — and thus the shielding — at the next pulse. The leverage
//!    grows with the inter-pulse interval, which is long for low-PRF ms regimes.
//!
//! The chirped bubble dynamics ([`chirped_dynamics`]) reuse the audited
//! Keller–Miksis core; the chirp waveform itself is [`chirp`]; the nuclei
//! population is [`nuclei`].
//!
//! # References
//! * Khokhlova et al. (2015), *Int. J. Hyperthermia* 31, 145 (boiling histotripsy).
//! * Maxwell et al. (2013), *Ultrasound Med. Biol.* 39, 449 (cavitation threshold).
//! * Epstein & Plesset (1950), *J. Chem. Phys.* 18, 1505 (gas-diffusion dissolution).
//! * Flynn (1964); Apfel & Holland (1991) (inertial-collapse criterion).

mod chirp;
mod chirped_dynamics;
mod clearance;
mod engagement;
mod nuclei;
mod staged;

#[cfg(test)]
mod tests;

pub use chirp::{FrequencySweep, SweepProfile};
pub use chirped_dynamics::{chirped_keller_miksis_rk4, chirped_peak_expansion_ratio};
pub use clearance::{
    inter_pulse_residual_clearance, residual_dissolution_time_s, tissue_gas_diffusion,
    InterPulseClearance,
};
pub use engagement::{
    cavitation_optimal_frequency, monochromatic_engaged_fraction, swept_engaged_fraction,
    swept_vs_monochromatic_engagement, CavitationMedium, EngagementConfig, EngagementResult,
};
pub use nuclei::NucleiSizeDistribution;
pub use staged::{staged_sonication_sweep, StagedSonication};
