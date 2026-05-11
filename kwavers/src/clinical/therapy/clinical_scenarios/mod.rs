//! Clinical exposure scenarios for histotripsy.
//!
//! This module defines literature-derived parameter bundles for the principal
//! histotripsy exposure regimes used in pre-clinical and clinical practice
//! (microsecond intrinsic-threshold, shock-scattering, boiling, and
//! sub-threshold millisecond cavitation), together with the optimal pulse
//! patterns reported in the recent histotripsy literature (2014–2024).
//!
//! References for parameter values:
//! - Maxwell A.D. et al. (2013) UMB 39(3) — intrinsic-threshold CDF.
//! - Vlaisavljevich E. et al. (2015) UMB 41(6) — frequency / stiffness scaling
//!   of the intrinsic threshold for water-rich soft tissue.
//! - Khokhlova V.A. et al. (2015) Int. J. Hyperthermia 31(2) — boiling
//!   histotripsy pulse design.
//! - Khokhlova T.D. et al. (2019) Sci. Rep. 9, 20176 — porcine BH parameters.
//! - Maxwell A.D. et al. (2011) UMB 37(3) — shock-scattering histotripsy.
//! - Macoskey J.J. et al. (2018) UMB 44(12) — dual-PRF cloud regeneration.
//! - Mancia L. et al. (2020) Phys. Med. Biol. 65 — dithered-PRF nucleation.
//! - Maeda K. et al. (2018) JASA 144(3) — bubble-cloud collective dynamics.
//! - Vlaisavljevich E. et al. (2018) UMB 44(7) — sub-threshold ms cavitation.
//! - Smolock A. et al. (2018) UMB 44(9) — clinical liver histotripsy dosing.
//!
//! All thresholds are quoted as positive magnitudes; peak negative pressure
//! values use the sign convention `peak_negative_pressure_pa < 0`.

mod scenario;

pub use scenario::{
    BenefitDetriment, HistotripsyRegime, HistotripsyScenario, PulsePattern,
    intrinsic_threshold_pa,
};

#[cfg(test)]
mod tests;
