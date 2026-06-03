//! Passive-cavitation harmonic-dose monitoring (clinical BBB / histotripsy).
//!
//! This is the first-principles analogue of the cavitation-dose controllers
//! deployed clinically (InsighTec Exablate Neuro for BBB opening; passive
//! cavitation detection for histotripsy treatment monitoring). The pipeline is:
//!
//! 1. [`emission`] – drive a microbubble (Keller–Miksis / Rayleigh–Plesset) and
//!    compute the far-field acoustic emission `p_sc(t)` the receivers detect.
//! 2. power spectrum – via [`super::bubble_power_spectrum`].
//! 3. [`spectral_bands`] – split the spectrum into harmonic / subharmonic /
//!    ultraharmonic / broadband energy above a noise floor.
//! 4. [`dose`] – integrate stable (sub+ultra) and inertial (broadband) emission
//!    over the sonication into cumulative cavitation doses, and step the
//!    closed-loop pressure controller.
//! 5. [`receiver`] – integrate the multi-element receiver array (and PAM source
//!    maps) over the sonication volume `V_s`.
//!
//! All physics lives here; the book chapters (ch21e histotripsy, ch24 BBB) call
//! these via PyO3 and only plot.

mod dose;
mod emission;
mod ensemble;
mod receiver;
mod simulate;
mod spectral_bands;
mod spectrum;

#[cfg(test)]
mod tests;

pub use dose::{cavitation_controller_pressure, cumulative_cavitation_dose};
pub use emission::bubble_acoustic_emission_pressure;
pub use ensemble::ensemble_emission_superposition;
pub use receiver::{emission_energy_in_volume, integrate_receiver_array_psd};
pub use simulate::{
    simulate_bubble_emission, simulate_coated_bubble_emission, BubbleDriveConfig,
    BubbleEmissionTrace, ShellDriveConfig,
};
pub use spectral_bands::{decompose_emission_spectrum, CavitationBandEnergies};
pub use spectrum::hann_windowed_power_spectrum;
