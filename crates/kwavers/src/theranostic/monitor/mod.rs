//! Hybrid monitored-slice reconstruction (ADR 024, Stage 4).
//!
//! RTM reflectivity alone cannot image a small thermal sound-speed lesion through
//! the skull (sub-wavelength contrast buried under skull reverberation). The
//! monitor therefore combines independent reconstructions of the **same** recorded
//! multistatic data, which a 1024-element hemisphere supplies in large surplus
//! (≈N² complex samples per frequency for ≈N² slice unknowns — heavily
//! overdetermined):
//!
//! - [`fd`] — **frequency-domain CBS differential inversion** around the fixed
//!   pre-therapy background. The quantitative thermal-lesion channel. The
//!   convergent Born series (Osnabrugge et al. 2016) handles the strong skull
//!   multiple scattering that breaks a single-scatter Born update.
//! - (planned) `rtm` — reflectivity structural channel for the strong cavitation
//!   scatterer, migrated through the FWI background.
//! - (planned) `pam` — passive cavitation source map from the therapy-burst
//!   receive data.
//! - (planned) `fusion` — RTM/PAM structure as a TV/edge-preserving prior on the
//!   FD quantitative inversion.

pub mod fd;
pub mod fusion;
pub mod pam;

pub use fd::{differential_lesion_map, reconstruct, ring_around_slice, FdMonitorConfig};
pub use fusion::{fuse_lesion_map, lesion_extent, FusedLesion, FusionWeights};
pub use pam::{passive_acoustic_map, synthesize_emission, PamMonitorConfig};
