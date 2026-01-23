//! Physics validation module
//!
//! Provides analytical solutions and validation tools for verifying
//! numerical implementations against known physics
//! TODO_AUDIT: P1 - Experimental Validation - Implement experimental validation against Brenner, Yasui, and Putterman sonoluminescence datasets, adding benchmark tests against real-world measurements
//! DEPENDS ON: validation/experimental/sonoluminescence.rs, validation/datasets/brenner_2002.rs
//! MISSING: Brenner et al. (2002) benchmark: T_max ≈ 10,000-15,000 K, R_min/R_0 ≈ 0.01-0.05
//! MISSING: Yasui (1997) multi-bubble model validation with experimental light intensity correlation
//! MISSING: Putterman single-bubble sonoluminescence spectra comparison (UV peak at 310 nm)
//! MISSING: Light emission pulse width validation: τ_pulse ≈ 100-200 ps
//! MISSING: Statistical analysis of experimental variability and parameter sensitivity

pub mod gaussian_beam;

pub use gaussian_beam::{measure_beam_radius, GaussianBeamParameters};
