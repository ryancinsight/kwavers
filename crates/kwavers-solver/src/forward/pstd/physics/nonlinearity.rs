//! Nonlinear shock-formation diagnostics for the PSTD solver.
//!
//! # Responsibility
//!
//! The PSTD nonlinear pressure update is performed inline in
//! `update_density_cartesian` (the canonical spectral path uses the Treeby &
//! Cox (2010) fractional-Laplacian operator).  This module provides
//! diagnostic utilities that operate on pre-computed fields derived from the
//! running simulation.
//!
//! # Shock Formation
//!
//! The Goldberg number σ = β·k·p₀·z / (ρ₀c₀³) characterises cumulative
//! nonlinearity.  Shock formation begins when σ ≥ 1 (Fubini solution limit).
//!
//! # References
//!
//! - Hamilton & Blackstock (1998). Nonlinear Acoustics. Academic Press. §4.3.
//! - Aanonsen SI et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768.

use leto::Array3;

/// Return `true` if any cell has crossed the shock-formation threshold σ ≥ 1.
///
/// ## Theorem (Fubini limit)
///
/// The Goldberg number σ = β·k·p₀·z / (ρ₀c₀³) equals unity at the shock
/// formation distance z_shock.  σ > 1 indicates the waveform has entered the
/// sawtooth (Blackstock) regime.
///
/// ## Reference
///
/// Hamilton & Blackstock (1998). Nonlinear Acoustics §4.3, eq. (4.3.5).
#[must_use]
pub fn is_shock_forming(sigma: &Array3<f64>) -> bool {
    sigma.iter().any(|&s| s > 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shock_detection() {
        let mut sigma = Array3::zeros((10, 10, 10));

        // σ < 1 everywhere: no shock.
        assert!(!is_shock_forming(&sigma));

        // Single cell crosses threshold.
        sigma[[5, 5, 5]] = 1.5;
        assert!(is_shock_forming(&sigma));
    }
}
