//! Noble-gas ionization data for Saha kinetics.

/// Supported noble gas species for multi-stage ionization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NobleGas {
    /// Helium: first and second ionization stages.
    Helium,
    /// Argon, the common gas in SBSL experiments.
    Argon,
    /// Xenon.
    Xenon,
    /// Krypton.
    Krypton,
}

impl NobleGas {
    /// First and second ionization energies in eV.
    ///
    /// Source: NIST Atomic Spectra Database values used by the prior module.
    #[must_use]
    pub fn ionization_energies_ev(self) -> (f64, f64) {
        match self {
            Self::Helium => (24.587, 54.418),
            Self::Argon => (15.760, 27.630),
            Self::Xenon => (12.130, 20.975),
            Self::Krypton => (13.999, 24.360),
        }
    }
}
