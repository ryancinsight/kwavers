//! Bremsstrahlung radiation and Saha-Boltzmann plasma kinetics.
//!
//! # Physics Overview
//!
//! Single-bubble sonoluminescence generates light through thermal
//! bremsstrahlung emitted by electrons decelerating in the Coulomb field of
//! ions inside the hot, compressed bubble plasma.
//!
//! ## Free-Free Emission Theorem
//!
//! The emission coefficient per unit volume, frequency, and steradian is
//!
//! ```text
//! j_nu^ff = n_e n_i Z^2 g_ff C_ff T^(-1/2) exp(-h nu / kT).
//! ```
//!
//! Proof: starting from the Maxwellian velocity average of electron-ion
//! free-free radiation, integrating over electron speeds yields the
//! `T^(-1/2) exp(-h nu/kT)` factor. The Coulomb cross-section contributes the
//! `n_e n_i Z^2` density and charge scaling, and the Gaunt factor `g_ff`
//! applies the quantum correction to the classical result.
//!
//! ## Saha Equilibrium Theorem
//!
//! In local thermodynamic equilibrium,
//!
//! ```text
//! n_e n_{j+1} / n_j
//!   = 2 (U_{j+1}/U_j) (2 pi m_e kT / h^2)^(3/2) exp(-chi_j/kT).
//! ```
//!
//! Proof: equating the chemical potentials of `j`, `j+1`, and the free
//! electron, then substituting the translational partition function of the
//! electron, gives the Saha ratio. The factor two is the electron spin
//! degeneracy.
//!
//! References: Saha (1920), Rybicki & Lightman (1979), Elwert (1939),
//! van Hoof et al. (2014), Raizer (1991), Brenner et al. (2002).

mod constants;
mod field;
mod gaunt;
mod model;
mod plasma;
mod species;

#[cfg(test)]
mod tests;

pub use field::calculate_bremsstrahlung_emission;
pub use gaunt::gaunt_factor_thermal;
pub use model::BremsstrahlungModel;
pub use plasma::PlasmaState;
pub use species::NobleGas;
