//! SI constants used by quantum-optics kernels.
//!
//! All universal constants below are `pub(super) use` re-exports of the
//! canonical [`kwavers_core::constants::fundamental`] SSOT entries — short
//! local aliases (`KB`, `HBAR`, `H_PLANCK`, `C`, `E_CHARGE`, `M_E`, `EPS0`)
//! are kept so the per-formula physics in this submodule reads naturally,
//! but their underlying values cannot drift from the workspace SSOT.

pub(super) use kwavers_core::constants::fundamental::{
    BOLTZMANN as KB, ELECTRON_MASS as M_E, ELEMENTARY_CHARGE as E_CHARGE, PLANCK as H_PLANCK,
    REDUCED_PLANCK as HBAR, SPEED_OF_LIGHT as C, VACUUM_PERMITTIVITY as EPS0,
};

/// Hydrogen 2s1/2 Lamb shift [eV].
pub(super) const LAMB_SHIFT_HYDROGEN_EV: f64 = 4.374e-6;
