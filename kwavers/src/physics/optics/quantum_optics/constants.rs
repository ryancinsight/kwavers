//! SI constants used by quantum-optics kernels.

/// Boltzmann constant [J/K].
pub(super) const KB: f64 = 1.380_649e-23;
/// Reduced Planck constant [J s].
pub(super) const HBAR: f64 = 1.054_571_817e-34;
/// Planck constant [J s].
pub(super) const H_PLANCK: f64 = 6.626_070_15e-34;
/// Speed of light [m/s].
pub(super) const C: f64 = 2.997_924_58e8;
/// Elementary charge [C].
pub(super) const E_CHARGE: f64 = 1.602_176_634e-19;
/// Electron mass [kg].
pub(super) const M_E: f64 = 9.109_383_701_5e-31;
/// Vacuum permittivity [F/m].
pub(super) const EPS0: f64 = 8.854_187_812_8e-12;
/// Hydrogen 2s1/2 Lamb shift [eV].
pub(super) const LAMB_SHIFT_HYDROGEN_EV: f64 = 4.374e-6;
