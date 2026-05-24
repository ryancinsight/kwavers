//! Chemistry constants for reactive species

/// Molecular weight of hydroxyl radical (g/mol)
pub const HYDROXYL_RADICAL_WEIGHT: f64 = 17.01;

/// Molecular weight of hydrogen peroxide (g/mol)
pub const HYDROGEN_PEROXIDE_WEIGHT: f64 = 34.01;

/// Molecular weight of superoxide (g/mol)
pub const SUPEROXIDE_WEIGHT: f64 = 32.00;

/// Molecular weight of singlet oxygen (g/mol)
pub const SINGLET_OXYGEN_WEIGHT: f64 = 32.00;

/// Molecular weight of nitric oxide (g/mol)
pub const NITRIC_OXIDE_WEIGHT: f64 = 30.01;

/// Molecular weight of peroxynitrite (g/mol)
pub const PEROXYNITRITE_WEIGHT: f64 = 62.00;

/// Base photochemical reaction rate (1/s)
pub const BASE_PHOTOCHEMICAL_RATE: f64 = 1e-4;

/// Ionization energy of Argon (eV)
/// Value: 15.759 eV
/// Reference: NIST Atomic Spectra Database
pub const ARGON_IONIZATION_ENERGY: f64 = 15.7596;

// ── Water vapor thermal dissociation parameters ───────────────────────────────
//
// H₂O → OH + H  (reverse of hydroxyl combination)
//
// References:
// - Baulch DL et al. (2005). "Evaluated kinetic data for combustion modelling:
//   supplement II." J. Phys. Chem. Ref. Data 34(3):757–1397.
// - Yasui K (1997). "Alternative model of single-bubble sonoluminescence."
//   Phys. Rev. E 56(6):6750–6760. Table I.

/// Standard enthalpy of water dissociation (O−H bond) [J/mol].
///
/// H₂O → OH + H, ΔH = 498.4 kJ/mol at 298 K.
///
/// Reference: Baulch et al. (2005) Reaction R5; JANAF Tables (4th ed.) H₂O entry.
pub const H_WATER_DISSOCIATION_J_MOL: f64 = 498_400.0; // J/mol

/// Activation energy for H₂O → OH + H at high temperature [J/mol].
///
/// Ea = 495.4 kJ/mol (essentially the O−H bond dissociation energy).
///
/// Reference: Baulch et al. (2005); Yasui (1997) Table I.
pub const EA_WATER_DECOMPOSITION_J_MOL: f64 = 495_400.0; // J/mol

/// High-pressure Arrhenius pre-exponential factor for H₂O → OH + H [s⁻¹].
///
/// A = 1.912×10¹⁶ s⁻¹ (unimolecular, high-pressure limit).
///
/// Reference: Yasui K (1997). Phys. Rev. E 56(6):6750–6760. Table I.
pub const K_WATER_DECOMPOSITION_PRE_EXP: f64 = 1.912e16; // s⁻¹

// ── Aqueous-phase radical rate constants at 25°C ──────────────────────────────
//
// References:
// - Buxton GV et al. (1988). "Critical review of rate constants for reactions of
//   hydrated electrons, hydrogen atoms, and hydroxyl radicals in aqueous solution."
//   J. Phys. Chem. Ref. Data 17(2):513–886.
// - Sehested K et al. (1968–1991), various NDRL/NIST compilations.

/// OH self-recombination rate constant in aqueous phase at 25°C [M⁻¹·s⁻¹].
///
/// 2·OH → H₂O₂
///
/// Reference: Buxton et al. (1988) Table II-B, reaction 1.
pub const K_OH_SELF_RECOMBINATION: f64 = 5.5e9; // M⁻¹·s⁻¹

/// Superoxide dismutation pseudo-rate constant in aqueous phase at 25°C [M⁻¹·s⁻¹].
///
/// 2·O₂⁻ → H₂O₂ + ¹O₂ (pH-dependent; this value applies near neutral pH).
///
/// Reference: Bielski BHJ et al. (1985) J. Phys. Chem. Ref. Data 14(4):1041–1100.
pub const K_SUPEROXIDE_DISMUTATION: f64 = 1e8; // M⁻¹·s⁻¹

/// H₂O₂ + OH rate constant in aqueous phase at 25°C [M⁻¹·s⁻¹].
///
/// H₂O₂ + ·OH → ·OOH + H₂O
///
/// Reference: Buxton et al. (1988) Table II-B, reaction H₂O₂/OH.
pub const K_H2O2_OH: f64 = 2.7e7; // M⁻¹·s⁻¹

/// Activation energy for the H₂O₂ + OH reaction [J/mol].
///
/// Reference: estimated from Arrhenius fit to Buxton et al. (1988) temperature data.
pub const EA_H2O2_OH: f64 = 2800.0; // J/mol

/// O₃ + OH rate constant in aqueous phase at 25°C [M⁻¹·s⁻¹].
///
/// O₃ + ·OH → ·OOH + O₂
///
/// Reference: Sehested K et al. (1984) Int. J. Radiat. Phys. Chem. 16:3;
/// Staehelin J & Hoigné J (1985) Environ. Sci. Technol. 19:1206.
pub const K_OZONE_OH: f64 = 1.1e8; // M⁻¹·s⁻¹

/// Singlet oxygen physical quenching rate constant in water at 25°C [s⁻¹].
///
/// ¹O₂ → O₂ (ground state), first-order rate in pure water.
///
/// Reference: Wilkinson F et al. (1995) J. Phys. Chem. Ref. Data 24(2):663–1021.
pub const K_SINGLET_OXYGEN_DECAY: f64 = 2.9e5; // s⁻¹

/// Atomic hydrogen self-recombination rate constant in aqueous phase at 25°C [M⁻¹·s⁻¹].
///
/// 2·H → H₂
///
/// Reference: Buxton et al. (1988) Table II-A, reaction H·/H·.
pub const K_H_ATOM_RECOMBINATION: f64 = 1e10; // M⁻¹·s⁻¹

/// G-value (radiochemical yield) for ·OH production in water at neutral pH [molecules/(100 eV)].
///
/// Reference: Spinks JWT & Woods RJ (1990) "An Introduction to Radiation Chemistry",
/// Wiley, 3rd ed., Appendix C; ICRU Report 16 (1970).
pub const G_OH_NEUTRAL_PH: f64 = 2.7; // molecules per 100 eV
