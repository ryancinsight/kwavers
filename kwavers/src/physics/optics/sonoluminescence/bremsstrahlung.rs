//! Bremsstrahlung Radiation and Saha–Boltzmann Plasma Kinetics
//!
//! # Physics Overview
//!
//! Single-bubble sonoluminescence (SBSL) generates light through thermal bremsstrahlung
//! ("braking radiation") emitted by electrons decelerating in the Coulomb field of ions
//! inside the hot, compressed bubble plasma.
//!
//! ## Thermal Bremsstrahlung Emission Coefficient
//!
//! The free-free emission coefficient per unit volume, per unit frequency, per steradian
//! (Rybicki & Lightman 1979, eq. 5.14a; SI units):
//!
//! ```text
//! j_ν^ff = n_e n_i Z² g_ff × C_ff × T^{−1/2} × exp(−hν / kT)   [W m⁻³ Hz⁻¹ sr⁻¹]
//! ```
//!
//! where `C_ff = 6.8×10⁻⁵¹ W m³ K^{1/2}` is the fundamental constant:
//!
//! ```text
//! C_ff = (8 e⁶) / (3 m_e² c³ (4πε₀)³) × (2π m_e / (3k))^{1/2} / (4π)
//! ```
//!
//! Note: this is already `/ 4π` (per steradian, isotropic emission assumed).
//!
//! ## Gaunt Factor
//!
//! The Gaunt factor g_ff(ν, T) is the quantum mechanical correction to the
//! classical bremsstrahlung formula.  For the thermally averaged Gaunt factor
//! (Elwert 1939; van Hoof et al. 2014):
//!
//! ```text
//! g_ff(ν, T) ≈ √3/π × ln(2kT / (hν))     hν < kT   (optical/UV in hot plasma)
//!            ≈ 1.0                          hν ≫ kT   (hard X-ray limit)
//! ```
//!
//! clamped to g_ff ∈ [1.0, 10.0] for numerical stability.
//!
//! ## Saha–Boltzmann Ionization Equilibrium
//!
//! ### Theorem (Saha 1920)
//!
//! In thermodynamic equilibrium (LTE), the ionization state of a gas satisfies:
//!
//! ```text
//! n_e n_{j+1} / n_j = K_j(T)
//!
//! K_j(T) = 2 × (U_{j+1}/U_j) × (2π m_e k T / h²)^{3/2} × exp(−χ_j / kT)
//! ```
//!
//! where:
//! - n_j = number density of the jth ionization stage [m⁻³]
//! - χ_j = ionization energy from stage j [J]
//! - U_j = partition function of stage j
//! - The factor of 2 accounts for the two spin states of the freed electron
//!
//! ### Single-stage quadratic solution
//!
//! For a pure gas with a single ionization stage, letting x = n_e / n_total:
//!
//! ```text
//! x² n_total / (1 − x) = K(T)
//! ```
//!
//! Rearranging to `x² + s·x − s = 0` (where s = K/n_total) gives:
//!
//! ```text
//! x = (−s + √(s² + 4s)) / 2
//! ```
//!
//! ### Multi-stage ionization (noble gases)
//!
//! For gases like Ar, Xe, He that undergo multiple ionization stages, the Saha
//! system becomes coupled.  The electron density satisfies:
//!
//! ```text
//! n_e = Σ_j j × n_j
//! ```
//!
//! This is solved iteratively using the Newton–Raphson method.
//!
//! ## References
//!
//! - Saha MN (1920). Philos. Mag. 40(238), 472–488. DOI:10.1080/14786441008636148
//! - Rybicki GB, Lightman AP (1979). Radiative Processes in Astrophysics.
//!   Wiley-Interscience. §5.2, eq. (5.14a).
//! - Elwert G (1939). Ann. Phys. 426(2), 178–208. DOI:10.1002/andp.19394260208
//! - van Hoof PAM et al. (2014). MNRAS 444, 420–428. DOI:10.1093/mnras/stu1438
//! - Raizer YP (1991). Gas Discharge Physics. Springer. §1.
//! - Brenner MP, Hilgenfeldt S, Lohse D (2002). Rev. Mod. Phys. 74(2), 425–484.

use ndarray::{Array1, Array3, Zip};
use std::f64::consts::PI;

use crate::core::constants::fundamental::{
    BOLTZMANN as BOLTZMANN_CONSTANT, ELECTRON_MASS, ELEMENTARY_CHARGE as ELECTRON_CHARGE,
    PLANCK as PLANCK_CONSTANT, SPEED_OF_LIGHT, VACUUM_PERMITTIVITY,
};

// ────────────────────────────────────────────────────────────────────────────
// Physical constants (derived)
// ────────────────────────────────────────────────────────────────────────────

/// Pre-factor constant C_ff for the free-free emission coefficient [W m³ K^{1/2} sr⁻¹].
///
/// Numerically equal to 6.8×10⁻⁵¹ (Rybicki & Lightman 1979, eq. 5.14a, converted to SI).
///
/// ## Formula
///
/// ```text
/// C_ff = 32π e⁶ / (3 m_e c³ (4πε₀)³) × (2π / (3k m_e))^{1/2}
/// ```
///
/// ## Derivation note
///
/// The m_e appears once in the denominator of the prefactor (not squared) and once
/// in the denominator of the thermal factor.  The product is dimensionally consistent:
/// the two appearances of m_e give a combined m_e^{−3/2}, which is correct for
/// j_ν^ff ∝ (m_e k T)^{−1/2} × m_e^{−1} (the latter from the Coulomb cross-section).
/// This is the standard form of the Gaunt-factor bremsstrahlung formula (see Rybicki
/// & Lightman 1979, eq. 5.14a; also Longair 2011, §3.6).
fn c_ff_per_sr() -> f64 {
    let e6 = ELECTRON_CHARGE.powi(6);
    let four_pi_eps0 = 4.0 * PI * VACUUM_PERMITTIVITY;
    // Prefactor: 32π e⁶ / (3 m_e c³ (4πε₀)³)
    let prefactor =
        32.0 * PI * e6 / (3.0 * ELECTRON_MASS * SPEED_OF_LIGHT.powi(3) * four_pi_eps0.powi(3));
    // Thermal velocity factor: (2π / (3k m_e))^{1/2}  [units: (J·kg)^{−1/2}]
    let thermal_coeff = (2.0 * PI / (3.0 * BOLTZMANN_CONSTANT * ELECTRON_MASS)).sqrt();
    prefactor * thermal_coeff
}

/// Saha equilibrium constant K₀(T) (without the T^{3/2} and exp(−χ/kT) factors).
///
/// K_j(T) = K₀ × 2 × (U_{j+1}/U_j) × T^{3/2} × exp(−χ_j/kT)
///
/// K₀ = (2π m_e k / h²)^{3/2}
fn saha_k0() -> f64 {
    (2.0 * PI * ELECTRON_MASS * BOLTZMANN_CONSTANT / PLANCK_CONSTANT.powi(2)).powf(1.5)
}

// ────────────────────────────────────────────────────────────────────────────
// Gaunt factor
// ────────────────────────────────────────────────────────────────────────────

/// Thermally averaged free-free Gaunt factor g_ff(ν, T).
///
/// ## Algorithm  (Elwert 1939 approximation)
///
/// For optical/UV frequencies (hν < kT):
/// ```text
/// g_ff(ν, T) = max(1.0,  √3/π × ln(2kT / (h·ν)))
/// ```
///
/// For hard X-ray (hν ≥ kT):
/// ```text
/// g_ff(ν, T) = 1.0
/// ```
///
/// Clamped to [1.0, 10.0] for numerical safety.
///
/// Accuracy: ~20 % for hν/kT ∈ [0.01, 100] (van Hoof et al. 2014 Table 1).
///
/// ## References
///
/// - Elwert G (1939). Ann. Phys. 426(2), 178–208.
/// - van Hoof PAM et al. (2014). MNRAS 444, 420–428.
#[must_use]
pub fn gaunt_factor_thermal(frequency: f64, temperature: f64) -> f64 {
    if frequency <= 0.0 || temperature <= 0.0 {
        return 1.0;
    }
    let h_nu = PLANCK_CONSTANT * frequency;
    let k_t = BOLTZMANN_CONSTANT * temperature;
    if h_nu >= k_t {
        // Hard X-ray limit: g_ff → 1
        1.0
    } else {
        // Optical/UV limit: Elwert Coulomb logarithm
        let g = (3.0_f64).sqrt() / PI * (2.0 * k_t / h_nu).ln();
        g.clamp(1.0, 10.0)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Plasma state
// ────────────────────────────────────────────────────────────────────────────

/// Self-consistent equilibrium plasma state computed from Saha–Boltzmann kinetics.
///
/// All densities in m⁻³.
#[derive(Debug, Clone)]
pub struct PlasmaState {
    /// Electron temperature [K]
    pub temperature: f64,
    /// Electron number density n_e [m⁻³]
    pub electron_density: f64,
    /// Effective Z²-weighted ion number density n_i [m⁻³]
    ///
    /// Used in the bremsstrahlung formula as `n_ion` (includes charge weighting
    /// so that Σ_j Z_j² n_j is correctly accounted for).
    pub ion_density_z2: f64,
    /// Neutral atom number density n₀ [m⁻³]
    pub neutral_density: f64,
    /// Mean ion charge (dimensionless)
    pub mean_charge: f64,
    /// Total ionization fraction x = n_e / n_total
    pub ionization_fraction: f64,
}

impl PlasmaState {
    /// Compute the equilibrium plasma state for a **single-stage** ionizable gas.
    ///
    /// ## Algorithm
    ///
    /// 1. Compute K(T) = 2 (U₁/U₀) K₀ T^{3/2} exp(−χ/kT)
    /// 2. Solve the quadratic: x² + s·x − s = 0 (s = K/n_total)
    ///    for x = n_e / n_total, the ionization fraction
    /// 3. Compute n_e = x × n_total, n_i = n_e, n₀ = (1−x) × n_total
    ///
    /// ## Arguments
    ///
    /// * `temperature` — temperature [K]
    /// * `pressure` — total pressure [Pa] (used via ideal gas law: n_total = P/kT)
    /// * `ionization_energy` — first ionization energy [eV]
    /// * `partition_ratio` — U₁/U₀ (statistical weight ratio of ion to neutral;
    ///   use 1.0 as a conservative default)
    ///
    /// ## References
    ///
    /// Saha (1920); Rybicki & Lightman (1979) eq. (9.48).
    #[must_use]
    pub fn from_single_stage(
        temperature: f64,
        pressure: f64,
        ionization_energy: f64,
        partition_ratio: f64,
    ) -> Self {
        if temperature <= 0.0 || pressure <= 0.0 {
            return Self::cold(pressure, temperature);
        }

        let e_ion = ionization_energy * ELECTRON_CHARGE; // eV → J
        let k_t = BOLTZMANN_CONSTANT * temperature;
        let n_total = pressure / k_t;

        // Saha equilibrium constant K(T):
        //   K = 2 × (U₁/U₀) × K₀ × T^{3/2} × exp(−χ/kT)
        // Factor of 2: electron spin degeneracy (Saha 1920; Rybicki & Lightman eq. 9.48)
        let k_saha =
            2.0 * partition_ratio * saha_k0() * temperature.powf(1.5) * (-e_ion / k_t).exp();

        // Dimensionless Saha factor s = K / n_total
        let s = k_saha / n_total;

        // Solve x²/(1−x) = s → x² + s·x − s = 0
        // Positive root: x = (−s + √(s² + 4s)) / 2
        let ionization_fraction = if s <= 0.0 {
            0.0
        } else {
            let disc = (s * s + 4.0 * s).sqrt();
            0.5 * (-s + disc)
        }
        .clamp(0.0, 1.0);

        let n_e = ionization_fraction * n_total;
        let n_i = n_e; // singly ionized: n_i = n_e

        Self {
            temperature,
            electron_density: n_e,
            ion_density_z2: n_i, // Z² = 1² = 1 for singly ionized
            neutral_density: (1.0 - ionization_fraction) * n_total,
            mean_charge: 1.0,
            ionization_fraction,
        }
    }

    /// Compute the equilibrium plasma state for **noble gases** (multi-stage ionization).
    ///
    /// Handles up to two ionization stages simultaneously using the Newton–Raphson
    /// method to find the electron density that satisfies the coupled Saha equations.
    ///
    /// ## Algorithm
    ///
    /// Let n_total = P/(kT) be the total particle density (atomic + ionic + electron).
    /// Charge neutrality: n_e = n₁ + 2n₂ (singly and doubly ionized).
    /// Atom conservation: n₀ + n₁ + n₂ = n_atom.
    ///
    /// Newton–Raphson iteration on n_e converges in ≤ 20 steps for all T.
    ///
    /// ## Arguments
    ///
    /// * `temperature` — temperature [K]
    /// * `pressure` — total pressure [Pa]
    /// * `species` — noble gas species (predefined ionization energies)
    ///
    /// ## References
    ///
    /// Raizer YP (1991). Gas Discharge Physics. §1.3.
    #[must_use]
    pub fn from_noble_gas(temperature: f64, pressure: f64, species: NobleGas) -> Self {
        if temperature <= 0.0 || pressure <= 0.0 {
            return Self::cold(pressure, temperature);
        }

        let k_t = BOLTZMANN_CONSTANT * temperature;
        // n_atom = P / (kT), treating all particles as atoms for the atom balance
        // (ideal gas: n_total = n₀ + n₁ + n₂ + n_e; iterative)
        let n_atom_approx = pressure / k_t;

        let (chi1_ev, chi2_ev) = species.ionization_energies_ev();
        let chi1 = chi1_ev * ELECTRON_CHARGE;
        let chi2 = chi2_ev * ELECTRON_CHARGE;

        let k0 = saha_k0() * temperature.powf(1.5);

        // Saha constants for each stage:
        // K₁ = 2 × K₀ × T^{3/2} × exp(−χ₁/kT)  [with U₁/U₀ ≈ 1 for simplicity]
        // K₂ = 2 × K₀ × T^{3/2} × exp(−χ₂/kT)
        let k1 = 2.0 * k0 * (-chi1 / k_t).exp();
        let k2 = 2.0 * k0 * (-chi2 / k_t).exp();

        // Newton–Raphson on n_e:
        //
        // Given n_e, solve for n₀, n₁, n₂:
        //   n₁ = n₀ × K₁ / n_e
        //   n₂ = n₁ × K₂ / n_e = n₀ × K₁ K₂ / n_e²
        //
        // Atom balance: n₀ + n₁ + n₂ = n_atom
        //   n₀ = n_atom / (1 + K₁/n_e + K₁K₂/n_e²)
        //
        // Charge balance: n_e = n₁ + 2n₂ = n₀(K₁/n_e + 2K₁K₂/n_e²) = f(n_e)
        //
        // Residual: r(n_e) = n_e − f(n_e)
        // Derivative: dr/d(n_e) computed analytically for Newton step.

        let n_atom = n_atom_approx; // approximate (n_e << n_total at low ionization)
        let k1k2 = k1 * k2;

        // Initial guess: single-stage Saha
        let s1 = k1 / n_atom;
        let ne_init = if s1 < 1e-10 {
            1.0
        } else {
            0.5 * (-s1 + (s1 * s1 + 4.0 * s1).sqrt()) * n_atom
        }
        .max(1.0);

        let mut n_e = ne_init;

        for _ in 0..30 {
            let ne2 = n_e * n_e;
            let denom = ne2 + k1 * n_e + k1k2;
            if denom <= 0.0 {
                break;
            }
            let n0 = n_atom * ne2 / denom;
            let n1 = n0 * k1 / n_e;
            let n2 = n1 * k2 / n_e;

            // Residual: f(n_e) = n₁ + 2n₂ − n_e
            let f = n1 + 2.0 * n2 - n_e;
            if f.abs() < 1e-6 * n_e.max(1.0) {
                break;
            }

            // Analytical derivative df/d(n_e)
            // d(n₁)/d(n_e) = K₁ × d(n₀/n_e)/d(n_e) = K₁ × [n₀'/n_e − n₀/n_e²]
            // (lengthy but exact; use numerical finite-diff for simplicity)
            let ne_step = n_e * 1e-6;
            let ne_p = n_e + ne_step;
            let ne2_p = ne_p * ne_p;
            let denom_p = ne2_p + k1 * ne_p + k1k2;
            let n0_p = n_atom * ne2_p / denom_p;
            let n1_p = n0_p * k1 / ne_p;
            let n2_p = n1_p * k2 / ne_p;
            let f_p = n1_p + 2.0 * n2_p - ne_p;
            let df = (f_p - f) / ne_step;

            if df.abs() < 1e-300 {
                break;
            }
            let delta = f / df;
            n_e = (n_e - delta).max(1.0);
        }

        n_e = n_e.max(0.0);

        let ne2 = n_e * n_e;
        let denom = ne2 + k1 * n_e + k1k2;
        let (n0, n1, n2) = if denom > 0.0 {
            let n0 = n_atom * ne2 / denom;
            let n1 = n0 * k1 / n_e;
            let n2 = n1 * k2 / n_e.max(1.0);
            (n0, n1, n2)
        } else {
            (n_atom, 0.0, 0.0)
        };

        let n_e_total = n1 + 2.0 * n2;
        let mean_z = if n1 + n2 > 0.0 {
            (n1 + 2.0 * n2) / (n1 + n2)
        } else {
            1.0
        };

        // Z²-weighted ion density for bremsstrahlung: Σ Z_j² n_j = 1²×n₁ + 2²×n₂
        let ion_density_z2 = n1 + 4.0 * n2;
        let ionization_fraction = n_e_total / (n0 + n1 + n2).max(1.0);

        Self {
            temperature,
            electron_density: n_e_total,
            ion_density_z2,
            neutral_density: n0,
            mean_charge: mean_z,
            ionization_fraction,
        }
    }

    /// Cold (un-ionized) plasma state.
    fn cold(pressure: f64, temperature: f64) -> Self {
        let n_total = if temperature > 0.0 && pressure > 0.0 {
            pressure / (BOLTZMANN_CONSTANT * temperature)
        } else {
            0.0
        };
        Self {
            temperature,
            electron_density: 0.0,
            ion_density_z2: 0.0,
            neutral_density: n_total,
            mean_charge: 0.0,
            ionization_fraction: 0.0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Noble gas species
// ────────────────────────────────────────────────────────────────────────────

/// Supported noble gas species for multi-stage ionization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NobleGas {
    /// Helium: χ₁ = 24.59 eV
    Helium,
    /// Argon: χ₁ = 15.76 eV, χ₂ = 27.63 eV
    ///
    /// Argon is the most common gas used in SBSL experiments.
    Argon,
    /// Xenon: χ₁ = 12.13 eV, χ₂ = 20.98 eV
    Xenon,
    /// Krypton: χ₁ = 14.00 eV, χ₂ = 24.36 eV
    Krypton,
}

impl NobleGas {
    /// First and second ionization energies in eV.
    ///
    /// Source: NIST Atomic Spectra Database (2023).
    #[must_use]
    pub fn ionization_energies_ev(self) -> (f64, f64) {
        match self {
            Self::Helium => (24.587, 54.418),  // He → He⁺ → He²⁺
            Self::Argon => (15.760, 27.630),   // Ar → Ar⁺ → Ar²⁺
            Self::Xenon => (12.130, 20.975),   // Xe → Xe⁺ → Xe²⁺
            Self::Krypton => (13.999, 24.360), // Kr → Kr⁺ → Kr²⁺
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Bremsstrahlung model
// ────────────────────────────────────────────────────────────────────────────

/// Bremsstrahlung (free-free) radiation model for sonoluminescence.
///
/// The model computes the thermal bremsstrahlung emission from a hot plasma
/// using the Gaunt-factor corrected formula of Rybicki & Lightman (1979).
///
/// The plasma state (n_e, n_i) can be provided externally or computed
/// self-consistently from temperature and pressure using the Saha equation
/// via [`PlasmaState`].
#[derive(Debug, Clone)]
pub struct BremsstrahlungModel {
    /// Average ion charge number Z (default: 1, singly ionized)
    pub z_ion: f64,
    /// Use temperature/frequency-dependent Gaunt factor (recommended: true)
    pub use_thermal_gaunt_factor: bool,
    /// Fixed Gaunt factor (used only when `use_thermal_gaunt_factor = false`)
    pub fixed_gaunt_factor: f64,
}

impl Default for BremsstrahlungModel {
    fn default() -> Self {
        Self {
            z_ion: 1.0,
            use_thermal_gaunt_factor: true,
            fixed_gaunt_factor: 1.2, // Typical value for optical bremsstrahlung
        }
    }
}

impl BremsstrahlungModel {
    /// Bremsstrahlung emission coefficient per unit volume, per unit frequency,
    /// per steradian [W m⁻³ Hz⁻¹ sr⁻¹].
    ///
    /// ## Formula
    ///
    /// ```text
    /// j_ν^ff = n_e n_i Z² g_ff(ν,T) × C_ff × T^{−1/2} × exp(−hν/kT)
    /// ```
    ///
    /// where `C_ff = 6.8×10⁻⁵¹ W m³ K^{1/2} sr⁻¹` (Rybicki & Lightman 1979, eq. 5.14a,
    /// converted to SI per steradian).
    ///
    /// ## Arguments
    ///
    /// * `frequency` — radiation frequency ν [Hz]
    /// * `temperature` — electron temperature T_e [K]
    /// * `n_electron` — electron number density n_e [m⁻³]
    /// * `n_ion` — Z²-weighted ion number density n_i [m⁻³]
    ///   (= Σ_j Z_j² n_j; for singly ionized: n_i = n_e)
    ///
    /// ## References
    ///
    /// Rybicki GB, Lightman AP (1979). Radiative Processes §5.2, eq. (5.14a).
    #[must_use]
    pub fn emission_coefficient(
        &self,
        frequency: f64,
        temperature: f64,
        n_electron: f64,
        n_ion: f64,
    ) -> f64 {
        if temperature <= 0.0 || frequency <= 0.0 || n_electron <= 0.0 || n_ion <= 0.0 {
            return 0.0;
        }

        let g_ff = if self.use_thermal_gaunt_factor {
            gaunt_factor_thermal(frequency, temperature)
        } else {
            self.fixed_gaunt_factor
        };

        let h_nu = PLANCK_CONSTANT * frequency;
        let exp_factor = (-h_nu / (BOLTZMANN_CONSTANT * temperature)).exp();

        c_ff_per_sr() * self.z_ion.powi(2) * g_ff * n_electron * n_ion / temperature.sqrt()
            * exp_factor
    }

    /// Spectral radiance of bremsstrahlung at a given wavelength [W m⁻² sr⁻¹ m⁻¹].
    ///
    /// Converts the frequency-space emission coefficient to wavelength-space using
    /// `dν/dλ = −c/λ²`, and multiplies by path length for surface radiance.
    ///
    /// ```text
    /// L_λ = j_ν^ff × (c/λ²) × path_length   [W m⁻² sr⁻¹ m⁻¹]
    /// ```
    #[must_use]
    pub fn spectral_radiance(
        &self,
        wavelength: f64,
        temperature: f64,
        n_electron: f64,
        n_ion: f64,
        path_length: f64,
    ) -> f64 {
        if wavelength <= 0.0 || path_length <= 0.0 {
            return 0.0;
        }
        let frequency = SPEED_OF_LIGHT / wavelength;
        let j_nu = self.emission_coefficient(frequency, temperature, n_electron, n_ion);
        // dν/dλ factor: |dν/dλ| = c/λ²
        j_nu * SPEED_OF_LIGHT / wavelength.powi(2) * path_length
    }

    /// Total bremsstrahlung power per unit volume integrated over all frequencies [W m⁻³].
    ///
    /// From the Stefan–Boltzmann-like integration of j_ν over all frequencies:
    ///
    /// ```text
    /// P_total = ∫₀^∞ 4π j_ν^ff dν = C_total × Z² g_ff n_e n_i T^{1/2}
    /// ```
    ///
    /// where `C_total = C_ff × 4π × kT/h` (from integrating exp(−hν/kT) over ν).
    ///
    /// Numerically: `C_total ≈ 1.69×10⁻³⁸ W m³ K^{−1/2}` (Raizer 1991 §1).
    ///
    /// ## References
    ///
    /// Raizer YP (1991). Gas Discharge Physics. Springer. §1.3.
    #[must_use]
    pub fn total_power(&self, temperature: f64, n_electron: f64, n_ion: f64, volume: f64) -> f64 {
        if temperature <= 0.0 || n_electron <= 0.0 || n_ion <= 0.0 || volume <= 0.0 {
            return 0.0;
        }

        // Use fixed Gaunt factor for the frequency-integrated formula (typical value = 1.2)
        let g_ff = self.fixed_gaunt_factor;

        // C_total = C_ff × 4π × kT/h  (from ∫₀^∞ exp(−hν/kT) dν = kT/h)
        let c_total = c_ff_per_sr() * 4.0 * PI * BOLTZMANN_CONSTANT * temperature / PLANCK_CONSTANT;

        c_total * self.z_ion.powi(2) * g_ff * n_electron * n_ion * temperature.sqrt().recip()
            * temperature.sqrt() // T^{1/2} net (T^{−1/2} from C_ff × T^{+1} from kT/h)
            * volume
    }

    /// Compute ionization fraction using the corrected single-stage Saha equation.
    ///
    /// ## Formula
    ///
    /// ```text
    /// K(T) = 2 × (U₁/U₀) × (2π m_e k T / h²)^{3/2} × exp(−χ/kT)
    /// ```
    ///
    /// The factor **2** is the electron spin degeneracy (Saha 1920; R&L eq. 9.48).
    /// The partition function ratio U₁/U₀ defaults to 1.0 (appropriate for
    /// hydrogen-like atoms; for argon see NIST tables).
    ///
    /// ## Arguments
    ///
    /// * `temperature` — temperature [K]
    /// * `pressure` — gas pressure [Pa]
    /// * `ionization_energy` — first ionization energy [eV]
    ///
    /// ## Returns
    ///
    /// Ionization fraction x = n_e / n_total ∈ [0, 1].
    #[must_use]
    pub fn saha_ionization(&self, temperature: f64, pressure: f64, ionization_energy: f64) -> f64 {
        let state = PlasmaState::from_single_stage(temperature, pressure, ionization_energy, 1.0);
        state.ionization_fraction
    }

    /// Compute full emission spectrum over a wavelength array.
    #[must_use]
    pub fn emission_spectrum(
        &self,
        temperature: f64,
        n_electron: f64,
        n_ion: f64,
        path_length: f64,
        wavelengths: &Array1<f64>,
    ) -> Array1<f64> {
        wavelengths.mapv(|lambda| {
            self.spectral_radiance(lambda, temperature, n_electron, n_ion, path_length)
        })
    }

    /// Compute self-consistent emission coefficient from temperature and pressure.
    ///
    /// Uses the Saha equation to determine n_e and n_i internally, then computes
    /// the emission coefficient.  This is the recommended interface for single-species
    /// gases.
    ///
    /// ## Arguments
    ///
    /// * `frequency` — radiation frequency [Hz]
    /// * `temperature` — temperature [K]
    /// * `pressure` — gas pressure [Pa]
    /// * `ionization_energy` — first ionization energy [eV]
    #[must_use]
    pub fn emission_from_temperature_pressure(
        &self,
        frequency: f64,
        temperature: f64,
        pressure: f64,
        ionization_energy: f64,
    ) -> f64 {
        let state = PlasmaState::from_single_stage(temperature, pressure, ionization_energy, 1.0);
        self.emission_coefficient(
            frequency,
            temperature,
            state.electron_density,
            state.ion_density_z2,
        )
    }
}

/// Calculate total bremsstrahlung emission field (power density [W m⁻³]).
#[must_use]
pub fn calculate_bremsstrahlung_emission(
    temperature_field: &Array3<f64>,
    electron_density_field: &Array3<f64>,
    ion_density_field: &Array3<f64>,
    model: &BremsstrahlungModel,
) -> Array3<f64> {
    let mut emission_field = Array3::zeros(temperature_field.dim());

    Zip::from(&mut emission_field)
        .and(temperature_field)
        .and(electron_density_field)
        .and(ion_density_field)
        .for_each(|out, &temp, &n_electron, &n_ion| {
            if n_electron > 0.0 && n_ion > 0.0 && temp > 0.0 {
                *out = model.total_power(temp, n_electron, n_ion, 1.0);
            }
        });

    emission_field
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Gaunt factor tests ────────────────────────────────────────────────

    /// Gaunt factor must be ≥ 1.0 for all inputs.
    #[test]
    fn gaunt_factor_lower_bound() {
        let cases = [(1e12, 10_000.0), (1e15, 20_000.0), (1e18, 100_000.0)];
        for (freq, temp) in cases {
            let g = gaunt_factor_thermal(freq, temp);
            assert!(g >= 1.0, "g_ff({freq}, {temp}) = {g} < 1.0");
            assert!(g <= 10.0, "g_ff({freq}, {temp}) = {g} > 10.0");
        }
    }

    /// At hν ≫ kT (hard X-ray), Gaunt factor must equal 1.0.
    #[test]
    fn gaunt_factor_hard_xray_limit() {
        // kT at 10,000 K: 1.381e-23 × 10000 ≈ 1.38e-19 J
        // hν at 1e20 Hz: 6.626e-34 × 1e20 = 6.626e-14 J >> kT
        let g = gaunt_factor_thermal(1e20, 10_000.0);
        assert_eq!(g, 1.0, "hard X-ray Gaunt factor must be 1.0");
    }

    /// Gaunt factor must increase with temperature (hotter plasma → larger Coulomb log).
    #[test]
    fn gaunt_factor_increases_with_temperature() {
        let freq = 1e14; // optical
        let g1 = gaunt_factor_thermal(freq, 10_000.0);
        let g2 = gaunt_factor_thermal(freq, 50_000.0);
        assert!(
            g2 > g1,
            "g_ff should increase with T at fixed ν (Coulomb log)"
        );
    }

    // ── Saha equation tests ───────────────────────────────────────────────

    /// Saha ionization fraction for hydrogen at 10,000 K and 1 atm.
    ///
    /// ## Reference value
    ///
    /// From the Saha equation with χ_H = 13.6 eV:
    /// K(10,000K) = 2 × (2π m_e k × 10000 / h²)^{3/2} × exp(−13.6×11605/10000)
    ///            ≈ 6.9×10²⁰ m⁻³
    /// n_total = 101325/(k × 10000) ≈ 7.34×10²³ m⁻³
    /// s = K/n ≈ 9.4×10⁻⁴
    /// x = (−s + √(s²+4s))/2 ≈ 0.030 (3%)
    ///
    /// Reference: Carroll & Ostlie (2017) "Modern Astrophysics" §8.1.
    #[test]
    fn saha_hydrogen_10000k_1atm() {
        let model = BremsstrahlungModel::default();
        let x = model.saha_ionization(10_000.0, 101_325.0, 13.6);

        // Physical range: 1–5 % at 10,000 K, 1 atm
        assert!(
            x > 0.01 && x < 0.10,
            "H ionization at 10,000K, 1 atm: x = {x:.4}, expected 1–10%"
        );
    }

    /// Hydrogen must be nearly fully ionized at 50,000 K.
    #[test]
    fn saha_hydrogen_fully_ionized_at_50000k() {
        let model = BremsstrahlungModel::default();
        let x = model.saha_ionization(50_000.0, 101_325.0, 13.6);
        assert!(x > 0.95, "H at 50,000 K must be >95 % ionized, got {x:.4}");
    }

    /// Saha ionization fraction must be strictly in (0, 1).
    #[test]
    fn saha_output_in_valid_range() {
        let model = BremsstrahlungModel::default();
        for (t, p) in [(5_000.0, 1e5), (20_000.0, 1e5), (100_000.0, 1e6)] {
            let x = model.saha_ionization(t, p, 13.6);
            assert!(
                (0.0..=1.0).contains(&x),
                "x({t}K, {p}Pa) = {x} out of [0,1]"
            );
        }
    }

    /// Ionization fraction must increase monotonically with temperature.
    #[test]
    fn saha_increases_with_temperature() {
        let model = BremsstrahlungModel::default();
        let temps = [5_000.0, 8_000.0, 12_000.0, 20_000.0, 50_000.0];
        let fracs: Vec<f64> = temps
            .iter()
            .map(|&t| model.saha_ionization(t, 101_325.0, 13.6))
            .collect();

        for i in 1..fracs.len() {
            assert!(
                fracs[i] >= fracs[i - 1],
                "Ionization must increase with T: x({}) = {:.5} < x({}) = {:.5}",
                temps[i],
                fracs[i],
                temps[i - 1],
                fracs[i - 1]
            );
        }
    }

    // ── Plasma state tests ────────────────────────────────────────────────

    /// PlasmaState charge neutrality: n_e ≈ n_i for singly-ionized gas.
    #[test]
    fn plasma_state_charge_neutrality() {
        let state = PlasmaState::from_single_stage(20_000.0, 1e5, 13.6, 1.0);
        // For singly ionized: n_e = n_i (Z²=1)
        let rel_err =
            (state.electron_density - state.ion_density_z2).abs() / state.electron_density.max(1.0);
        assert!(
            rel_err < 1e-10,
            "Charge neutrality violated: rel_err = {rel_err}"
        );
    }

    /// Noble gas argon at 20,000 K: first ionization should dominate.
    #[test]
    fn argon_plasma_first_ionization_dominant_at_20kk() {
        let state = PlasmaState::from_noble_gas(20_000.0, 1e5, NobleGas::Argon);
        // At 20,000 K, argon is partially ionized (χ₁ = 15.76 eV)
        assert!(
            state.ionization_fraction > 0.0,
            "Argon must have nonzero ionization"
        );
        assert!(
            state.ionization_fraction < 1.0,
            "Argon should not be fully ionized at 20,000 K"
        );
    }

    /// Noble gas argon at 100,000 K: highly ionized (both stages active).
    #[test]
    fn argon_plasma_highly_ionized_at_100kk() {
        let state = PlasmaState::from_noble_gas(100_000.0, 1e5, NobleGas::Argon);
        assert!(
            state.ionization_fraction > 0.5,
            "Argon at 100,000 K should be >50% ionized, got {:.4}",
            state.ionization_fraction
        );
    }

    // ── Emission coefficient tests ────────────────────────────────────────

    /// Emission coefficient must be positive and finite.
    #[test]
    fn emission_coefficient_positive_finite() {
        let model = BremsstrahlungModel::default();
        let j = model.emission_coefficient(1e15, 20_000.0, 1e24, 1e24);
        assert!(
            j > 0.0 && j.is_finite(),
            "emission_coefficient must be positive finite, got {j}"
        );
    }

    /// Emission coefficient must decrease with frequency (exponential cutoff).
    #[test]
    fn emission_coefficient_decreases_with_frequency() {
        let model = BremsstrahlungModel::default();
        let j_low = model.emission_coefficient(1e14, 20_000.0, 1e24, 1e24);
        let j_high = model.emission_coefficient(1e16, 20_000.0, 1e24, 1e24);
        assert!(j_high < j_low, "Emission must decrease with ν");
    }

    /// Emission coefficient must be proportional to n_e × n_i.
    #[test]
    fn emission_coefficient_quadratic_in_density() {
        let model = BremsstrahlungModel {
            use_thermal_gaunt_factor: false,
            ..Default::default()
        };
        let j1 = model.emission_coefficient(1e15, 20_000.0, 1e24, 1e24);
        let j2 = model.emission_coefficient(1e15, 20_000.0, 2e24, 2e24);
        let ratio = j2 / j1;
        assert!(
            (ratio - 4.0).abs() < 1e-10,
            "Emission must scale as (n_e × n_i) ∝ 4×: ratio = {ratio}"
        );
    }

    /// Magnitude check against Rybicki & Lightman (1979) reference constant.
    ///
    /// The per-steradian emission coefficient should satisfy:
    ///   j_ν ≈ 6.8×10⁻⁵¹ × n_e × n_i × T^{−1/2} × g_ff × exp(−hν/kT)
    ///
    /// At T=20,000K, n_e=n_i=1e24 m⁻³, ν=1e15 Hz, g_ff=1 (fixed):
    ///   j_ν ≈ 6.8e-51 × 1e24 × 1e24 × (20000)^{-0.5} × exp(−hν/kT)
    ///
    /// Tolerance: ±10% (accounts for different Gaunt factor conventions).
    #[test]
    fn emission_coefficient_magnitude_rybicki_lightman() {
        let model = BremsstrahlungModel {
            z_ion: 1.0,
            use_thermal_gaunt_factor: false,
            fixed_gaunt_factor: 1.0,
        };

        let temperature = 20_000.0_f64;
        let n_e = 1e24_f64;
        let frequency = 1e15_f64;

        let j_computed = model.emission_coefficient(frequency, temperature, n_e, n_e);

        // R&L reference value: C_ff = 6.8e-51 W m³ K^{1/2} sr⁻¹ (per steradian)
        let c_ff_ref = 6.8e-51_f64;
        let h_nu = PLANCK_CONSTANT * frequency;
        let k_t = BOLTZMANN_CONSTANT * temperature;
        let j_reference = c_ff_ref * n_e * n_e * temperature.powf(-0.5) * (-h_nu / k_t).exp();

        let rel_err = (j_computed - j_reference).abs() / j_reference;
        assert!(
            rel_err < 0.10,
            "j_ν = {j_computed:.3e} W/(m³·Hz·sr), R&L reference = {j_reference:.3e}, rel_err = {rel_err:.3}"
        );
    }

    /// Self-consistent emission must increase with temperature (more ionization + hotter plasma).
    #[test]
    fn emission_from_temperature_pressure_increases_with_temperature() {
        let model = BremsstrahlungModel::default();
        let freq = 1e14; // optical UV
        let pressure = 1e8; // high pressure (dense bubble at collapse)
        let e_ion = 15.76; // argon

        let j1 = model.emission_from_temperature_pressure(freq, 20_000.0, pressure, e_ion);
        let j2 = model.emission_from_temperature_pressure(freq, 50_000.0, pressure, e_ion);

        assert!(
            j2 > j1,
            "Emission must increase with temperature (more ionization + hotter)"
        );
    }
}
