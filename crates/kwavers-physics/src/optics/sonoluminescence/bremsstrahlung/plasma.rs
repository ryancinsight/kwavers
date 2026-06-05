//! Saha-Boltzmann plasma-state solver.

use super::constants::saha_k0;
use super::species::NobleGas;
use kwavers_core::constants::fundamental::{
    BOLTZMANN as BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE as ELECTRON_CHARGE,
};

/// Self-consistent equilibrium plasma state computed from Saha-Boltzmann kinetics.
///
/// All densities use SI units `m^-3`.
#[derive(Debug, Clone)]
pub struct PlasmaState {
    /// Electron temperature (K).
    pub temperature: f64,
    /// Electron number density.
    pub electron_density: f64,
    /// Effective `Z^2`-weighted ion number density.
    pub ion_density_z2: f64,
    /// Neutral atom number density.
    pub neutral_density: f64,
    /// Mean ion charge.
    pub mean_charge: f64,
    /// Total ionization fraction `n_e / n_total`.
    pub ionization_fraction: f64,
}

impl PlasmaState {
    /// Compute a single-stage Saha equilibrium state.
    ///
    /// # Theorem
    ///
    /// With `x=n_e/n_total` and `s=K(T)/n_total`, the single-stage Saha equation
    /// is `x^2/(1-x)=s`. Rearrangement gives `x^2+s x-s=0`, whose physical
    /// root is `(-s+sqrt(s^2+4s))/2`.
    ///
    /// Proof: the second quadratic root is negative for `s>0`, so only the
    /// displayed root lies in the admissible interval `[0,1]`.
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

        let e_ion = ionization_energy * ELECTRON_CHARGE;
        let k_t = BOLTZMANN_CONSTANT * temperature;
        let n_total = pressure / k_t;
        let k_saha =
            2.0 * partition_ratio * saha_k0() * temperature.powf(1.5) * (-e_ion / k_t).exp();
        let s = k_saha / n_total;

        let ionization_fraction = if s <= 0.0 {
            0.0
        } else {
            0.5 * (-s + (s * s + 4.0 * s).sqrt())
        }
        .clamp(0.0, 1.0);

        let n_e = ionization_fraction * n_total;
        Self {
            temperature,
            electron_density: n_e,
            ion_density_z2: n_e,
            neutral_density: (1.0 - ionization_fraction) * n_total,
            mean_charge: 1.0,
            ionization_fraction,
        }
    }

    /// Compute a two-stage noble-gas Saha equilibrium state.
    ///
    /// # Theorem
    ///
    /// Given electron density `n_e`, the stage densities are determined by
    /// `n1=n0 K1/n_e`, `n2=n1 K2/n_e`, and atom conservation. The charge
    /// residual `r(n_e)=n1+2n2-n_e` vanishes at charge neutrality.
    ///
    /// Proof: substituting the Saha ratios into atom conservation gives a
    /// unique `n0(n_e)>0`; inserting the resulting `n1,n2` into charge balance
    /// yields a scalar root problem. Newton iteration preserves positivity by
    /// clamping trial densities to the positive domain.
    #[must_use]
    pub fn from_noble_gas(temperature: f64, pressure: f64, species: NobleGas) -> Self {
        if temperature <= 0.0 || pressure <= 0.0 {
            return Self::cold(pressure, temperature);
        }

        let k_t = BOLTZMANN_CONSTANT * temperature;
        let n_atom = pressure / k_t;
        let (chi1_ev, chi2_ev) = species.ionization_energies_ev();
        let chi1 = chi1_ev * ELECTRON_CHARGE;
        let chi2 = chi2_ev * ELECTRON_CHARGE;

        let k0 = saha_k0() * temperature.powf(1.5);
        let k1 = 2.0 * k0 * (-chi1 / k_t).exp();
        let k2 = 2.0 * k0 * (-chi2 / k_t).exp();
        let k1k2 = k1 * k2;

        let s1 = k1 / n_atom;
        let mut n_e = if s1 < 1e-10 {
            1.0
        } else {
            0.5 * (-s1 + (s1 * s1 + 4.0 * s1).sqrt()) * n_atom
        }
        .max(1.0);

        for _ in 0..30 {
            let (f, df) = charge_residual_and_derivative(n_e, n_atom, k1, k1k2);
            if f.abs() < 1e-6 * n_e.max(1.0) || df.abs() < 1e-300 {
                break;
            }
            n_e = (n_e - f / df).max(1.0);
        }

        let (n0, n1, n2) = stage_densities(n_e.max(0.0), n_atom, k1, k1k2);
        let n_e_total = 2.0f64.mul_add(n2, n1);
        let mean_z = if n1 + n2 > 0.0 {
            2.0f64.mul_add(n2, n1) / (n1 + n2)
        } else {
            1.0
        };
        let ion_density_z2 = 4.0f64.mul_add(n2, n1);
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

fn stage_densities(n_e: f64, n_atom: f64, k1: f64, k1k2: f64) -> (f64, f64, f64) {
    let ne = n_e.max(1.0);
    let ne2 = ne * ne;
    let denom = k1.mul_add(ne, ne2) + k1k2;
    if denom > 0.0 {
        let n0 = n_atom * ne2 / denom;
        let n1 = n0 * k1 / ne;
        let n2 = n0 * k1k2 / ne2;
        (n0, n1, n2)
    } else {
        (n_atom, 0.0, 0.0)
    }
}

fn charge_residual_and_derivative(n_e: f64, n_atom: f64, k1: f64, k1k2: f64) -> (f64, f64) {
    let (_, n1, n2) = stage_densities(n_e, n_atom, k1, k1k2);
    let f = 2.0f64.mul_add(n2, n1) - n_e;
    let ne_step = (n_e * 1e-6).max(1.0);
    let (_, n1_p, n2_p) = stage_densities(n_e + ne_step, n_atom, k1, k1k2);
    let f_p = 2.0f64.mul_add(n2_p, n1_p) - (n_e + ne_step);
    (f, (f_p - f) / ne_step)
}
