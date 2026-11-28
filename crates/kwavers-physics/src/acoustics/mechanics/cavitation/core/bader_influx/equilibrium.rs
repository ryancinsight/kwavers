use kwavers_core::constants::fundamental::{AVOGADRO, BOLTZMANN};
use std::f64::consts::PI;

/// Gas content [molecules] of a bubble at mechanical equilibrium:
/// `n = (p0 + 2σ/R)·(4π/3)·R³/(k_B·T)`. The forward direction of the
/// implicit residual-radius map.
#[must_use]
pub fn equilibrium_gas_content(
    radius: f64,
    ambient_pressure: f64,
    surface_tension: f64,
    temperature: f64,
) -> f64 {
    if radius <= 0.0 || temperature <= 0.0 {
        return 0.0;
    }
    (ambient_pressure + 2.0 * surface_tension / radius) * (4.0 / 3.0) * PI * radius.powi(3)
        / (BOLTZMANN * temperature)
}

/// `dn/dR` [molecules/m] of the equilibrium gas-content map:
/// `(4π·p0·R² + 16π·σ·R/3)/(k_B·T)`. The conversion between a molar /
/// molecular flux and a radius rate in quasi-static integration.
#[must_use]
pub fn equilibrium_gas_content_derivative(
    radius: f64,
    ambient_pressure: f64,
    surface_tension: f64,
    temperature: f64,
) -> f64 {
    if temperature <= 0.0 {
        return 0.0;
    }
    (4.0 * PI * ambient_pressure * radius * radius + 16.0 * PI * surface_tension * radius / 3.0)
        / (BOLTZMANN * temperature)
}

/// Solve `R³·(P + 2σ/R) = 3·k_B·T·n/(4π)` for `R` [m] by bisection —
/// the equilibrium radius of a bubble holding `gas_molecules` under a
/// total external pressure `external_pressure` [Pa].
///
/// With `external_pressure = p0` this is the implicit residual-radius
/// map; with `external_pressure = p0 + p+` it is the pressure-balance
/// (collapse-floor) radius of the Build 6 outflux analysis — a bubble
/// **can** contract below its `p0`-equilibrium radius when the
/// external pressure exceeds `p0` by ~100×, so the collapse floor is
/// this radius, not the residual radius.
#[must_use]
pub fn equilibrium_radius_for_pressure(
    gas_molecules: f64,
    external_pressure: f64,
    surface_tension: f64,
    temperature: f64,
) -> f64 {
    if gas_molecules <= 0.0 || external_pressure <= 0.0 || temperature <= 0.0 {
        return 0.0;
    }
    let rhs = 3.0 * BOLTZMANN * temperature * gas_molecules / (4.0 * PI);
    let f = |r: f64| external_pressure * r.powi(3) + 2.0 * surface_tension * r.powi(2) - rhs;
    let (mut lo, mut hi) = (1e-12_f64, 1e-2_f64);
    if f(lo) > 0.0 {
        return lo;
    }
    if f(hi) < 0.0 {
        return hi;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Residual (equilibrium) radius [m] from gas content [molecules] —
/// the implicit mechanical-equilibrium map
/// `R³·(p0 + 2σ/R) = 3·k_B·T·n/(4π)`. The formation model's primary
/// map; see the module docs for the arbitration against the
/// as-printed form.
#[must_use]
pub fn residual_radius_equilibrium(
    gas_molecules: f64,
    ambient_pressure: f64,
    surface_tension: f64,
    temperature: f64,
) -> f64 {
    equilibrium_radius_for_pressure(
        gas_molecules,
        ambient_pressure,
        surface_tension,
        temperature,
    )
}

/// Residual radius [m] from gas content [molecules] — the as-printed
/// Bader Eq. (8) map, `R = [3·k_B·T·n/(4π·(p0 + 2σ/R_nuc))]^(1/3)`,
/// with the Laplace pressure evaluated at the **pre-exposure nucleus
/// radius** `nucleus_radius` [m].
///
/// Retained as the published law and as a labelled sensitivity; it is
/// **not** a mechanical equilibrium (see the module docs). Do not
/// silently swap it for [`residual_radius_equilibrium`].
#[must_use]
pub fn residual_radius_as_printed(
    gas_molecules: f64,
    nucleus_radius: f64,
    ambient_pressure: f64,
    surface_tension: f64,
    temperature: f64,
) -> f64 {
    if gas_molecules <= 0.0 || nucleus_radius <= 0.0 || temperature <= 0.0 {
        return 0.0;
    }
    let rhs = 3.0 * BOLTZMANN * temperature * gas_molecules / (4.0 * PI);
    let p_eff = ambient_pressure + 2.0 * surface_tension / nucleus_radius;
    (rhs / p_eff).powf(1.0 / 3.0)
}

/// Avogadro conversion kept beside the laws that need it: molecules
/// per mole (the papers in this family mix the two; see the module
/// docs' units trap).
pub const MOLECULES_PER_MOLE: f64 = AVOGADRO;
