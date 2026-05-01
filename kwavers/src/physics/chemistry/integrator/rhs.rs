//! ODE right-hand side and species-order mapping for radical integration.

use crate::physics::chemistry::ros_plasma::radical_kinetics::RadicalKinetics;
use crate::physics::chemistry::ros_plasma::ros_species::ROSSpecies;
use std::collections::{HashMap, HashSet};

/// Evaluate the ODE right-hand side using `RadicalKinetics::calculate_rates`.
pub(super) fn eval_rhs(
    kinetics: &RadicalKinetics,
    species: &[ROSSpecies],
    y: &[f64],
    out: &mut [f64],
) {
    let concs: HashMap<ROSSpecies, f64> = species
        .iter()
        .zip(y.iter())
        .map(|(&s, &v)| (s, v.max(0.0)))
        .collect();
    let rates = kinetics.calculate_rates(&concs);
    for (i, s) in species.iter().enumerate() {
        out[i] = rates.get(s).copied().unwrap_or(0.0);
    }
}

/// Autonomous RHS wrapper; `_t` is retained to keep RK stage calls uniform.
#[inline]
pub(super) fn eval_rhs_at(
    kinetics: &RadicalKinetics,
    species: &[ROSSpecies],
    y: &[f64],
    _t: f64,
    out: &mut [f64],
) {
    eval_rhs(kinetics, species, y, out);
}

/// Build the canonical species list for the integration.
pub(super) fn collect_species(
    kinetics: &RadicalKinetics,
    concentrations: &HashMap<ROSSpecies, f64>,
) -> Vec<ROSSpecies> {
    let mut seen = HashSet::new();
    let mut list = Vec::new();

    for reaction in &kinetics.reactions {
        for (species, _) in &reaction.reactants {
            if seen.insert(*species) {
                list.push(*species);
            }
        }
        for (species, _) in &reaction.products {
            if seen.insert(*species) {
                list.push(*species);
            }
        }
    }

    for &species in concentrations.keys() {
        if seen.insert(species) {
            list.push(species);
        }
    }

    list.sort_by_key(|species| *species as usize);
    list
}
