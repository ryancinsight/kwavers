//! Residual cavitation-gas → acoustic-medium coupling (genuine shielding).
//!
//! A residual microbubble cloud left by a previous pulse modifies the medium the
//! next pulse propagates through. This module applies that coupling to the
//! solver's per-voxel medium arrays: the Wood (1930) sound-speed collapse and
//! the Commander–Prosperetti (1989) excess attenuation, both as functions of the
//! local gas void fraction `β(x)` (from a
//! [`ResidualGasField`](kwavers_simulation-style field) or any β map). The
//! returned arrays feed the heterogeneous-medium update path
//! (`ArrayAccess::sound_speed_array_mut` / `absorption_array_mut`) and the PSTD
//! absorption-operator re-initialisation, so the subsequent pulse is physically
//! shielded rather than empirically derated.

use kwavers_domain::medium::Medium;
use kwavers_physics::acoustics::bubble_dynamics::{
    commander_prosperetti_attenuation, wood_sound_speed,
};
use kwavers_physics::acoustics::mechanics::absorption::np_m_to_power_law_db_cm;
use ndarray::{Array3, ArrayView3};

/// Liquid/gas acoustic properties for the residual-gas medium coupling.
#[derive(Debug, Clone, Copy)]
pub struct BubblyMediumProps {
    /// Gas sound speed [m/s].
    pub c_gas: f64,
    /// Gas density [kg/m³].
    pub rho_gas: f64,
    /// Liquid dynamic viscosity [Pa·s] (damping in the attenuation model).
    pub mu_liquid: f64,
    /// Ambient pressure [Pa].
    pub p0: f64,
    /// Gas polytropic exponent.
    pub polytropic: f64,
    /// Representative residual-bubble radius [m].
    pub bubble_radius: f64,
    /// Drive frequency [Hz] for the attenuation model.
    pub frequency: f64,
}

impl BubblyMediumProps {
    /// Air-in-water defaults at the given representative radius and frequency.
    #[must_use]
    pub fn air_water(bubble_radius_m: f64, frequency_hz: f64) -> Self {
        Self {
            c_gas: 343.0,
            rho_gas: 1.2,
            mu_liquid: 1.0e-3,
            p0: 101_325.0,
            polytropic: 1.4,
            bubble_radius: bubble_radius_m,
            frequency: frequency_hz,
        }
    }
}

/// Medium properties modified by the residual gas cloud, ready to write back
/// into the solver's heterogeneous medium arrays.
#[derive(Debug, Clone)]
pub struct ShieldedMedium {
    /// Wood-modified sound speed [m/s].
    pub sound_speed: Array3<f64>,
    /// Base absorption + Commander–Prosperetti excess attenuation [Np/m].
    pub absorption: Array3<f64>,
}

/// Apply residual-gas shielding to the base medium arrays.
///
/// For each voxel with gas void fraction `β`:
/// * `c ← wood_sound_speed(β, c_base, ρ_base, c_gas, ρ_gas)` — collapses toward
///   a few hundred m/s for `β ≳ 10⁻⁴`, creating an impedance mismatch;
/// * `α ← α_base + commander_prosperetti(β, …)` — adds resonant-scattering
///   absorption that peaks near the bubble resonance.
///
/// Voxels with `β = 0` are returned unchanged. The arrays must share a shape.
#[must_use]
pub fn apply_residual_gas_shielding(
    base_sound_speed: ArrayView3<'_, f64>,
    base_absorption: ArrayView3<'_, f64>,
    base_density: ArrayView3<'_, f64>,
    void_fraction: ArrayView3<'_, f64>,
    props: &BubblyMediumProps,
) -> ShieldedMedium {
    let dim = base_sound_speed.dim();
    let mut sound_speed = base_sound_speed.to_owned();
    let mut absorption = base_absorption.to_owned();
    if base_absorption.dim() != dim || base_density.dim() != dim || void_fraction.dim() != dim {
        return ShieldedMedium {
            sound_speed,
            absorption,
        };
    }
    ndarray::Zip::from(&mut sound_speed)
        .and(&mut absorption)
        .and(base_density)
        .and(void_fraction)
        .for_each(|c, a, &rho, &beta| {
            if beta > 0.0 {
                let c_base_local = *c; // local liquid sound speed before modification
                *a += commander_prosperetti_attenuation(
                    props.frequency,
                    beta,
                    props.bubble_radius,
                    c_base_local,
                    rho,
                    props.mu_liquid,
                    props.p0,
                    props.polytropic,
                );
                *c = wood_sound_speed(beta, c_base_local, rho, props.c_gas, props.rho_gas);
            }
        });
    ShieldedMedium {
        sound_speed,
        absorption,
    }
}

/// Apply only the Wood (1930) sound-speed collapse in place to a heterogeneous
/// medium (impedance mismatch + slowdown / refraction), leaving absorption
/// untouched.
///
/// This bakes the quasi-static (Wood) sound-speed collapse into the medium for a
/// narrowband / single-frequency workflow. Returns `true` if the medium exposed a
/// mutable sound-speed array, `false` otherwise.
///
/// # Do not combine with the broadband operator
/// The frequency-exact operator
/// [`PSTDSolver::set_residual_gas_absorption`](crate::pstd::PSTDSolver::set_residual_gas_absorption)
/// already carries the full dispersive phase velocity `c_p(ω)`, whose `ω→0`
/// limit reproduces this Wood speed. The **complete, approximation-free** path is
/// therefore the operator alone on the *unmodified host medium* — applying Wood
/// here as well would double-count the low-frequency speed change. Use this
/// function only for a Wood-only (quasi-static, no-loss) study or as part of the
/// self-contained narrowband [`apply_residual_gas_shielding_in_place`].
pub fn apply_residual_gas_wood_in_place(
    medium: &mut dyn Medium,
    void_fraction: ArrayView3<'_, f64>,
    props: &BubblyMediumProps,
) -> bool {
    let dim = void_fraction.dim();
    let base_rho = medium.density_array().to_owned();
    if base_rho.dim() != dim {
        return false;
    }
    if let Some(mut c_arr) = medium.sound_speed_array_mut() {
        if c_arr.dim() == dim {
            // The current array value equals the base liquid speed (not yet
            // modified), so use it directly as the host sound speed.
            ndarray::Zip::from(&mut c_arr)
                .and(&base_rho)
                .and(void_fraction)
                .for_each(|c, &rho, &beta| {
                    if beta > 0.0 {
                        *c = wood_sound_speed(beta, *c, rho, props.c_gas, props.rho_gas);
                    }
                });
            return true;
        }
    }
    false
}

/// Apply residual-gas shielding **in place** to a heterogeneous medium so the
/// PSTD operators built from it are physically shielded.
///
/// This is the genuine PSTD time-loop hook: the forward solver constructs its
/// pressure-update sound-speed term and its fractional-Laplacian absorption
/// operators (`τ`, `η`; Treeby & Cox 2010) directly from the medium's per-voxel
/// `sound_speed` and `alpha_coefficient` arrays. Mutating those arrays — then
/// rebuilding (or refreshing) the solver for the next pulse — makes the next
/// pulse propagate through the shielded medium with no further wiring.
///
/// For each voxel with gas void fraction `β > 0`:
/// * sound speed ← Wood (1930) mixture speed (impedance mismatch + slowdown);
/// * absorption prefactor ← `α₀ + Δα₀`, where `Δα₀` is the Commander–Prosperetti
///   (1989) excess attenuation `[Np/m]` at the drive frequency, converted to the
///   medium's power-law prefactor units `dB/(MHz^y·cm)` via
///   [`np_m_to_power_law_db_cm`] so the operator reproduces that attenuation at
///   `f` (and scales as `f^y` off-frequency).
///
/// `alpha_power` is the medium's power-law exponent `y` used for the conversion.
/// Returns `true` if the medium exposed mutable arrays and shielding was applied;
/// `false` (no-op) for homogeneous media without mutable-array access.
///
/// The absorption array stores the power-law prefactor (k-Wave `α₀`,
/// `dB/(MHz^y·cm)`), not `Np/m` — hence the conversion. Sound speed is in SI and
/// written directly.
///
/// # Attenuation accuracy and double-counting
/// Both the sound-speed (Wood, quasi-static) and the absorption (CP folded into a
/// power-law prefactor) written here are **narrowband**: the absorption is exact
/// only at the drive frequency and extrapolates as `f^y` off it. This call is a
/// self-contained option for CW / single-band drive.
///
/// For genuinely broadband content (subharmonic, ultraharmonics, inertial
/// broadband), do **not** use this function; install the frequency-exact spectral
/// operator
/// [`PSTDSolver::set_residual_gas_absorption`](crate::pstd::PSTDSolver::set_residual_gas_absorption)
/// on the *unmodified host medium*. That operator carries both the true `α_cp(ω)`
/// loss and the full dispersive phase velocity `c_p(ω)` (whose `ω→0` limit is the
/// Wood speed) at every frequency — exact, with no approximation.
///
/// The two are **mutually exclusive**: the medium modifications here and the
/// broadband operator both account for the same cloud, so applying both
/// double-counts loss *and* speed change. Use this narrowband call **or** the
/// broadband operator, never both.
pub fn apply_residual_gas_shielding_in_place(
    medium: &mut dyn Medium,
    void_fraction: ArrayView3<'_, f64>,
    alpha_power: f64,
    props: &BubblyMediumProps,
) -> bool {
    let dim = void_fraction.dim();
    // Snapshot the base liquid sound speed and density BEFORE mutation: the
    // Commander–Prosperetti attenuation depends on the local liquid sound speed,
    // which the Wood step overwrites.
    let base_c = medium.sound_speed_array().to_owned();
    let base_rho = medium.density_array().to_owned();
    if base_c.dim() != dim || base_rho.dim() != dim {
        return false;
    }

    // Sound speed: Wood collapse (impedance / refraction).
    let mut applied = apply_residual_gas_wood_in_place(medium, void_fraction, props);

    // Absorption: add the Commander–Prosperetti excess (converted to the medium's
    // power-law prefactor units) using the base liquid sound speed snapshot.
    if let Some(mut a_arr) = medium.absorption_array_mut() {
        if a_arr.dim() == dim {
            ndarray::Zip::from(&mut a_arr)
                .and(&base_c)
                .and(&base_rho)
                .and(void_fraction)
                .for_each(|a, &c0, &rho, &beta| {
                    if beta > 0.0 {
                        let alpha_np_m = commander_prosperetti_attenuation(
                            props.frequency,
                            beta,
                            props.bubble_radius,
                            c0,
                            rho,
                            props.mu_liquid,
                            props.p0,
                            props.polytropic,
                        );
                        *a += np_m_to_power_law_db_cm(alpha_np_m, props.frequency, alpha_power);
                    }
                });
            applied = true;
        }
    }

    applied
}

/// Two-way amplitude transmission through a 1-D path under an absorption profile
/// `α(x)` [Np/m] with voxel spacing `dx`: `T = exp(−2 Σ α dx)`. A diagnostic for
/// how strongly a residual-gas band shields the focus on the return path.
#[must_use]
pub fn two_way_transmission(absorption_1d: &[f64], dx_m: f64) -> f64 {
    let integral: f64 = absorption_1d.iter().map(|&a| a.max(0.0) * dx_m).sum();
    (-2.0 * integral).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn base(nx: usize) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let c = Array3::from_elem((nx, 1, 1), 1481.0); // water
        let a = Array3::from_elem((nx, 1, 1), 2.5); // base absorption [Np/m]
        let rho = Array3::from_elem((nx, 1, 1), 998.0);
        (c, a, rho)
    }

    #[test]
    fn no_gas_leaves_medium_unchanged() {
        let (c, a, rho) = base(8);
        let beta = Array3::zeros((8, 1, 1));
        let props = BubblyMediumProps::air_water(2e-6, 1e6);
        let out = apply_residual_gas_shielding(c.view(), a.view(), rho.view(), beta.view(), &props);
        assert!(out.sound_speed.iter().all(|&v| (v - 1481.0).abs() < 1e-6));
        assert!(out.absorption.iter().all(|&v| (v - 2.5).abs() < 1e-9));
    }

    #[test]
    fn residual_gas_band_collapses_c_and_raises_absorption() {
        // Gas band in the middle voxels.
        let nx = 11;
        let (c, a, rho) = base(nx);
        let mut beta = Array3::zeros((nx, 1, 1));
        for i in 4..7 {
            beta[[i, 0, 0]] = 1e-4;
        }
        let props = BubblyMediumProps::air_water(2e-6, 1e6);
        let out = apply_residual_gas_shielding(c.view(), a.view(), rho.view(), beta.view(), &props);
        // Inside the band: sound speed collapses, absorption rises.
        assert!(
            out.sound_speed[[5, 0, 0]] < 1000.0,
            "Wood c-collapse in gas band"
        );
        assert!(
            out.absorption[[5, 0, 0]] > 2.5,
            "excess attenuation in gas band"
        );
        // Outside the band: unchanged.
        assert!((out.sound_speed[[0, 0, 0]] - 1481.0).abs() < 1e-6);
        assert!((out.absorption[[0, 0, 0]] - 2.5).abs() < 1e-9);
    }

    #[test]
    fn in_place_shielding_modifies_heterogeneous_medium_pstd_inputs() {
        use kwavers_grid::Grid;
        use kwavers_domain::medium::heterogeneous::tissue::{
            AbsorptionTissueType, HeterogeneousTissueMedium,
        };
        use kwavers_domain::medium::{ArrayAccess, CoreMedium};
        use kwavers_physics::acoustics::bubble_dynamics::commander_prosperetti_attenuation;
        use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;

        let nx = 11usize;
        let grid = Grid::new(nx, 1, 1, 0.5e-3, 0.5e-3, 0.5e-3).expect("valid grid");
        let mut medium = HeterogeneousTissueMedium::new(grid, AbsorptionTissueType::Water);

        // Snapshots BEFORE shielding — exactly the per-voxel inputs the PSTD
        // pressure update and absorption operators are built from.
        let c_before = medium.sound_speed(5, 0, 0);
        let a_before = medium.absorption_array()[[5, 0, 0]];
        let rho5 = medium.density_array()[[5, 0, 0]];
        let c_edge_before = medium.sound_speed(0, 0, 0);

        let mut beta = Array3::zeros((nx, 1, 1));
        for i in 4..7 {
            beta[[i, 0, 0]] = 1e-4;
        }
        let props = BubblyMediumProps::air_water(2e-6, 0.5e6);
        let y = 2.0; // water power-law exponent used for the prefactor conversion

        let applied = apply_residual_gas_shielding_in_place(&mut medium, beta.view(), y, &props);
        assert!(
            applied,
            "heterogeneous tissue medium exposes mutable arrays"
        );

        // Sound speed collapses inside the gas band (Wood), unchanged outside.
        let c_after = medium.sound_speed(5, 0, 0);
        assert!(
            c_after < c_before && c_after < 1000.0,
            "Wood c-collapse: {c_before} -> {c_after}"
        );
        assert!(
            (medium.sound_speed(0, 0, 0) - c_edge_before).abs() < 1e-9,
            "voxel outside the gas band is untouched"
        );

        // Absorption prefactor raised inside the band.
        let a_after = medium.absorption_array()[[5, 0, 0]];
        assert!(
            a_after > a_before,
            "prefactor raised: {a_before} -> {a_after}"
        );

        // End-to-end operator proof: the SI absorption the fractional-Laplacian
        // operators (τ, η) are built from must rise by EXACTLY the injected
        // Commander–Prosperetti attenuation at the drive frequency.
        let omega_pow = (2.0 * std::f64::consts::PI * props.frequency).powf(y);
        let alpha_si_before = power_law_db_cm_to_np_omega_m(a_before, y) * omega_pow;
        let alpha_si_after = power_law_db_cm_to_np_omega_m(a_after, y) * omega_pow;
        let alpha_cp_expected = commander_prosperetti_attenuation(
            props.frequency,
            1e-4,
            props.bubble_radius,
            c_before,
            rho5,
            props.mu_liquid,
            props.p0,
            props.polytropic,
        );
        let delta = alpha_si_after - alpha_si_before;
        assert!(alpha_cp_expected > 0.0, "CP attenuation is positive");
        assert!(
            (delta - alpha_cp_expected).abs() <= 1e-6 * alpha_cp_expected.max(1.0),
            "SI absorption rises by exactly the CP excess: Δ={delta:.6} Np/m, expected={alpha_cp_expected:.6} Np/m"
        );
    }

    #[test]
    fn wood_only_changes_sound_speed_but_not_absorption() {
        use kwavers_grid::Grid;
        use kwavers_domain::medium::heterogeneous::tissue::{
            AbsorptionTissueType, HeterogeneousTissueMedium,
        };
        use kwavers_domain::medium::{ArrayAccess, CoreMedium};

        let nx = 9usize;
        let grid = Grid::new(nx, 1, 1, 0.5e-3, 0.5e-3, 0.5e-3).expect("valid grid");
        let mut medium = HeterogeneousTissueMedium::new(grid, AbsorptionTissueType::Water);
        let a_before = medium.absorption_array()[[4, 0, 0]];
        let c_before = medium.sound_speed(4, 0, 0);

        let mut beta = Array3::zeros((nx, 1, 1));
        beta[[4, 0, 0]] = 1e-4;
        let props = BubblyMediumProps::air_water(2e-6, 0.5e6);

        assert!(apply_residual_gas_wood_in_place(
            &mut medium,
            beta.view(),
            &props
        ));
        // Sound speed collapses; absorption is left for the broadband operator.
        assert!(medium.sound_speed(4, 0, 0) < c_before && medium.sound_speed(4, 0, 0) < 1000.0);
        assert!((medium.absorption_array()[[4, 0, 0]] - a_before).abs() < 1e-12);
    }

    #[test]
    fn residual_gas_reduces_two_way_transmission() {
        // The focus on the return path is more strongly shielded with gas.
        let nx = 21;
        let (c, a, rho) = base(nx);
        let mut beta = Array3::zeros((nx, 1, 1));
        for i in 8..13 {
            beta[[i, 0, 0]] = 1e-4;
        }
        let props = BubblyMediumProps::air_water(2e-6, 1e6);
        let out = apply_residual_gas_shielding(c.view(), a.view(), rho.view(), beta.view(), &props);
        let dx = 0.5e-3;
        let alpha_gas: Vec<f64> = out.absorption.iter().copied().collect();
        let alpha_base: Vec<f64> = a.iter().copied().collect();
        let t_gas = two_way_transmission(&alpha_gas, dx);
        let t_base = two_way_transmission(&alpha_base, dx);
        assert!(
            t_gas < t_base,
            "residual gas must reduce transmission: t_gas={t_gas:.4}, t_base={t_base:.4}"
        );
    }
}
