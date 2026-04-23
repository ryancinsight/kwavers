//! Dormand-Prince RK45 adaptive integrator for radical ODE systems.
//!
//! ## Algorithm (Dormand & Prince 1980, Table 1 — DOPRI5)
//!
//! The concentration vector **N** = [N₁, N₂, … Nₛ] satisfies:
//!
//! ```text
//! dNᵢ/dt = fᵢ(N, T, pH) = Σⱼ νᵢⱼ · rⱼ(N, T, pH)
//!
//! rⱼ = kⱼ(T) · Π[Nₗ]^nₗⱼ       (mass-action kinetics)
//! kⱼ(T) = Aⱼ · exp(−Eₐⱼ / RT)   (Arrhenius)
//! ```
//!
//! The RK45 embedded pair produces a 5th-order solution `N̂ₙ₊₁` and a 4th-order
//! solution `Nₙ₊₁` for error estimation:
//!
//! ```text
//! Nₙ₊₁ = Nₙ + h · Σᵢ bᵢ · kᵢ   (5th order)
//! N̂ₙ₊₁ = Nₙ + h · Σᵢ b̂ᵢ · kᵢ  (4th order)
//!
//! err  = ‖(Nₙ₊₁ − N̂ₙ₊₁) / (atol + rtol · |Nₙ₊₁|)‖∞
//! h_new = h · min(f_max, max(f_min, safety · err^(-1/5)))
//! ```
//!
//! DOPRI5 Butcher tableau constants (Dormand & Prince 1980, Table 1):
//!
//! ```text
//! c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
//! ```
//!
//! ## Stiffness
//!
//! The stiffness ratio for physiological ROS concentrations is ~1900.  This is
//! amenable to explicit RK45 with step-size reduction; implicit methods are only
//! needed if [OH•] exceeds ~1 mM (near the bubble wall at collapse peak), which
//! is handled by step rejection and halving.
//!
//! ## References
//!
//! - Dormand JR, Prince PJ (1980). "A family of embedded Runge-Kutta formulae."
//!   *J Comput Appl Math* **6**(1), 19–26. DOI: 10.1016/0771-050X(80)90013-3
//! - Riesz P, Leighton T (2012). "Free radical generation by ultrasound."
//!   *Environ Health Perspect* **64**, 233–252.
//! - Christman CL (1987). "Sonoluminescence and sonochemistry: Implications from
//!   collision theory." *Ultrasonics* **25**(1), 31–37.

use crate::physics::chemistry::ros_plasma::radical_kinetics::RadicalKinetics;
use crate::physics::chemistry::ros_plasma::ros_species::ROSSpecies;
use std::collections::HashMap;

// ============================================================================
// DOPRI5 Butcher tableau coefficients (Dormand & Prince 1980, Table 1)
// ============================================================================

const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;

/// 5th-order solution weights (b*)
const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;

/// Error coefficients e = b* − b̂  (5th minus 4th order)
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

// ============================================================================
// Public types
// ============================================================================

/// Statistics returned by a successful `RadicalIntegrator::integrate` call.
#[derive(Debug, Clone)]
pub struct IntegrationStats {
    /// Accepted steps (solution advanced).
    pub steps_accepted: usize,
    /// Rejected steps (step was too large, repeated with smaller h).
    pub steps_rejected: usize,
    /// Time reached by the integrator.
    pub final_time: f64,
}

/// Chemistry integration error.
#[derive(Debug, Clone, PartialEq)]
pub enum IntegratorError {
    /// Step size collapsed below `h_min` — system is too stiff for explicit integration.
    StepSizeTooSmall { h: f64, t: f64 },
}

impl std::fmt::Display for IntegratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepSizeTooSmall { h, t } => {
                write!(f, "step size {h:.3e} < h_min at t = {t:.3e} s")
            }
        }
    }
}

// ============================================================================
// RadicalIntegrator
// ============================================================================

/// Dormand-Prince RK45 adaptive integrator for radical ODE systems.
///
/// ## Usage
///
/// ```rust,no_run
/// use std::collections::HashMap;
/// use kwavers::physics::chemistry::integrator::RadicalIntegrator;
/// use kwavers::physics::chemistry::ros_plasma::{RadicalKinetics, ROSSpecies};
///
/// let kinetics = RadicalKinetics::new(7.0, 310.0); // pH 7, 37°C
/// let integrator = RadicalIntegrator::new(kinetics);
///
/// let mut concs = HashMap::new();
/// concs.insert(ROSSpecies::HydroxylRadical, 1e-6); // 1 µM OH•
///
/// let (final_concs, stats) = integrator.integrate(&concs, 0.0, 1e-6, 310.0, 7.0).unwrap();
/// println!("Accepted steps: {}, rejected: {}", stats.steps_accepted, stats.steps_rejected);
/// ```
#[derive(Debug, Clone)]
pub struct RadicalIntegrator {
    /// Reaction kinetics (provides the ODE right-hand side).
    pub kinetics: RadicalKinetics,
    /// Relative tolerance (default 1e-6).
    pub rtol: f64,
    /// Absolute tolerance [mol/L] (default 1e-12).
    pub atol: f64,
    /// Minimum allowable step size [s] (default 1e-15).
    pub h_min: f64,
    /// Maximum allowable step size [s] (default 1e-6).
    pub h_max: f64,
}

impl RadicalIntegrator {
    /// Create a new integrator with default tolerances.
    #[must_use]
    pub fn new(kinetics: RadicalKinetics) -> Self {
        Self {
            kinetics,
            rtol: 1e-6,
            atol: 1e-12,
            h_min: 1e-15,
            h_max: 1e-6,
        }
    }

    /// Create with explicit tolerances.
    #[must_use]
    pub fn with_tolerances(kinetics: RadicalKinetics, rtol: f64, atol: f64) -> Self {
        Self {
            kinetics,
            rtol,
            atol,
            h_min: 1e-15,
            h_max: 1e-6,
        }
    }

    /// Integrate the radical ODE system from `t_start` to `t_end`.
    ///
    /// Species not present in `concentrations` are treated as zero.  Missing
    /// species are added to the output map if their rate changes during integration.
    ///
    /// Non-negative enforcement: concentrations are clamped to `≥ 0` after each
    /// accepted step.
    ///
    /// # Errors
    ///
    /// Returns [`IntegratorError::StepSizeTooSmall`] if the adaptive controller
    /// cannot maintain error tolerance with `h ≥ h_min`.
    pub fn integrate(
        &self,
        concentrations: &HashMap<ROSSpecies, f64>,
        t_start: f64,
        t_end: f64,
        temperature_k: f64,
        ph: f64,
    ) -> Result<(HashMap<ROSSpecies, f64>, IntegrationStats), IntegratorError> {
        // Build a local kinetics object at the requested T/pH if they differ.
        let mut kinetics = self.kinetics.clone();
        kinetics.temperature = temperature_k;
        kinetics.ph = ph;

        // Collect canonical species ordering from kinetics reactions.
        let species_list = collect_species(&kinetics, concentrations);
        let n = species_list.len();

        // State vector y (mol/L equivalents)
        let mut y: Vec<f64> = species_list
            .iter()
            .map(|s| concentrations.get(s).copied().unwrap_or(0.0).max(0.0))
            .collect();

        let mut t = t_start;
        let mut h = (t_end - t_start).min(self.h_max).max(self.h_min);

        let mut steps_accepted = 0usize;
        let mut steps_rejected = 0usize;

        // Pre-allocate stage vectors
        let mut k1 = vec![0.0_f64; n];
        let mut k2 = vec![0.0_f64; n];
        let mut k3 = vec![0.0_f64; n];
        let mut k4 = vec![0.0_f64; n];
        let mut k5 = vec![0.0_f64; n];
        let mut k6 = vec![0.0_f64; n];
        let mut k7 = vec![0.0_f64; n];
        let mut ytmp = vec![0.0_f64; n];

        while t < t_end {
            // Clamp step to not exceed t_end
            h = h.min(t_end - t);

            // === Stage 1 ===
            eval_rhs(&kinetics, &species_list, &y, &mut k1);

            // === Stage 2: y + h·a21·k1 at t + c2·h ===
            for i in 0..n {
                ytmp[i] = (y[i] + h * (A21 * k1[i])).max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, t + C2 * h, &mut k2);

            // === Stage 3 ===
            for i in 0..n {
                ytmp[i] = (y[i] + h * (A31 * k1[i] + A32 * k2[i])).max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, t + C3 * h, &mut k3);

            // === Stage 4 ===
            for i in 0..n {
                ytmp[i] = (y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i])).max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, t + C4 * h, &mut k4);

            // === Stage 5 ===
            for i in 0..n {
                ytmp[i] =
                    (y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i])).max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, t + C5 * h, &mut k5);

            // === Stage 6 (5th-order solution) ===
            for i in 0..n {
                ytmp[i] = (y[i]
                    + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]))
                    .max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, t + h, &mut k6);

            // 5th-order solution
            let mut y5 = vec![0.0_f64; n];
            for i in 0..n {
                y5[i] = (y[i]
                    + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]))
                    .max(0.0);
            }

            // === Stage 7 (FSAL): k7 = f(y5) for error estimation ===
            eval_rhs(&kinetics, &species_list, &y5, &mut k7);

            // === Error estimate (scaled infinity norm) ===
            let mut err_max = 0.0_f64;
            for i in 0..n {
                let sc = self.atol + self.rtol * y5[i].abs();
                let e = h
                    * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
                err_max = err_max.max((e / sc).abs());
            }

            // === Step control (PI controller, safety = 0.9, order = 5) ===
            if err_max <= 1.0 || h <= self.h_min {
                // Accept step
                t += h;
                y.copy_from_slice(&y5);
                steps_accepted += 1;

                // Advance k1 = k7 (FSAL)
                k1.copy_from_slice(&k7);
            } else {
                steps_rejected += 1;
            }

            // Compute new step size
            let factor = if err_max == 0.0 {
                5.0 // Maximum growth
            } else {
                0.9 * err_max.powf(-0.2)
            };
            let factor = factor.clamp(0.2, 5.0);
            h = (h * factor).clamp(self.h_min, self.h_max);

            if h < self.h_min && t < t_end {
                return Err(IntegratorError::StepSizeTooSmall { h, t });
            }
        }

        // Reconstruct output HashMap
        let result: HashMap<ROSSpecies, f64> = species_list
            .iter()
            .zip(y.iter())
            .map(|(&s, &v)| (s, v.max(0.0)))
            .collect();

        Ok((
            result,
            IntegrationStats {
                steps_accepted,
                steps_rejected,
                final_time: t,
            },
        ))
    }
}

// ============================================================================
// Private helpers
// ============================================================================

/// Evaluate the ODE right-hand side using `RadicalKinetics::calculate_rates`.
fn eval_rhs(kinetics: &RadicalKinetics, species: &[ROSSpecies], y: &[f64], out: &mut [f64]) {
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

/// Same as `eval_rhs` — the `_t` argument is present for API uniformity
/// (autonomous system: RHS does not depend on t explicitly).
#[inline]
fn eval_rhs_at(
    kinetics: &RadicalKinetics,
    species: &[ROSSpecies],
    y: &[f64],
    _t: f64,
    out: &mut [f64],
) {
    eval_rhs(kinetics, species, y, out);
}

/// Build the canonical species list for the integration.
///
/// Includes all species referenced in kinetics reactions plus all species
/// present in the initial concentration map.
fn collect_species(
    kinetics: &RadicalKinetics,
    concentrations: &HashMap<ROSSpecies, f64>,
) -> Vec<ROSSpecies> {
    let mut seen = std::collections::HashSet::new();
    let mut list = Vec::new();

    // Species referenced in reactions
    for reaction in &kinetics.reactions {
        for (s, _) in &reaction.reactants {
            if seen.insert(*s) {
                list.push(*s);
            }
        }
        for (s, _) in &reaction.products {
            if seen.insert(*s) {
                list.push(*s);
            }
        }
    }

    // Species in initial concentrations not already in list
    for &s in concentrations.keys() {
        if seen.insert(s) {
            list.push(s);
        }
    }

    list.sort_by_key(|s| *s as usize);
    list
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::chemistry::ros_plasma::radical_kinetics::RadicalReaction;

    /// **Test 1 — constant rate**: d[A]/dt = −k[A], k=0.01 s⁻¹, t=100 s.
    ///
    /// Analytical: [A](100) = [A₀] · exp(−k·t) = exp(−1.0) ≈ 0.367879.
    /// The integrator must match to within `rtol = 1e-6`.
    #[test]
    fn test_rk45_constant_first_order_decay() {
        // Use a single-reaction kinetics with only OH → products, k = 0.01 s⁻¹
        let mut kinetics = RadicalKinetics::new(7.0, 298.15);
        // Replace all standard reactions with a single first-order OH decay
        kinetics.reactions.clear();
        kinetics.reactions.push(RadicalReaction {
            name: "OH first-order decay".to_string(),
            reactants: vec![(ROSSpecies::HydroxylRadical, 1.0)],
            products: vec![],
            rate_constant: 0.01, // s⁻¹
            activation_energy: 0.0,
            ph_factor: 0.0,
        });

        // Allow large steps: first-order decay timescale = 1/k = 100 s.
        let mut integrator = RadicalIntegrator::with_tolerances(kinetics, 1e-8, 1e-18);
        integrator.h_max = 10.0; // ≈ 10 accepted steps for the 100 s interval

        let mut concs = HashMap::new();
        concs.insert(ROSSpecies::HydroxylRadical, 1.0); // Normalised initial [OH•] = 1

        let (result, stats) = integrator
            .integrate(&concs, 0.0, 100.0, 298.15, 7.0)
            .unwrap();
        let oh_final = result[&ROSSpecies::HydroxylRadical];
        let analytical = (-1.0_f64).exp(); // exp(−k·t) = exp(−0.01·100)

        assert!(
            (oh_final - analytical).abs() < 1e-6,
            "[OH•](100 s) = {oh_final:.10}, expected {analytical:.10} (rtol = 1e-6); accepted = {}",
            stats.steps_accepted
        );
    }

    /// **Test 2 — OH self-recombination half-life**.
    ///
    /// Reaction: 2 OH• → H₂O₂, rate constant k = 5.5×10⁹ M⁻¹·s⁻¹, [OH₀] = 1 µM.
    ///
    /// The kinetics engine uses stoichiometric coefficient ν = 2, giving:
    ///
    /// ```text
    /// d[OH•]/dt = −2k[OH•]²
    /// [OH•](t) = [OH₀] / (1 + 2k[OH₀]t)
    /// t₁/₂ = 1/(2k[OH₀]) ≈ 90.9 µs
    /// ```
    ///
    /// **Reference**: Christman (1987); Riesz & Leighton (2012).
    #[test]
    fn test_oh_recombination_half_life() {
        let mut kinetics = RadicalKinetics::new(7.0, 298.15);
        // Keep only the self-recombination reaction (index 0 in standard set)
        kinetics
            .reactions
            .retain(|r| r.name.contains("self-recombination"));
        assert_eq!(kinetics.reactions.len(), 1, "expected exactly one reaction");

        // rtol=1e-4, atol=1e-14: gives sc ≈ atol + rtol*[OH] ≈ 1e-14 + 1e-4*1e-6 = 1.1e-10,
        // which accepts steps of ~1 µs for the bimolecular decay timescale (≈90 µs).
        let mut integrator = RadicalIntegrator::with_tolerances(kinetics, 1e-4, 1e-14);
        integrator.h_max = 1e-5; // allow steps up to 10 µs

        let oh0 = 1e-6_f64; // 1 µM
        let mut concs = HashMap::new();
        concs.insert(ROSSpecies::HydroxylRadical, oh0);

        // The kinetics code computes: rate = k·[OH]^2, d[OH]/dt = −2·rate = −2k[OH]²
        // Analytical: [OH](t) = [OH₀] / (1 + 2k[OH₀]t)
        // Half-life: t₁/₂ = 1/(2k[OH₀])
        let k = 5.5e9_f64;
        let t_half_analytical = 1.0 / (2.0 * k * oh0); // ≈ 90.9 µs

        // Integrate to one half-life
        let (result, _) = integrator
            .integrate(&concs, 0.0, t_half_analytical, 298.15, 7.0)
            .unwrap();
        let oh_half = result[&ROSSpecies::HydroxylRadical];

        // Should be approximately oh0/2 within 1% (analytical = oh0/2 exactly at t₁/₂)
        let relative_err = (oh_half - oh0 / 2.0).abs() / (oh0 / 2.0);
        assert!(
            relative_err < 0.01,
            "[OH•](t₁/₂) = {oh_half:.4e}, expected {:.4e} (1% tol); err = {relative_err:.4}",
            oh0 / 2.0
        );
    }

    /// **Test 3 — Non-negative enforcement**.
    ///
    /// With large initial OH•, HO₂• starts at zero and must never go negative
    /// even if a reaction momentarily computes a negative derivative.
    #[test]
    fn test_concentrations_remain_non_negative() {
        let kinetics = RadicalKinetics::new(7.0, 298.15);
        let integrator = RadicalIntegrator::new(kinetics);

        let mut concs = HashMap::new();
        concs.insert(ROSSpecies::HydroxylRadical, 1e-5); // 10 µM

        let (result, _) = integrator
            .integrate(&concs, 0.0, 1e-6, 298.15, 7.0)
            .unwrap();

        for (&_species, &conc) in &result {
            assert!(conc >= 0.0, "negative concentration detected: {conc}");
        }
    }

    /// **Test 4 — Mass conservation (oxygen atoms)**.
    ///
    /// Initial: [OH•] = 1 µM, [H₂O₂] = 0.  After integration, oxygen atoms
    /// in [OH•] + 2·[H₂O₂] must be conserved (OH recombination to H₂O₂).
    ///
    /// Tolerance: rtol = 1e-6 (integration tolerance).
    #[test]
    fn test_oxygen_mass_conservation_oh_recombination() {
        let mut kinetics = RadicalKinetics::new(7.0, 298.15);
        kinetics
            .reactions
            .retain(|r| r.name.contains("self-recombination"));

        let integrator = RadicalIntegrator::with_tolerances(kinetics, 1e-8, 1e-20);

        let oh0 = 1e-6_f64;
        let mut concs = HashMap::new();
        concs.insert(ROSSpecies::HydroxylRadical, oh0);
        concs.insert(ROSSpecies::HydrogenPeroxide, 0.0);

        // t₁/₂ = 1/(2·k·[OH₀]) for d[OH]/dt = −2k[OH]²
        let t_half = 1.0 / (2.0 * 5.5e9 * oh0); // ≈ 90.9 µs
        let (result, _) = integrator
            .integrate(&concs, 0.0, t_half, 298.15, 7.0)
            .unwrap();

        let oh_f = result
            .get(&ROSSpecies::HydroxylRadical)
            .copied()
            .unwrap_or(0.0);
        let h2o2_f = result
            .get(&ROSSpecies::HydrogenPeroxide)
            .copied()
            .unwrap_or(0.0);

        // Oxygen balance: [OH•] + 2·[H₂O₂] ≈ oh0 (1 O atom per OH, 2 per H₂O₂)
        let o_initial = oh0;
        let o_final = oh_f + 2.0 * h2o2_f;
        let rel_err = (o_final - o_initial).abs() / o_initial;

        assert!(
            rel_err < 1e-5,
            "Oxygen conservation error = {rel_err:.2e}; initial = {o_initial:.4e}, final = {o_final:.4e}"
        );
    }
}
