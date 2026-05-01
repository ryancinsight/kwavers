//! Dormand-Prince RK45 adaptive integrator for radical ODE systems.
//!
//! ## Algorithm (Dormand & Prince 1980, Table 1 -- DOPRI5)
//!
//! The concentration vector **N** = [N1, N2, ..., Ns] satisfies:
//!
//! ```text
//! dNi/dt = fi(N, T, pH) = sum_j nu_ij * r_j(N, T, pH)
//!
//! r_j = k_j(T) * product_l [N_l]^n_lj       (mass-action kinetics)
//! k_j(T) = A_j * exp(-Ea_j / RT)            (Arrhenius)
//! ```
//!
//! The RK45 embedded pair produces a 5th-order solution and a 4th-order
//! companion solution for error estimation:
//!
//! ```text
//! y5 = y_n + h * sum_i b_i * k_i
//! err = || h * sum_i e_i * k_i / (atol + rtol * |y5|) ||_inf
//! h_new = h * clamp(0.9 * err^(-1/5), 0.2, 5.0)
//! ```
//!
//! ## References
//!
//! - Dormand JR, Prince PJ (1980). "A family of embedded Runge-Kutta formulae."
//!   *J Comput Appl Math* **6**(1), 19-26. DOI: 10.1016/0771-050X(80)90013-3
//! - Riesz P, Leighton T (2012). "Free radical generation by ultrasound."
//!   *Environ Health Perspect* **64**, 233-252.
//! - Christman CL (1987). "Sonoluminescence and sonochemistry: Implications from
//!   collision theory." *Ultrasonics* **25**(1), 31-37.

mod rhs;
mod tableau;
mod types;

#[cfg(test)]
mod tests;

use self::rhs::{collect_species, eval_rhs, eval_rhs_at};
use self::tableau::*;
pub use self::types::{IntegrationStats, IntegratorError};
use crate::physics::chemistry::ros_plasma::radical_kinetics::RadicalKinetics;
use crate::physics::chemistry::ros_plasma::ros_species::ROSSpecies;
use std::collections::HashMap;

/// Dormand-Prince RK45 adaptive integrator for radical ODE systems.
///
/// ## Usage
///
/// ```rust,no_run
/// use std::collections::HashMap;
/// use kwavers::physics::chemistry::integrator::RadicalIntegrator;
/// use kwavers::physics::chemistry::ros_plasma::{RadicalKinetics, ROSSpecies};
///
/// let kinetics = RadicalKinetics::new(7.0, 310.0);
/// let integrator = RadicalIntegrator::new(kinetics);
///
/// let mut concs = HashMap::new();
/// concs.insert(ROSSpecies::HydroxylRadical, 1e-6);
///
/// let (final_concs, stats) = integrator.integrate(&concs, 0.0, 1e-6, 310.0, 7.0).unwrap();
/// println!("Accepted steps: {}, rejected: {}", stats.steps_accepted, stats.steps_rejected);
/// ```
#[derive(Debug, Clone)]
pub struct RadicalIntegrator {
    /// Reaction kinetics providing the ODE right-hand side.
    pub kinetics: RadicalKinetics,
    /// Relative tolerance.
    pub rtol: f64,
    /// Absolute tolerance [mol/L].
    pub atol: f64,
    /// Minimum allowable step size [s].
    pub h_min: f64,
    /// Maximum allowable step size [s].
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

    /// Create an integrator with explicit relative and absolute tolerances.
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
    /// Species not present in `concentrations` are treated as zero. Missing
    /// species are added to the output map if their rate changes during
    /// integration. Accepted states are clamped to non-negative concentrations.
    ///
    /// # Errors
    ///
    /// Returns [`IntegratorError::StepSizeTooSmall`] if the adaptive controller
    /// cannot maintain error tolerance with `h >= h_min`.
    pub fn integrate(
        &self,
        concentrations: &HashMap<ROSSpecies, f64>,
        t_start: f64,
        t_end: f64,
        temperature_k: f64,
        ph: f64,
    ) -> Result<(HashMap<ROSSpecies, f64>, IntegrationStats), IntegratorError> {
        let mut kinetics = self.kinetics.clone();
        kinetics.temperature = temperature_k;
        kinetics.ph = ph;

        let species_list = collect_species(&kinetics, concentrations);
        let n = species_list.len();

        let mut y: Vec<f64> = species_list
            .iter()
            .map(|s| concentrations.get(s).copied().unwrap_or(0.0).max(0.0))
            .collect();

        let mut t = t_start;
        let mut h = (t_end - t_start).min(self.h_max).max(self.h_min);

        let mut steps_accepted = 0usize;
        let mut steps_rejected = 0usize;

        let mut k1 = vec![0.0_f64; n];
        let mut k2 = vec![0.0_f64; n];
        let mut k3 = vec![0.0_f64; n];
        let mut k4 = vec![0.0_f64; n];
        let mut k5 = vec![0.0_f64; n];
        let mut k6 = vec![0.0_f64; n];
        let mut k7 = vec![0.0_f64; n];
        let mut ytmp = vec![0.0_f64; n];
        let mut y5 = vec![0.0_f64; n];

        while t < t_end {
            h = h.min(t_end - t);

            eval_rhs(&kinetics, &species_list, &y, &mut k1);

            for i in 0..n {
                ytmp[i] = (y[i] + h * (A21 * k1[i])).max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, t + C2 * h, &mut k2);

            for i in 0..n {
                ytmp[i] = (y[i] + h * (A31 * k1[i] + A32 * k2[i])).max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, t + C3 * h, &mut k3);

            for i in 0..n {
                ytmp[i] = (y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i])).max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, t + C4 * h, &mut k4);

            for i in 0..n {
                ytmp[i] =
                    (y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i])).max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, t + C5 * h, &mut k5);

            for i in 0..n {
                ytmp[i] = (y[i]
                    + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]))
                    .max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, t + h, &mut k6);

            for i in 0..n {
                y5[i] = (y[i]
                    + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]))
                    .max(0.0);
            }

            eval_rhs(&kinetics, &species_list, &y5, &mut k7);

            let mut err_max = 0.0_f64;
            for i in 0..n {
                let sc = self.atol + self.rtol * y5[i].abs();
                let e = h
                    * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
                err_max = err_max.max((e / sc).abs());
            }

            if err_max <= 1.0 || h <= self.h_min {
                t += h;
                y.copy_from_slice(&y5);
                steps_accepted += 1;
                k1.copy_from_slice(&k7);
            } else {
                steps_rejected += 1;
            }

            let factor = if err_max == 0.0 {
                5.0
            } else {
                0.9 * err_max.powf(-0.2)
            };
            h = (h * factor.clamp(0.2, 5.0)).clamp(self.h_min, self.h_max);

            if h < self.h_min && t < t_end {
                return Err(IntegratorError::StepSizeTooSmall { h, t });
            }
        }

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
