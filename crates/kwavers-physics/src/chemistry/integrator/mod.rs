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
use self::tableau::{
    A21, A31, A32, A41, A42, A43, A51, A52, A53, A54, A61, A62, A63, A64, A65, B1, B3, B4, B5, B6,
    C2, C3, C4, C5, E1, E3, E4, E5, E6, E7,
};
pub use self::types::{IntegrationStats, IntegratorError};
use crate::chemistry::ros_plasma::radical_kinetics::RadicalKinetics;
use crate::chemistry::ros_plasma::ros_species::ROSSpecies;
use std::collections::HashMap;

/// Dormand-Prince RK45 adaptive integrator for radical ODE systems.
///
/// ## Usage
///
/// ```rust,no_run
/// use std::collections::HashMap;
/// use kwavers_physics::chemistry::integrator::RadicalIntegrator;
/// use kwavers_physics::chemistry::ros_plasma::{RadicalKinetics, ROSSpecies};
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
    /// Minimum allowable step size (s).
    pub h_min: f64,
    /// Maximum allowable step size (s).
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
                ytmp[i] = h.mul_add(A21 * k1[i], y[i]).max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, C2.mul_add(h, t), &mut k2);

            for i in 0..n {
                ytmp[i] = h.mul_add(A31.mul_add(k1[i], A32 * k2[i]), y[i]).max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, C3.mul_add(h, t), &mut k3);

            for i in 0..n {
                ytmp[i] = h
                    .mul_add(A43.mul_add(k3[i], A41.mul_add(k1[i], A42 * k2[i])), y[i])
                    .max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, C4.mul_add(h, t), &mut k4);

            for i in 0..n {
                ytmp[i] = h
                    .mul_add(
                        A54.mul_add(k4[i], A53.mul_add(k3[i], A51.mul_add(k1[i], A52 * k2[i]))),
                        y[i],
                    )
                    .max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, C5.mul_add(h, t), &mut k5);

            for i in 0..n {
                ytmp[i] = h
                    .mul_add(
                        A65.mul_add(
                            k5[i],
                            A64.mul_add(k4[i], A63.mul_add(k3[i], A61.mul_add(k1[i], A62 * k2[i]))),
                        ),
                        y[i],
                    )
                    .max(0.0);
            }
            eval_rhs_at(&kinetics, &species_list, &ytmp, t + h, &mut k6);

            for i in 0..n {
                y5[i] = h
                    .mul_add(
                        B6.mul_add(
                            k6[i],
                            B5.mul_add(k5[i], B4.mul_add(k4[i], B1.mul_add(k1[i], B3 * k3[i]))),
                        ),
                        y[i],
                    )
                    .max(0.0);
            }

            eval_rhs(&kinetics, &species_list, &y5, &mut k7);

            let mut err_max = 0.0_f64;
            for i in 0..n {
                let sc = self.rtol.mul_add(y5[i].abs(), self.atol);
                let e = h * E7.mul_add(
                    k7[i],
                    E6.mul_add(
                        k6[i],
                        E5.mul_add(k5[i], E4.mul_add(k4[i], E1.mul_add(k1[i], E3 * k3[i]))),
                    ),
                );
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
