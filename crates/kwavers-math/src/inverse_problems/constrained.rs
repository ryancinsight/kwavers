//! Box-constrained inversion via projected gradient descent.
//!
//! Many inverse problems (FWI, elastography) have physically meaningful bounds on
//! the model parameters — e.g. soft-tissue sound speed `c ∈ [1400, 1650]` m/s or
//! density `ρ ∈ [900, 1100]` kg/m³. Projecting each gradient step back onto the
//! feasible box keeps the iterate physical and improves conditioning:
//!
//! ```text
//! m^{k+1} = Π_[m_min, m_max]( m^k − α_k ∇J(m^k) )
//! ```
//!
//! where `Π` is the pointwise clip onto `[m_min, m_max]`. For a separable convex
//! objective the box-constrained minimiser is exactly the projection of the
//! unconstrained minimiser onto the box, and projected gradient descent converges
//! to it.

use ndarray::{Array3, Zip};

/// Pointwise box constraints `lower ≤ m(r) ≤ upper` on a model field.
#[derive(Debug, Clone, Copy)]
pub struct BoxConstraints {
    lower: f64,
    upper: f64,
}

impl BoxConstraints {
    /// Create box constraints. The arguments are ordered so the result always
    /// satisfies `lower ≤ upper` (swapped if necessary).
    #[must_use]
    pub fn new(a: f64, b: f64) -> Self {
        if a <= b {
            Self { lower: a, upper: b }
        } else {
            Self { lower: b, upper: a }
        }
    }

    /// Physiological soft-tissue sound-speed bounds `[1400, 1650]` m/s.
    #[must_use]
    pub fn sound_speed_tissue() -> Self {
        Self {
            lower: 1400.0,
            upper: 1650.0,
        }
    }

    /// Physiological soft-tissue density bounds `[900, 1100]` kg/m³.
    #[must_use]
    pub fn density_tissue() -> Self {
        Self {
            lower: 900.0,
            upper: 1100.0,
        }
    }

    /// Lower bound `m_min`.
    #[must_use]
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// Upper bound `m_max`.
    #[must_use]
    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// Project a single value onto `[lower, upper]`.
    #[must_use]
    pub fn project_value(&self, v: f64) -> f64 {
        v.clamp(self.lower, self.upper)
    }

    /// Project a model field onto the box in place — the `Π` operator.
    pub fn project(&self, model: &mut Array3<f64>) {
        let (lo, hi) = (self.lower, self.upper);
        model.iter_mut().for_each(|m| *m = m.clamp(lo, hi));
    }
}

/// Box-constrained gradient descent.
///
/// Starting from `model` (projected onto the box first), performs `iterations`
/// steps of `m ← Π( m − step · ∇J(m) )` and returns the constrained model. The
/// gradient is supplied as a closure so this works for any objective (FWI data
/// misfit, regularised elastography, etc.).
///
/// # Panics
/// Does not panic; a non-positive `step` simply leaves the projected model
/// unchanged each iteration.
#[must_use]
pub fn projected_gradient_descent<F>(
    mut model: Array3<f64>,
    constraints: BoxConstraints,
    step: f64,
    iterations: usize,
    mut gradient: F,
) -> Array3<f64>
where
    F: FnMut(&Array3<f64>) -> Array3<f64>,
{
    constraints.project(&mut model); // start feasible
    for _ in 0..iterations {
        let grad = gradient(&model);
        Zip::from(&mut model)
            .and(&grad)
            .for_each(|m, &g| *m -= step * g);
        constraints.project(&mut model);
    }
    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn new_orders_bounds() {
        let c = BoxConstraints::new(1650.0, 1400.0);
        assert_eq!((c.lower(), c.upper()), (1400.0, 1650.0));
    }

    #[test]
    fn project_clamps_out_of_range_keeps_in_range() {
        let c = BoxConstraints::sound_speed_tissue(); // [1400, 1650]
        assert_eq!(c.project_value(1200.0), 1400.0); // below → lower
        assert_eq!(c.project_value(1800.0), 1650.0); // above → upper
        assert_eq!(c.project_value(1540.0), 1540.0); // inside → unchanged

        let mut field =
            Array3::from_shape_vec((2, 1, 2), vec![1200.0, 1540.0, 1800.0, 1500.0]).unwrap();
        c.project(&mut field);
        assert_eq!(
            field.iter().cloned().collect::<Vec<_>>(),
            vec![1400.0, 1540.0, 1650.0, 1500.0]
        );
    }

    #[test]
    fn pgd_converges_to_projection_of_unconstrained_minimiser() {
        // J(m) = ½‖m − t‖²  ⇒  ∇J = m − t ; separable, so the box-constrained
        // minimiser is clip(t, [lo, hi]) element-wise.
        let c = BoxConstraints::sound_speed_tissue(); // [1400, 1650]
                                                      // target field: below / inside / above the box
        let target = Array3::from_shape_vec((3, 1, 1), vec![1000.0, 1500.0, 2000.0]).unwrap();
        let t = target.clone();
        let start = Array3::from_elem((3, 1, 1), 1540.0);

        let result = projected_gradient_descent(start, c, 0.5, 200, move |m| m - &t);

        let got: Vec<f64> = result.iter().cloned().collect();
        // clip(1000,…)=1400 ; clip(1500)=1500 ; clip(2000,…)=1650
        let expected = [1400.0, 1500.0, 1650.0];
        for (g, e) in got.iter().zip(expected) {
            assert!((g - e).abs() < 1e-6, "PGD got {g}, expected {e}");
        }
        // sanity: the returned model is feasible everywhere
        assert!(result
            .iter()
            .all(|&m| m >= c.lower() - 1e-9 && m <= c.upper() + 1e-9));
        let _ = target; // target outlives the closure via clone
    }

    #[test]
    fn pgd_keeps_feasible_model_unchanged_at_optimum() {
        // if the start already equals the (feasible) target, the gradient is zero
        let c = BoxConstraints::new(0.0, 10.0);
        let target = Array3::from_elem((4, 4, 4), 5.0);
        let t = target.clone();
        let result = projected_gradient_descent(target, c, 0.3, 50, move |m| m - &t);
        assert!(result.iter().all(|&m| (m - 5.0).abs() < 1e-12));
    }
}
