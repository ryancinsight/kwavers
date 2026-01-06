use crate::error::KwaversResult;
use ndarray::Array3;

#[derive(Debug)]
pub struct HybridCoupler {
    tolerance: f64,
}

impl HybridCoupler {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }
}

impl super::traits::SolutionCoupling for HybridCoupler {
    fn couple(
        &self,
        solution1: &Array3<f64>,
        solution2: &Array3<f64>,
        mask: &Array3<bool>,
        original: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let tol = self.tolerance.max(f64::EPSILON);
        let mut result = original.clone();
        for ((out, &use_solution2), (&s1, &s2)) in result
            .iter_mut()
            .zip(mask.iter())
            .zip(solution1.iter().zip(solution2.iter()))
        {
            if use_solution2 {
                let diff = (s1 - s2).abs();
                let w = (diff / tol).min(1.0);
                *out = (1.0 - w) * s1 + w * s2;
            } else {
                *out = s1;
            }
        }
        Ok(result)
    }
}
