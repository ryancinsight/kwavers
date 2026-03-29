use crate::core::error::KwaversResult;
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
    /// Zero-allocation coupling — writes directly into `output` without cloning.
    ///
    /// ## Algorithm
    /// For each cell (i, j, k):
    /// - If `mask[i,j,k]` is `true` (DG region): blend s1 and s2 with weight
    ///   `w = clamp(|s1-s2| / tol, 0, 1)` so that closely-matching solutions
    ///   favour the spectral value and strongly-differing ones favour DG.
    /// - Otherwise (smooth spectral region): use s1 directly.
    ///
    /// `output` is fully overwritten; it does not need to be pre-initialized.
    fn couple_into(
        &self,
        solution1: &Array3<f64>,
        solution2: &Array3<f64>,
        mask: &Array3<bool>,
        _original: &Array3<f64>,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let tol = self.tolerance.max(f64::EPSILON);
        for ((out, &use_solution2), (&s1, &s2)) in output
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
        Ok(())
    }
}
