//! Christoffel equation solver for wave propagation in anisotropic media
//!
//! References:
//! - Auld, B. A. (1973). "Acoustic Fields and Waves in Solids"

use super::stiffness::AnisotropicStiffnessTensor;
use kwavers_core::error::KwaversResult;
use ndarray::{Array1, Array2};

/// Christoffel equation solver for anisotropic wave propagation
#[derive(Debug)]
pub struct ChristoffelEquation {
    /// Stiffness tensor
    stiffness: AnisotropicStiffnessTensor,
    /// Material density
    density: f64,
}

impl ChristoffelEquation {
    /// Create Christoffel equation solver
    #[must_use]
    pub fn create(stiffness: AnisotropicStiffnessTensor, density: f64) -> Self {
        Self { stiffness, density }
    }

    /// Compute Christoffel matrix for given propagation direction
    #[must_use]
    pub fn christoffel_matrix(&self, direction: &[f64; 3]) -> Array2<f64> {
        let mut gamma = Array2::zeros((3, 3));
        let c = &self.stiffness.c;
        let n = direction;

        // Γik = Cijkl * nj * nl (Einstein summation)
        gamma[[0, 0]] = 2.0f64.mul_add(
            (c[[4, 5]] * n[1]).mul_add(
                n[2],
                (c[[0, 5]] * n[0]).mul_add(n[1], c[[0, 4]] * n[0] * n[2]),
            ),
            (c[[4, 4]] * n[2]).mul_add(
                n[2],
                (c[[0, 0]] * n[0]).mul_add(n[0], c[[5, 5]] * n[1] * n[1]),
            ),
        );

        gamma[[1, 1]] = 2.0f64.mul_add(
            (c[[1, 3]] * n[1]).mul_add(
                n[2],
                (c[[1, 5]] * n[0]).mul_add(n[1], c[[3, 5]] * n[0] * n[2]),
            ),
            (c[[3, 3]] * n[2]).mul_add(
                n[2],
                (c[[5, 5]] * n[0]).mul_add(n[0], c[[1, 1]] * n[1] * n[1]),
            ),
        );

        gamma[[2, 2]] = 2.0f64.mul_add(
            (c[[2, 3]] * n[1]).mul_add(
                n[2],
                (c[[3, 4]] * n[0]).mul_add(n[1], c[[2, 4]] * n[0] * n[2]),
            ),
            (c[[2, 2]] * n[2]).mul_add(
                n[2],
                (c[[4, 4]] * n[0]).mul_add(n[0], c[[3, 3]] * n[1] * n[1]),
            ),
        );

        // Off-diagonal terms (symmetric)
        gamma[[0, 1]] = ((c[[1, 4]] + c[[3, 5]]) * n[1]).mul_add(
            n[2],
            ((c[[0, 3]] + c[[4, 5]]) * n[0]).mul_add(
                n[2],
                ((c[[0, 1]] + c[[5, 5]]) * n[0]).mul_add(
                    n[1],
                    (c[[3, 4]] * n[2]).mul_add(
                        n[2],
                        (c[[0, 5]] * n[0]).mul_add(n[0], c[[1, 5]] * n[1] * n[1]),
                    ),
                ),
            ),
        );
        gamma[[1, 0]] = gamma[[0, 1]];

        gamma[[0, 2]] = ((c[[2, 5]] + c[[3, 4]]) * n[1]).mul_add(
            n[2],
            ((c[[0, 2]] + c[[4, 4]]) * n[0]).mul_add(
                n[2],
                ((c[[0, 3]] + c[[4, 5]]) * n[0]).mul_add(
                    n[1],
                    (c[[2, 4]] * n[2]).mul_add(
                        n[2],
                        (c[[0, 4]] * n[0]).mul_add(n[0], c[[3, 5]] * n[1] * n[1]),
                    ),
                ),
            ),
        );
        gamma[[2, 0]] = gamma[[0, 2]];

        gamma[[1, 2]] = ((c[[1, 2]] + c[[3, 3]]) * n[1]).mul_add(
            n[2],
            ((c[[2, 5]] + c[[3, 4]]) * n[0]).mul_add(
                n[2],
                ((c[[1, 4]] + c[[3, 5]]) * n[0]).mul_add(
                    n[1],
                    (c[[2, 3]] * n[2]).mul_add(
                        n[2],
                        (c[[4, 5]] * n[0]).mul_add(n[0], c[[1, 3]] * n[1] * n[1]),
                    ),
                ),
            ),
        );
        gamma[[2, 1]] = gamma[[1, 2]];

        gamma
    }

    /// Sorted eigendecomposition of the Christoffel matrix along `direction`.
    ///
    /// The Christoffel matrix `Γᵢₖ = Cᵢⱼₖₗ nⱼ nₗ` is real and symmetric, so a
    /// symmetric eigensolver yields exact real eigenvalues `ρv²` **including
    /// degenerate (repeated) roots** — e.g. the two equal quasi-shear modes of
    /// an isotropic or on-axis transversely-isotropic medium. Returns the
    /// `(eigenvalue, eigenvector)` pairs sorted by **descending** eigenvalue, so
    /// index 0 is the quasi-longitudinal mode and 1, 2 the quasi-shear modes —
    /// the same ordering for both [`phase_velocities`] and
    /// [`polarization_vectors`].
    ///
    /// [`phase_velocities`]: Self::phase_velocities
    /// [`polarization_vectors`]: Self::polarization_vectors
    fn sorted_eigen(&self, direction: &[f64; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
        use nalgebra::{Matrix3, SymmetricEigen};
        let g = self.christoffel_matrix(direction);
        let m = Matrix3::new(
            g[[0, 0]], g[[0, 1]], g[[0, 2]], g[[1, 0]], g[[1, 1]], g[[1, 2]], g[[2, 0]], g[[2, 1]],
            g[[2, 2]],
        );
        let eig = SymmetricEigen::new(m);
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| {
            eig.eigenvalues[b]
                .partial_cmp(&eig.eigenvalues[a])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let mut vals = [0.0; 3];
        let mut vecs = [[0.0; 3]; 3];
        for (k, &o) in order.iter().enumerate() {
            vals[k] = eig.eigenvalues[o];
            vecs[k] = [
                eig.eigenvectors[(0, o)],
                eig.eigenvectors[(1, o)],
                eig.eigenvectors[(2, o)],
            ];
        }
        (vals, vecs)
    }

    /// Solve for the three phase velocities along `direction`, sorted descending
    /// (quasi-longitudinal first, then the two quasi-shear modes).
    ///
    /// `vₖ = √(ρv²ₖ / ρ)` from the Christoffel eigenvalues. A real symmetric
    /// eigensolver is used, so **degenerate cases (isotropic / on-axis) are
    /// handled exactly** rather than collapsing to a fallback.
    /// # Errors
    /// - Returns [`Err`] if the medium density is non-positive.
    pub fn phase_velocities(&self, direction: &[f64; 3]) -> KwaversResult<[f64; 3]> {
        if self.density <= 0.0 {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "ChristoffelEquation density must be positive".to_owned(),
            ));
        }
        let (vals, _) = self.sorted_eigen(direction);
        Ok([
            (vals[0].max(0.0) / self.density).sqrt(),
            (vals[1].max(0.0) / self.density).sqrt(),
            (vals[2].max(0.0) / self.density).sqrt(),
        ])
    }

    /// Get polarization (particle-motion) unit vectors for each wave mode, in
    /// the same descending-eigenvalue order as [`phase_velocities`]
    /// (quasi-longitudinal, quasi-shear, quasi-shear).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// [`phase_velocities`]: Self::phase_velocities
    pub fn polarization_vectors(&self, direction: &[f64; 3]) -> KwaversResult<[Array1<f64>; 3]> {
        let (_, vecs) = self.sorted_eigen(direction);
        Ok([
            Array1::from(vecs[0].to_vec()),
            Array1::from(vecs[1].to_vec()),
            Array1::from(vecs[2].to_vec()),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anisotropic::stiffness::AnisotropicStiffnessTensor;

    /// The previously-broken degenerate case: an **isotropic** medium has two
    /// equal quasi-shear speeds, so the characteristic cubic has a repeated root
    /// (discriminant = 0). The symmetric eigensolver recovers the exact Lamé
    /// speeds `c_P=√((λ+2μ)/ρ)`, `c_S=√(μ/ρ)` (twice) along any direction —
    /// where the old Cardano fallback returned a bogus [1,1,1].
    #[test]
    fn isotropic_phase_velocities_recover_lame_speeds() {
        let (lambda, mu, rho) = (5.0e9, 2.0e9, 2000.0);
        let c = AnisotropicStiffnessTensor::isotropic(lambda, mu);
        let solver = ChristoffelEquation::create(c, rho);

        let c_p = ((lambda + 2.0 * mu) / rho).sqrt();
        let c_s = (mu / rho).sqrt();
        // Try several propagation directions, including off-axis.
        let dirs = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            {
                let s = 1.0 / 3.0_f64.sqrt();
                [s, s, s]
            },
        ];
        for d in &dirs {
            let v = solver.phase_velocities(d).unwrap();
            assert!((v[0] - c_p).abs() < 1e-3 * c_p, "qP {} vs {c_p}", v[0]);
            assert!((v[1] - c_s).abs() < 1e-3 * c_s, "qS1 {} vs {c_s}", v[1]);
            assert!((v[2] - c_s).abs() < 1e-3 * c_s, "qS2 {} vs {c_s}", v[2]);
            assert!(v[0] > v[1], "longitudinal must be fastest");
        }
    }

    /// Eigenvalue invariant: `Σ ρv²ₖ = tr(Γ)` along any direction (the trace of
    /// the Christoffel matrix), for a genuinely anisotropic (transversely
    /// isotropic) tensor.
    #[test]
    fn phase_velocity_eigenvalues_match_christoffel_trace() {
        let c = AnisotropicStiffnessTensor::transversely_isotropic(10.0e9, 8.0e9, 4.0e9, 12.0e9, 3.0e9)
            .expect("valid TI tensor");
        let rho = 1500.0;
        let solver = ChristoffelEquation::create(c, rho);
        let dir = {
            let s = 1.0 / 2.0_f64.sqrt();
            [s, 0.0, s]
        };
        let gamma = solver.christoffel_matrix(&dir);
        let trace = gamma[[0, 0]] + gamma[[1, 1]] + gamma[[2, 2]];
        let v = solver.phase_velocities(&dir).unwrap();
        let sum_rho_v2: f64 = v.iter().map(|&vi| rho * vi * vi).sum();
        assert!(
            (sum_rho_v2 - trace).abs() < 1e-3 * trace.abs(),
            "Σρv² {sum_rho_v2} must equal tr(Γ) {trace}"
        );
    }

    /// In an isotropic medium the quasi-longitudinal polarization is parallel to
    /// the propagation direction and the shear polarizations are orthogonal to
    /// it (and the velocity/polarization orderings agree).
    #[test]
    fn isotropic_polarizations_are_longitudinal_and_transverse() {
        let c = AnisotropicStiffnessTensor::isotropic(5.0e9, 2.0e9);
        let solver = ChristoffelEquation::create(c, 2000.0);
        let dir = [1.0, 0.0, 0.0];
        let pol = solver.polarization_vectors(&dir).unwrap();
        // qP ∥ dir ⇒ |pol₀ · dir| ≈ 1.
        let dot0 = pol[0][0] * dir[0] + pol[0][1] * dir[1] + pol[0][2] * dir[2];
        assert!(dot0.abs() > 0.999, "qP polarization must be longitudinal");
        // Shear modes ⊥ dir ⇒ |pol · dir| ≈ 0.
        for s in 1..=2 {
            let dot = pol[s][0] * dir[0] + pol[s][1] * dir[1] + pol[s][2] * dir[2];
            assert!(dot.abs() < 1e-6, "qS{s} polarization must be transverse");
        }
    }
}
