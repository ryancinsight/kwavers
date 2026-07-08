//! Christoffel equation solver for wave propagation in anisotropic media
//!
//! References:
//! - Auld, B. A. (1973). "Acoustic Fields and Waves in Solids"

use super::stiffness::AnisotropicStiffnessTensor;
use kwavers_core::error::KwaversResult;
use leto::{Array1, Array2};

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
        let mut gamma = Array2::zeros([3, 3]);
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
        use leto::application::fixed::FixedMatrix;
        let g = self.christoffel_matrix(direction);
        let m = FixedMatrix::from_rows([
            [g[[0, 0]], g[[0, 1]], g[[0, 2]]],
            [g[[1, 0]], g[[1, 1]], g[[1, 2]]],
            [g[[2, 0]], g[[2, 1]], g[[2, 2]]],
        ]);
        let (eigenvalues, eigenvectors) = m.symmetric_eigen();
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let mut vals = [0.0; 3];
        let mut vecs = [[0.0; 3]; 3];
        for (k, &o) in order.iter().enumerate() {
            vals[k] = eigenvalues[o];
            vecs[k] = [
                eigenvectors[(0, o)],
                eigenvectors[(1, o)],
                eigenvectors[(2, o)],
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

    /// Full-tensor stiffness component `C_ijkl` recovered from the Voigt 6×6
    /// matrix (no factor-of-2: those live in the strain–Voigt convention, not in
    /// the stiffness recovery — consistent with [`christoffel_matrix`]).
    ///
    /// [`christoffel_matrix`]: Self::christoffel_matrix
    #[inline]
    fn c_ijkl(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        self.stiffness.c[[voigt_index(i, j), voigt_index(k, l)]]
    }

    /// **Group (energy) velocity** vector for each wave mode \[m·s⁻¹], in the
    /// same descending order as [`phase_velocities`].
    ///
    /// Energy propagates along `V_g`, which in an anisotropic medium deviates
    /// from the phase-propagation direction `n̂` (beam steering / "walk-off").
    /// For mode `m` with unit polarization `p` and phase speed `V_p`,
    ///
    /// ```text
    /// V_{g,i} = (1/(ρ V_p)) · Σ_{jkl} C_ijkl p_j p_k n̂_l        (Auld 1973, §7).
    /// ```
    ///
    /// In an **isotropic** medium `V_g = V_p·n̂` (energy and phase coincide). The
    /// input direction is normalised internally. A degenerate (zero-speed) mode
    /// yields a zero vector.
    /// # Errors
    /// - Returns [`Err`] if the medium density is non-positive or `direction` is
    ///   the zero vector.
    ///
    /// [`phase_velocities`]: Self::phase_velocities
    pub fn group_velocities(&self, direction: &[f64; 3]) -> KwaversResult<[[f64; 3]; 3]> {
        let norm = direction[2]
            .mul_add(
                direction[2],
                direction[0].mul_add(direction[0], direction[1] * direction[1]),
            )
            .sqrt();
        if self.density <= 0.0 || norm <= 0.0 {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "group_velocities: positive density and non-zero direction required".to_owned(),
            ));
        }
        let n = [
            direction[0] / norm,
            direction[1] / norm,
            direction[2] / norm,
        ];
        let (vals, vecs) = self.sorted_eigen(&n);

        let mut out = [[0.0_f64; 3]; 3];
        for m in 0..3 {
            let v_phase = (vals[m].max(0.0) / self.density).sqrt();
            if v_phase <= 1e-30 {
                continue;
            }
            let p = vecs[m];
            for (i, vi) in out[m].iter_mut().enumerate() {
                let mut acc = 0.0;
                for (j, &pj) in p.iter().enumerate() {
                    for (k, &pk) in p.iter().enumerate() {
                        for (l, &nl) in n.iter().enumerate() {
                            acc += self.c_ijkl(i, j, k, l) * pj * pk * nl;
                        }
                    }
                }
                *vi = acc / (self.density * v_phase);
            }
        }
        Ok(out)
    }

    /// Maximum **quasi-longitudinal** phase speed over propagation directions
    /// \[m·s⁻¹] — the reference speed for an anisotropic-medium CFL/timestep
    /// bound. Computed by sampling the unit sphere (principal axes, face/body
    /// diagonals, and a deterministic Fibonacci sphere) and taking the largest
    /// qP speed; phase velocity is `n → −n` symmetric so a sampled set suffices.
    ///
    /// For an isotropic medium this returns `c_P = √((λ+2μ)/ρ)` exactly (the
    /// speed is direction-independent). For strongly anisotropic media apply the
    /// usual CFL safety margin, since this is a sampled (not analytic) supremum.
    /// # Errors
    /// - Propagates errors from [`phase_velocities`](Self::phase_velocities).
    pub fn max_phase_velocity(&self) -> KwaversResult<f64> {
        let mut v_max = 0.0_f64;
        for dir in sample_propagation_directions() {
            let v = self.phase_velocities(&dir)?;
            v_max = v_max.max(v[0]); // v[0] is the (descending-sorted) qP speed
        }
        Ok(v_max)
    }
}

/// Deterministic set of unit propagation directions covering a hemisphere
/// (phase velocity is even in `n`): principal axes, face/body diagonals, and a
/// Fibonacci sphere for general anisotropy.
fn sample_propagation_directions() -> Vec<[f64; 3]> {
    let mut dirs = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let s2 = 1.0 / 2.0_f64.sqrt();
    for &(a, b, c) in &[
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, -1.0),
    ] {
        dirs.push([a * s2, b * s2, c * s2]);
    }
    let s3 = 1.0 / 3.0_f64.sqrt();
    for &(a, b, c) in &[
        (1.0, 1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, -1.0, -1.0),
    ] {
        dirs.push([a * s3, b * s3, c * s3]);
    }
    // Fibonacci sphere (golden-angle spiral) for off-symmetry extrema.
    const N: usize = 96;
    const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653; // π(3 − √5)
    for i in 0..N {
        let z = 1.0 - (2.0 * i as f64 + 1.0) / N as f64;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let theta = GOLDEN_ANGLE * i as f64;
        dirs.push([r * theta.cos(), r * theta.sin(), z]);
    }
    dirs
}

/// Voigt index of a tensor index pair: `(0,0)→0, (1,1)→1, (2,2)→2,
/// (1,2)/(2,1)→3, (0,2)/(2,0)→4, (0,1)/(1,0)→5`.
#[inline]
fn voigt_index(i: usize, j: usize) -> usize {
    match (i, j) {
        (0, 0) => 0,
        (1, 1) => 1,
        (2, 2) => 2,
        (1, 2) | (2, 1) => 3,
        (0, 2) | (2, 0) => 4,
        (0, 1) | (1, 0) => 5,
        _ => unreachable!("tensor indices i, j are always in 0..3"),
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
        let dirs = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], {
            let s = 1.0 / 3.0_f64.sqrt();
            [s, s, s]
        }];
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
        let c =
            AnisotropicStiffnessTensor::transversely_isotropic(10.0e9, 8.0e9, 4.0e9, 12.0e9, 3.0e9)
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

    /// In an **isotropic** medium energy and phase velocities coincide:
    /// `V_g = V_p·n̂` for every mode (both magnitude and direction), so |V_g|
    /// equals the phase speed and V_g is parallel to the propagation direction.
    /// This is the discriminating check for the energy-velocity contraction.
    #[test]
    fn isotropic_group_velocity_equals_phase_velocity_along_n() {
        let (lambda, mu, rho) = (5.0e9, 2.0e9, 2000.0);
        let solver =
            ChristoffelEquation::create(AnisotropicStiffnessTensor::isotropic(lambda, mu), rho);
        let c_p = ((lambda + 2.0 * mu) / rho).sqrt();
        let c_s = (mu / rho).sqrt();

        // Off-axis direction to exercise the full contraction.
        let s = 1.0 / 3.0_f64.sqrt();
        let n = [s, s, s];
        let vg = solver.group_velocities(&n).unwrap();
        let speeds = [c_p, c_s, c_s];
        for (m, (mode_vg, &speed)) in vg.iter().zip(speeds.iter()).enumerate() {
            // V_g = speed · n̂  (parallel to n, magnitude = phase speed).
            for (i, (&component, &direction)) in mode_vg.iter().zip(n.iter()).enumerate() {
                assert!(
                    (component - speed * direction).abs() < 1e-3 * c_p,
                    "mode {m} comp {i}: {} vs {}",
                    component,
                    speed * direction
                );
            }
            let mag = (mode_vg[0].powi(2) + mode_vg[1].powi(2) + mode_vg[2].powi(2)).sqrt();
            assert!((mag - speed).abs() < 1e-3 * c_p, "|V_g| mode {m}");
        }
        // Zero direction / bad density rejected.
        assert!(solver.group_velocities(&[0.0, 0.0, 0.0]).is_err());
    }

    /// In a genuinely anisotropic (transversely isotropic) medium the group
    /// velocity generally differs from `V_p·n̂` off the symmetry axis (energy
    /// walk-off) — but along the symmetry axis (z) the quasi-longitudinal energy
    /// velocity is still axial.
    #[test]
    fn anisotropic_group_velocity_is_finite_and_axial_on_symmetry_axis() {
        let c =
            AnisotropicStiffnessTensor::transversely_isotropic(10.0e9, 8.0e9, 4.0e9, 12.0e9, 3.0e9)
                .expect("valid TI");
        let solver = ChristoffelEquation::create(c, 1500.0);
        // Along z (symmetry axis): qP energy velocity is purely axial.
        let vg = solver.group_velocities(&[0.0, 0.0, 1.0]).unwrap();
        assert!(
            vg[0][0].abs() < 1e-6 && vg[0][1].abs() < 1e-6,
            "qP V_g axial on z"
        );
        assert!(vg[0][2] > 0.0, "qP energy travels +z");
        // All components finite.
        for mode_vg in &vg {
            for component in mode_vg {
                assert!(component.is_finite());
            }
        }
    }

    /// `max_phase_velocity` returns the isotropic `c_P` exactly (the speed is
    /// direction-independent, so every sampled direction agrees), and for a TI
    /// medium it is at least the on-axis quasi-P speed (a sampled supremum).
    #[test]
    fn max_phase_velocity_isotropic_and_ti() {
        let (lambda, mu, rho) = (5.0e9, 2.0e9, 2000.0);
        let iso =
            ChristoffelEquation::create(AnisotropicStiffnessTensor::isotropic(lambda, mu), rho);
        let c_p = ((lambda + 2.0 * mu) / rho).sqrt();
        let vmax = iso.max_phase_velocity().unwrap();
        assert!(
            (vmax - c_p).abs() < 1e-6 * c_p,
            "isotropic v_max {vmax} vs c_P {c_p}"
        );

        let ti = ChristoffelEquation::create(
            AnisotropicStiffnessTensor::transversely_isotropic(10.0e9, 8.0e9, 4.0e9, 12.0e9, 3.0e9)
                .unwrap(),
            1500.0,
        );
        let vmax_ti = ti.max_phase_velocity().unwrap();
        let on_axis_qp = ti.phase_velocities(&[0.0, 0.0, 1.0]).unwrap()[0];
        assert!(
            vmax_ti >= on_axis_qp - 1e-6,
            "TI v_max {vmax_ti} ≥ on-axis qP {on_axis_qp}"
        );
        assert!(vmax_ti.is_finite() && vmax_ti > 0.0);
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
        for (s, polarization) in pol.iter().enumerate().skip(1).take(2) {
            let dot =
                polarization[0] * dir[0] + polarization[1] * dir[1] + polarization[2] * dir[2];
            assert!(dot.abs() < 1e-6, "qS{s} polarization must be transverse");
        }
    }
}
