//! Murnaghan third-order elastic constitutive model (ADR 022).
//!
//! Generalises linear isotropic elasticity to **third order** in the
//! Green–Lagrange strain `E`, the material law whose small-strain pre-stress
//! limit is the analytical acousto-elastic relation in
//! [`super::elastography`] (ADR 014, `A = (m+n)/(2(λ+μ))`). This is the
//! constitutive core for nonlinear elastodynamics; the small-on-large
//! acousto-elastic tangent and the time-domain forward PDE are staged
//! follow-ons (ADR 022 "Status / staging").
//!
//! # Strain-energy density
//!
//! In the **power-sum invariant** convention of Chapter 11 §11.9.1 (the same
//! convention that defines the `(m, n)` consumed by
//! [`super::elastography::acoustoelastic_sensitivity`]), with
//! `I₁ = tr E`, `tr E²`, `tr E³`:
//!
//! ```text
//! W(E) = (λ/2)(tr E)² + μ tr(E²) + (l/3)(tr E)³ + m (tr E) tr(E²) + n tr(E³)
//! ```
//!
//! The second-order part is exactly St-Venant–Kirchhoff, so `l = m = n = 0`
//! recovers StVK and the small-strain limit recovers linear Hooke `(λ, μ)`.
//!
//! # Second Piola–Kirchhoff stress `S = ∂W/∂E`
//!
//! Using `∂(tr E)/∂E = I`, `∂ tr(E²)/∂E = 2E`, `∂ tr(E³)/∂E = 3E²`:
//!
//! ```text
//! S = [λ tr E + l (tr E)² + m tr(E²)] I + (2μ + 2m tr E) E + 3n E²
//! ```
//!
//! # References
//! - Murnaghan, F.D. (1951). *Finite Deformation of an Elastic Solid*. Wiley.
//! - Hughes, D.S. & Kelly, J.L. (1953). "Second-order elastic deformation of
//!   solids." *Phys. Rev.* 92(5), 1145–1149.
//! - Landau, L.D. & Lifshitz, E.M. (1986). *Theory of Elasticity* §26.

/// Symmetric 3×3 tensor in row-major dense form.
pub type Tensor3 = [[f64; 3]; 3];

/// `3×3` identity.
const IDENTITY: Tensor3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Trace `tr M = M₀₀ + M₁₁ + M₂₂`.
#[inline]
#[must_use]
pub fn trace(m: &Tensor3) -> f64 {
    m[0][0] + m[1][1] + m[2][2]
}

/// Frobenius double contraction `A : B = Σ_ij A_ij B_ij`.
#[inline]
#[must_use]
pub fn double_dot(a: &Tensor3, b: &Tensor3) -> f64 {
    let mut s = 0.0;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (x, y) in ra.iter().zip(rb.iter()) {
            s = x.mul_add(*y, s);
        }
    }
    s
}

/// `tr(M²) = Σ_ij M_ij M_ji`. For symmetric `M` this equals `M : M`.
#[inline]
#[must_use]
fn trace_of_square(m: &Tensor3) -> f64 {
    let mut s = 0.0;
    for (i, row) in m.iter().enumerate() {
        for (j, &mij) in row.iter().enumerate() {
            s = mij.mul_add(m[j][i], s);
        }
    }
    s
}

/// Matrix product `A·B`.
#[inline]
#[must_use]
fn matmul(a: &Tensor3, b: &Tensor3) -> Tensor3 {
    let mut out = [[0.0_f64; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for k in 0..3 {
                s = a[i][k].mul_add(b[k][j], s);
            }
            *cell = s;
        }
    }
    out
}

/// `a·I + b·E + c·F` (linear combination of identity, `E`, and a third tensor).
#[inline]
fn combine(a: f64, b: f64, e: &Tensor3, c: f64, f: &Tensor3) -> Tensor3 {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = c.mul_add(f[i][j], b.mul_add(e[i][j], a * IDENTITY[i][j]));
        }
    }
    out
}

/// Murnaghan elastic constants for an isotropic solid.
///
/// `lambda`, `mu` are the second-order Lamé parameters `Pa`; `l`, `m`, `n` are
/// the third-order Murnaghan constants `Pa` in the power-sum convention
/// (Chapter 11 §11.9.1). With `l = m = n = 0` the model is
/// St-Venant–Kirchhoff. The pair `(m, n)` feeds the analytical acousto-elastic
/// sensitivity `A = (m+n)/(2(λ+μ))` ([`super::elastography::acoustoelastic_sensitivity`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MurnaghanConstants {
    /// Lamé first parameter λ `Pa`.
    pub lambda: f64,
    /// Lamé second parameter μ (shear modulus) `Pa`.
    pub mu: f64,
    /// Murnaghan third-order constant l `Pa`.
    pub l: f64,
    /// Murnaghan third-order constant m `Pa`.
    pub m: f64,
    /// Murnaghan third-order constant n `Pa`.
    pub n: f64,
}

impl MurnaghanConstants {
    /// Construct from second- and third-order constants.
    #[must_use]
    pub fn new(lambda: f64, mu: f64, l: f64, m: f64, n: f64) -> Self {
        Self {
            lambda,
            mu,
            l,
            m,
            n,
        }
    }

    /// St-Venant–Kirchhoff special case (`l = m = n = 0`).
    #[must_use]
    pub fn saint_venant_kirchhoff(lambda: f64, mu: f64) -> Self {
        Self::new(lambda, mu, 0.0, 0.0, 0.0)
    }

    /// Strain-energy density `W(E)` `Pa` (= [J/m³]) for the symmetric
    /// Green–Lagrange strain `E`.
    #[must_use]
    pub fn strain_energy(&self, e: &Tensor3) -> f64 {
        let i1 = trace(e);
        let tr_e2 = trace_of_square(e);
        let e2 = matmul(e, e);
        let tr_e3 = double_dot(&e2, e); // tr(E³) for symmetric E
        let second = self.mu.mul_add(tr_e2, 0.5 * self.lambda * i1 * i1);
        let third = self.n.mul_add(
            tr_e3,
            self.m.mul_add(i1 * tr_e2, self.l / 3.0 * i1 * i1 * i1),
        );
        second + third
    }

    /// Second Piola–Kirchhoff stress `S = ∂W/∂E` `Pa` for the symmetric
    /// Green–Lagrange strain `E`.
    #[must_use]
    pub fn second_pk_stress(&self, e: &Tensor3) -> Tensor3 {
        let i1 = trace(e);
        let tr_e2 = trace_of_square(e);
        let e2 = matmul(e, e);
        // S = [λ I₁ + l I₁² + m tr(E²)] I + (2μ + 2m I₁) E + 3n E²
        let coeff_i = self
            .m
            .mul_add(tr_e2, self.l.mul_add(i1 * i1, self.lambda * i1));
        let coeff_e = 2.0_f64.mul_add(self.m * i1, 2.0 * self.mu);
        combine(coeff_i, coeff_e, e, 3.0 * self.n, &e2)
    }

    /// Apply the reference material tangent `ℂ₀ = ∂²W/∂E²|_{E=0}` to a symmetric
    /// increment `H`: `ℂ₀ : H = λ tr(H) I + 2μ H` (isotropic linear stiffness).
    #[must_use]
    pub fn apply_reference_tangent(&self, h: &Tensor3) -> Tensor3 {
        let tr = trace(h);
        combine(self.lambda * tr, 2.0 * self.mu, h, 0.0, &IDENTITY)
    }

    /// Apply the **finite-strain** material tangent `ℂ(E) = ∂²W/∂E² = ∂S/∂E`
    /// to a symmetric increment `H`, returning `ℂ(E) : H` `Pa`.
    ///
    /// Differentiating `S` (the power-sum form) once more:
    ///
    /// ```text
    /// ℂ(E):H = [λ trH + 2l I₁ trH + 2m (E:H)] I
    ///          + 2m trH · E
    ///          + (2μ + 2m I₁) H
    ///          + 3n (H·E + E·H)
    /// ```
    ///
    /// with `I₁ = tr E`. At `E = 0` this reduces to the reference tangent
    /// `λ trH I + 2μ H`. As the second derivative of a potential it is
    /// **major-symmetric** (`ℂ_{ijkl}=ℂ_{klij}`), the prerequisite for the
    /// acousto-elastic acoustic tensor and for implicit time integration.
    #[must_use]
    pub fn material_tangent(&self, e: &Tensor3, h: &Tensor3) -> Tensor3 {
        let i1 = trace(e);
        let tr_h = trace(h);
        let e_dd_h = double_dot(e, h);
        let he = matmul(h, e);
        let eh = matmul(e, h);

        let coeff_i = (2.0 * self.l).mul_add(i1 * tr_h, self.lambda * tr_h) + 2.0 * self.m * e_dd_h;
        let coeff_e = 2.0 * self.m * tr_h;
        let coeff_h = 2.0_f64.mul_add(self.m * i1, 2.0 * self.mu);
        let coeff_geom = 3.0 * self.n;

        let mut out = [[0.0_f64; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let geom = he[i][j] + eh[i][j];
                *cell = coeff_geom.mul_add(
                    geom,
                    coeff_h.mul_add(h[i][j], coeff_e.mul_add(e[i][j], coeff_i * IDENTITY[i][j])),
                );
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hooke(lambda: f64, mu: f64, e: &Tensor3) -> Tensor3 {
        // σ = λ tr(E) I + 2μ E
        let tr = trace(e);
        combine(lambda * tr, 2.0 * mu, e, 0.0, &IDENTITY)
    }

    fn frob_norm(m: &Tensor3) -> f64 {
        double_dot(m, m).sqrt()
    }

    fn sub(a: &Tensor3, b: &Tensor3) -> Tensor3 {
        let mut out = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[i][j] - b[i][j];
            }
        }
        out
    }

    fn axpy(a: &Tensor3, t: f64, h: &Tensor3) -> Tensor3 {
        let mut out = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = h[i][j].mul_add(t, a[i][j]);
            }
        }
        out
    }

    // Representative soft-solid constants (third-order constants negative, as
    // for most real materials).
    fn constants() -> MurnaghanConstants {
        MurnaghanConstants::new(2.0e9, 1.0e6, -3.0e9, -2.0e9, -1.0e9)
    }

    /// With l = m = n = 0 the model is exactly St-Venant–Kirchhoff:
    /// S(E) == λ tr(E) I + 2μ E for arbitrary symmetric E.
    #[test]
    fn reduces_to_saint_venant_kirchhoff_when_third_order_zero() {
        let svk = MurnaghanConstants::saint_venant_kirchhoff(2.0e9, 5.0e5);
        let e: Tensor3 = [
            [0.01, 0.004, -0.002],
            [0.004, -0.006, 0.003],
            [-0.002, 0.003, 0.008],
        ];
        let s = svk.second_pk_stress(&e);
        let expected = hooke(2.0e9, 5.0e5, &e);
        assert!(
            frob_norm(&sub(&s, &expected)) <= 1e-6 * frob_norm(&expected),
            "StVK stress must equal Hooke; got {s:?} vs {expected:?}"
        );
    }

    /// As E → 0 the third-order terms (O(E²)) vanish: the relative deviation of
    /// the Murnaghan stress from linear Hooke shrinks with the strain scale.
    #[test]
    fn linear_limit_recovers_hooke() {
        let c = constants();
        let dir: Tensor3 = [[1.0, 0.3, -0.2], [0.3, -0.6, 0.4], [-0.2, 0.4, 0.5]];
        let mut prev_rel = f64::INFINITY;
        for k in 1..6 {
            let scale = 10f64.powi(-k); // 1e-1 … 1e-5
            let e = axpy(&[[0.0; 3]; 3], scale, &dir);
            let s = c.second_pk_stress(&e);
            let h = hooke(c.lambda, c.mu, &e);
            let rel = frob_norm(&sub(&s, &h)) / frob_norm(&h);
            assert!(
                rel < prev_rel,
                "Hooke deviation must shrink with strain (scale {scale}): {rel} !< {prev_rel}"
            );
            prev_rel = rel;
        }
        assert!(
            prev_rel < 1e-3,
            "deep linear limit must be near-Hooke: {prev_rel}"
        );
    }

    /// Uniaxial closed form (power-sum convention): for E = diag(e,0,0),
    /// S_xx = (λ+2μ)e + (l+3m+3n)e²,  S_yy = S_zz = λe + (l+m)e²,  off-diag 0.
    #[test]
    fn uniaxial_strain_matches_closed_form() {
        let c = constants();
        let e_val = 0.02;
        let e: Tensor3 = [[e_val, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let s = c.second_pk_stress(&e);

        let e2 = e_val * e_val;
        let sxx = (c.l + 3.0 * c.m + 3.0 * c.n).mul_add(e2, (c.lambda + 2.0 * c.mu) * e_val);
        let syy = (c.l + c.m).mul_add(e2, c.lambda * e_val);
        assert!((s[0][0] - sxx).abs() <= 1e-6 * sxx.abs().max(1.0), "S_xx");
        assert!((s[1][1] - syy).abs() <= 1e-6 * syy.abs().max(1.0), "S_yy");
        assert!((s[2][2] - syy).abs() <= 1e-6 * syy.abs().max(1.0), "S_zz");
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            assert!(
                s[i][j].abs() <= 1e-3,
                "off-diagonal S[{i}][{j}] must vanish: {}",
                s[i][j]
            );
        }
    }

    /// S = ∂W/∂E verified tensorially: for several symmetric directions H, the
    /// central finite difference of W along H equals S : H.
    #[test]
    fn stress_is_energy_gradient() {
        let c = constants();
        let e: Tensor3 = [
            [0.008, 0.003, -0.001],
            [0.003, -0.004, 0.002],
            [-0.001, 0.002, 0.006],
        ];
        let s = c.second_pk_stress(&e);

        let dirs: [Tensor3; 4] = [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], // symmetric shear
            [[0.2, 0.1, 0.3], [0.1, -0.4, 0.2], [0.3, 0.2, 0.5]],
        ];
        let eps = 1e-7;
        for h in &dirs {
            let w_plus = c.strain_energy(&axpy(&e, eps, h));
            let w_minus = c.strain_energy(&axpy(&e, -eps, h));
            let fd = (w_plus - w_minus) / (2.0 * eps);
            let analytic = double_dot(&s, h);
            let scale = analytic.abs().max(1.0);
            assert!(
                (fd - analytic).abs() <= 1e-4 * scale,
                "S:H must match dW/dH; fd={fd}, S:H={analytic}"
            );
        }
    }

    /// S is symmetric for symmetric E.
    #[test]
    fn stress_is_symmetric() {
        let c = constants();
        let e: Tensor3 = [
            [0.01, 0.005, -0.003],
            [0.005, -0.007, 0.004],
            [-0.003, 0.004, 0.009],
        ];
        let s = c.second_pk_stress(&e);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            assert!(
                (s[i][j] - s[j][i]).abs() <= 1e-9 * frob_norm(&s).max(1.0),
                "S must be symmetric at ({i},{j})"
            );
        }
    }

    /// Reference tangent equals the isotropic linear stiffness and recovers
    /// (λ, μ): a hydrostatic increment gives pressure 3λ+2μ, a pure shear gives 2μ.
    #[test]
    fn reference_tangent_is_isotropic_linear_stiffness() {
        let c = constants();
        // Hydrostatic H = I: ℂ₀:I = (3λ+2μ) I.
        let hyd = c.apply_reference_tangent(&IDENTITY);
        let expected_diag = 3.0f64.mul_add(c.lambda, 2.0 * c.mu);
        assert!((hyd[0][0] - expected_diag).abs() <= 1e-6 * expected_diag.abs());
        assert!(hyd[0][1].abs() <= 1e-6 * expected_diag.abs());

        // Pure shear H (off-diagonal): ℂ₀:H = 2μ H (trace 0 kills λ term).
        let shear: Tensor3 = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let out = c.apply_reference_tangent(&shear);
        assert!((out[0][1] - 2.0 * c.mu).abs() <= 1e-6 * (2.0 * c.mu));
        assert!(out[0][0].abs() <= 1e-6 * (2.0 * c.mu));
    }

    /// At E = 0 the finite-strain tangent reduces to the reference (isotropic
    /// linear) tangent for arbitrary increments.
    #[test]
    fn finite_tangent_reduces_to_reference_at_zero_strain() {
        let c = constants();
        let zero = [[0.0_f64; 3]; 3];
        let dirs: [Tensor3; 2] = [
            [[1.0, 0.2, 0.0], [0.2, -0.5, 0.3], [0.0, 0.3, 0.4]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ];
        for h in &dirs {
            let finite = c.material_tangent(&zero, h);
            let reference = c.apply_reference_tangent(h);
            assert!(
                frob_norm(&sub(&finite, &reference)) <= 1e-6 * frob_norm(&reference).max(1.0),
                "ℂ(0):H must equal the reference tangent"
            );
        }
    }

    /// ℂ(E) = ∂S/∂E verified by central finite difference of `second_pk_stress`
    /// along several increments H (component-wise).
    #[test]
    fn material_tangent_is_stress_gradient() {
        let c = constants();
        let e: Tensor3 = [
            [0.006, 0.002, -0.001],
            [0.002, -0.003, 0.0015],
            [-0.001, 0.0015, 0.004],
        ];
        let dirs: [Tensor3; 3] = [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.2, 0.1, 0.3], [0.1, -0.4, 0.2], [0.3, 0.2, 0.5]],
        ];
        let eps = 1e-7;
        for h in &dirs {
            let analytic = c.material_tangent(&e, h);
            let s_plus = c.second_pk_stress(&axpy(&e, eps, h));
            let s_minus = c.second_pk_stress(&axpy(&e, -eps, h));
            let mut fd = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    fd[i][j] = (s_plus[i][j] - s_minus[i][j]) / (2.0 * eps);
                }
            }
            let scale = frob_norm(&analytic).max(1.0);
            assert!(
                frob_norm(&sub(&fd, &analytic)) <= 1e-4 * scale,
                "ℂ(E):H must equal dS/dH; got {analytic:?} vs FD {fd:?}"
            );
        }
    }

    /// The tangent is major-symmetric: `H₁ : ℂ : H₂ == H₂ : ℂ : H₁`
    /// (it is the second derivative of a scalar potential W).
    #[test]
    fn material_tangent_is_major_symmetric() {
        let c = constants();
        let e: Tensor3 = [
            [0.01, 0.004, -0.002],
            [0.004, -0.006, 0.003],
            [-0.002, 0.003, 0.008],
        ];
        let h1: Tensor3 = [[0.3, 0.1, 0.0], [0.1, -0.2, 0.4], [0.0, 0.4, 0.5]];
        let h2: Tensor3 = [[-0.1, 0.5, 0.2], [0.5, 0.3, -0.1], [0.2, -0.1, 0.2]];
        let a = double_dot(&c.material_tangent(&e, &h1), &h2);
        let b = double_dot(&c.material_tangent(&e, &h2), &h1);
        assert!(
            (a - b).abs() <= 1e-6 * a.abs().max(1.0),
            "major symmetry violated: {a} vs {b}"
        );
    }

    /// `ℂ(E):H` is symmetric for symmetric H (minor symmetry on the output pair).
    #[test]
    fn material_tangent_output_is_symmetric() {
        let c = constants();
        let e: Tensor3 = [
            [0.01, 0.005, 0.0],
            [0.005, -0.004, 0.002],
            [0.0, 0.002, 0.006],
        ];
        let h: Tensor3 = [[0.2, 0.3, -0.1], [0.3, 0.1, 0.4], [-0.1, 0.4, 0.5]];
        let out = c.material_tangent(&e, &h);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            assert!(
                (out[i][j] - out[j][i]).abs() <= 1e-9 * frob_norm(&out).max(1.0),
                "ℂ(E):H must be symmetric at ({i},{j})"
            );
        }
    }
}
