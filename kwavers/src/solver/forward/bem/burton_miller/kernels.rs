use num_complex::Complex64;
use std::f64::consts::PI;

use super::assembler::BurtonMillerAssembler;

impl BurtonMillerAssembler {
    pub(super) fn distance(&self, p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        (p1[2] - p2[2]).mul_add(p1[2] - p2[2], (p1[1] - p2[1]).mul_add(p1[1] - p2[1], (p1[0] - p2[0]).powi(2))).sqrt()
    }

    pub(super) fn greens_function_helmholtz(&self, k: f64, r: f64) -> Complex64 {
        let phase = Complex64::new(0.0, k * r);
        phase.exp() / (4.0 * PI * r)
    }

    /// ∂G/∂n_y — full 3D dot product (Colton & Kress 2013, §3.1, Eq. 3.41).
    pub(super) fn greens_function_normal_derivative_full(
        &self,
        k: f64,
        r: f64,
        collocation: &[f64; 3],
        point: &[f64; 3],
        normal_y: &[f64; 3],
    ) -> Complex64 {
        if r < 1e-12 {
            return Complex64::new(0.0, 0.0);
        }
        let rhat = [
            (point[0] - collocation[0]) / r,
            (point[1] - collocation[1]) / r,
            (point[2] - collocation[2]) / r,
        ];
        let cos_ny = rhat[2].mul_add(normal_y[2], rhat[0].mul_add(normal_y[0], rhat[1] * normal_y[1]));
        let g = self.greens_function_helmholtz(k, r);
        let alpha = Complex64::new(0.0, k) - Complex64::new(1.0 / r, 0.0);
        g * alpha * cos_ny
    }

    /// Legacy 1-D approximation of ∂G/∂n (x-axis normal only).
    pub(super) fn greens_function_normal_derivative(
        &self,
        k: f64,
        r: f64,
        collocation: &[f64; 3],
        point: &[f64; 3],
    ) -> Complex64 {
        if r < 1e-12 {
            return Complex64::new(0.0, 0.0);
        }
        let dr_dn = (collocation[0] - point[0]) / r;
        let phase = Complex64::new(0.0, k * r);
        let exp_ikr = phase.exp();
        let dg_dr =
            (Complex64::new(0.0, k) - Complex64::new(1.0 / r, 0.0)) * exp_ikr / (4.0 * PI * r * r);
        dg_dr * dr_dn
    }

    /// ∂²G/(∂n_x ∂n_y) — hypersingular kernel (Colton & Kress 2013, §3.3, Theorem 3.3).
    pub(super) fn greens_function_double_normal_derivative(
        &self,
        k: f64,
        r: f64,
        collocation: &[f64; 3],
        point: &[f64; 3],
        normal_y: &[f64; 3],
        normal_x: &[f64; 3],
    ) -> Complex64 {
        if r < 1e-10 {
            return Complex64::new(0.0, 0.0);
        }
        let rhat = [
            (point[0] - collocation[0]) / r,
            (point[1] - collocation[1]) / r,
            (point[2] - collocation[2]) / r,
        ];
        let cos_nx = rhat[2].mul_add(normal_x[2], rhat[0].mul_add(normal_x[0], rhat[1] * normal_x[1]));
        let cos_ny = rhat[2].mul_add(normal_y[2], rhat[0].mul_add(normal_y[0], rhat[1] * normal_y[1]));
        let nxny =
            normal_x[2].mul_add(normal_y[2], normal_x[0].mul_add(normal_y[0], normal_x[1] * normal_y[1]));
        let g = self.greens_function_helmholtz(k, r);
        let coeff_cos = Complex64::new(k.mul_add(k, -(3.0 / (r * r))), 3.0 * k / r);
        let coeff_nx = Complex64::new(1.0 / (r * r), -k / r);
        g * (coeff_cos * (cos_nx * cos_ny) + coeff_nx * nxny)
    }

    pub(super) fn triangle_normal(&self, p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> [f64; 3] {
        let e1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let e2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];
        let normal = [
            e1[1].mul_add(e2[2], -(e1[2] * e2[1])),
            e1[2].mul_add(e2[0], -(e1[0] * e2[2])),
            e1[0].mul_add(e2[1], -(e1[1] * e2[0])),
        ];
        let norm = normal[2].mul_add(normal[2], normal[1].mul_add(normal[1], normal[0].powi(2))).sqrt();
        if norm > 1e-12 {
            [normal[0] / norm, normal[1] / norm, normal[2] / norm]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    pub(super) fn triangle_area(&self, p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
        let e1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let e2 = [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]];
        let cross = [
            e1[1].mul_add(e2[2], -(e1[2] * e2[1])),
            e1[2].mul_add(e2[0], -(e1[0] * e2[2])),
            e1[0].mul_add(e2[1], -(e1[1] * e2[0])),
        ];
        0.5 * cross[2].mul_add(cross[2], cross[1].mul_add(cross[1], cross[0].powi(2))).sqrt()
    }
}
