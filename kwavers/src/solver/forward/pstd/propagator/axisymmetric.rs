//! Axisymmetric (CylindricalAS) PSTD propagator -- WSWA-FFT radial operators.
//!
//! All per-time-step intermediate arrays are pre-allocated in AsContext::new
//! and reused on every step, eliminating heap allocation on the hot path.
//! The 2-D FFT plan is obtained from FFT_CACHE_2D (cost paid once per shape).
//!
//! References:
//!   k-Wave MATLAB source kspaceFirstOrderAS.m, WSWA-FFT case.
//!   Treeby et al. (2012). k-Wave axisymmetric documentation.

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::{Complex64, Fft2d, Shape2D, FFT_CACHE_2D};
use ndarray::{s, Array1, Array2, ArrayView2, Zip};
use std::f64::consts::PI;
use std::sync::Arc;

/// Precomputed operators and pre-allocated scratch buffers for WSWA-FFT
/// axisymmetric propagation.
#[derive(Debug)]
pub struct AsContext {
    pub nx: usize,
    pub nr: usize,
    pub nr_exp: usize,
    /// r_sg(m) = (m + 0.5) * dr
    pub r_sg: Array1<f64>,
    /// i * kz * exp(+i*kz*dr/2)
    pub ddy_k_shift_pos: Array1<Complex64>,
    /// i * kz (no shift)
    pub ddy_k: Array1<Complex64>,
    /// exp(-i*kz*dr/2)
    pub y_shift_neg: Array1<Complex64>,
    /// sinc(c_ref * |k| * dt/2) on (nx, nr_exp)
    pub kappa_2d: Array2<f64>,
    /// i * kx * exp(+i*kx*dx/2), shape (nx,)
    pub ddx_k_shift_pos: Array1<Complex64>,
    /// i * kx * exp(-i*kx*dx/2), shape (nx,)
    pub ddx_k_shift_neg: Array1<Complex64>,

    fft_plan: Arc<Fft2d>,

    // Pre-allocated real expansion buffers, shape (nx, nr_exp).
    p_exp: Array2<f64>,
    ux_exp: Array2<f64>,
    uz_exp: Array2<f64>,
    uz_on_r_exp: Array2<f64>,

    // Pre-allocated 2-D slice buffers, shape (nx, nr).
    p_2d: Array2<f64>,
    ux_2d: Array2<f64>,
    uz_2d: Array2<f64>,
    uz_on_r: Array2<f64>,

    // Pre-allocated k-space working buffers, shape (nx, nr_exp).
    ak: Array2<Complex64>,
    g: Array2<Complex64>,

    // Pre-allocated result buffers, shape (nx, nr).
    /// dp/dx -- filled by compute_vel_grads.
    pub dpdx: Array2<f64>,
    /// dp/dr -- filled by compute_vel_grads.
    pub dpdr: Array2<f64>,
    /// du_x/dx -- filled by compute_density_divs.
    pub duxdx: Array2<f64>,
    /// du_r/dr + u_r/r -- filled by compute_density_divs.
    pub duzdr: Array2<f64>,
    /// Scratch coefficient buffer for density update, shape (nx, nr).
    /// Eliminates the per-step Array2 allocation in update_density_as.
    pub coef: Array2<f64>,
}

impl AsContext {
    /// Construct AsContext from grid and solver parameters.
    /// Pre-allocates all expansion and k-space scratch buffers.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn new(
        nx: usize,
        nr: usize,
        dx: f64,
        dr: f64,
        c_ref: f64,
        dt: f64,
        ddx_k_shift_pos: Array1<Complex64>,
        ddx_k_shift_neg: Array1<Complex64>,
    ) -> KwaversResult<Self> {
        if nr == 0 {
            return Err(KwaversError::InvalidInput(
                "CylindricalAS requires nz (Nr) >= 1".into(),
            ));
        }
        let nr_exp = 4 * nr;
        let dk_z = 2.0 * PI / (nr_exp as f64 * dr);
        let dk_x = 2.0 * PI / (nx as f64 * dx);

        let r_sg = Array1::from_iter((0..nr).map(|m| (m as f64 + 0.5) * dr));

        let kz: Array1<f64> = Array1::from_iter((0..nr_exp).map(|k| {
            let ki = k as i64;
            let n = nr_exp as i64;
            if ki <= n / 2 {
                ki as f64 * dk_z
            } else {
                (ki - n) as f64 * dk_z
            }
        }));

        let ddy_k_shift_pos = kz
            .iter()
            .map(|&v| Complex64::new(0.0, v) * Complex64::from_polar(1.0, v * dr / 2.0))
            .collect();
        let ddy_k = kz.iter().map(|&v| Complex64::new(0.0, v)).collect();
        let y_shift_neg = kz
            .iter()
            .map(|&v| Complex64::from_polar(1.0, -v * dr / 2.0))
            .collect();

        let kx: Array1<f64> = Array1::from_iter((0..nx).map(|i| {
            let ii = i as i64;
            let n = nx as i64;
            if ii <= n / 2 {
                ii as f64 * dk_x
            } else {
                (ii - n) as f64 * dk_x
            }
        }));

        let kappa_2d = Array2::from_shape_fn((nx, nr_exp), |(i, k)| {
            let k2d = kx[i].hypot(kz[k]);
            let arg = c_ref * k2d * dt / 2.0;
            if arg.abs() < 1e-12 {
                1.0
            } else {
                arg.sin() / arg
            }
        });

        let fft_plan = FFT_CACHE_2D.get_or_create(Shape2D { nx, ny: nr_exp });
        let ze = || Array2::<f64>::zeros((nx, nr_exp));
        let zn = || Array2::<f64>::zeros((nx, nr));
        let zc = || Array2::<Complex64>::zeros((nx, nr_exp));

        Ok(Self {
            nx,
            nr,
            nr_exp,
            r_sg,
            ddy_k_shift_pos,
            ddy_k,
            y_shift_neg,
            kappa_2d,
            ddx_k_shift_pos,
            ddx_k_shift_neg,
            fft_plan,
            p_exp: ze(),
            ux_exp: ze(),
            uz_exp: ze(),
            uz_on_r_exp: ze(),
            p_2d: zn(),
            ux_2d: zn(),
            uz_2d: zn(),
            uz_on_r: zn(),
            ak: zc(),
            g: zc(),
            dpdx: zn(),
            dpdr: zn(),
            duxdx: zn(),
            duzdr: zn(),
            coef: zn(),
        })
    }

    // ---- Zero-allocation hot-path compute methods -------------------------

    /// Compute pressure gradients dpdx and dpdr from p (shape (nx, nr)).
    ///
    /// No heap allocation: all work uses pre-allocated scratch fields.
    ///
    /// Algorithm:
    ///  1. WS-expand p into p_exp.
    ///  2. ak = kappa_2d * FFT2(p_exp).
    ///  3. `g = ddx_k_shift_pos[row]*ak`; IFFT g in-place; `dpdx = Re(g)[..,0..nr]`.
    ///  4. `g = ddy_k_shift_pos[col]*ak`; IFFT g in-place; `dpdr = Re(g)[..,0..nr]`.
    ///
    /// No extra 1/N factor: apollo-fft inverse_complex_inplace uses FFTW-compatible
    /// 1/N normalisation, so IFFT(FFT(x)) = x without additional scaling.
    pub fn compute_vel_grads(&mut self, p: ArrayView2<'_, f64>) {
        let nr = self.nr;
        let nx = self.nx;
        let nr_exp = self.nr_exp;
        let plan = Arc::clone(&self.fft_plan);

        self.p_2d.assign(&p);
        Self::ws_expand(&self.p_2d, &mut self.p_exp, nr);
        plan.forward_into(&self.p_exp, &mut self.ak);
        Zip::from(&mut self.ak)
            .and(&self.kappa_2d)
            .par_for_each(|c, &k| *c *= k);

        // grad_x
        for i in 0..nx {
            let op = self.ddx_k_shift_pos[i];
            Zip::from(self.g.slice_mut(s![i, ..]))
                .and(self.ak.slice(s![i, ..]))
                .for_each(|o, &v| *o = op * v);
        }
        plan.inverse_complex_inplace(&mut self.g);
        Zip::from(&mut self.dpdx)
            .and(self.g.slice(s![.., 0..nr]))
            .par_for_each(|o, v| *o = v.re);

        // grad_r
        for k in 0..nr_exp {
            let op = self.ddy_k_shift_pos[k];
            Zip::from(self.g.slice_mut(s![.., k]))
                .and(self.ak.slice(s![.., k]))
                .for_each(|o, &v| *o = op * v);
        }
        plan.inverse_complex_inplace(&mut self.g);
        Zip::from(&mut self.dpdr)
            .and(self.g.slice(s![.., 0..nr]))
            .par_for_each(|o, v| *o = v.re);
    }

    /// Compute divergences duxdx and duzdr from ux and uz (shape (nx, nr)).
    ///
    /// No heap allocation.
    ///
    /// Algorithm:
    ///  1. WS-expand ux; duxdx = Re(IFFT2(kappa*ddx_shift_neg*FFT2(ux_exp)))[..,0..nr].
    ///  2. HAHS-expand uz; HSHA-expand uz/r.
    ///  3. ak = FFT2(uz_exp); g = FFT2(uz_on_r_exp).
    ///  4. ak = (ddy_k*ak + g)*y_shift_neg*kappa; IFFT ak; duzdr = Re(ak)[..,0..nr].
    ///
    /// No extra 1/N factor: apollo-fft inverse_complex_inplace uses FFTW-compatible
    /// 1/N normalisation, so IFFT(FFT(x)) = x without additional scaling.
    pub fn compute_density_divs(&mut self, ux: ArrayView2<'_, f64>, uz: ArrayView2<'_, f64>) {
        let nr = self.nr;
        let nx = self.nx;
        let nr_exp = self.nr_exp;
        let plan = Arc::clone(&self.fft_plan);

        self.ux_2d.assign(&ux);
        self.uz_2d.assign(&uz);

        // uz_on_r = uz / r_sg (staggered radial positions)
        Zip::indexed(&mut self.uz_on_r)
            .and(&self.uz_2d)
            .par_for_each(|(_i, k), o, &v| *o = v / self.r_sg[k]);

        // div_x
        Self::ws_expand(&self.ux_2d, &mut self.ux_exp, nr);
        plan.forward_into(&self.ux_exp, &mut self.ak);
        Zip::from(&mut self.ak)
            .and(&self.kappa_2d)
            .par_for_each(|c, &k| *c *= k);
        for i in 0..nx {
            let op = self.ddx_k_shift_neg[i];
            Zip::from(self.g.slice_mut(s![i, ..]))
                .and(self.ak.slice(s![i, ..]))
                .for_each(|o, &v| *o = op * v);
        }
        plan.inverse_complex_inplace(&mut self.g);
        Zip::from(&mut self.duxdx)
            .and(self.g.slice(s![.., 0..nr]))
            .par_for_each(|o, v| *o = v.re);

        // div_r_cylindrical
        Self::hahs_expand(&self.uz_2d, &mut self.uz_exp, nr);
        Self::hsha_expand(&self.uz_on_r, &mut self.uz_on_r_exp, nr);
        plan.forward_into(&self.uz_exp, &mut self.ak);
        plan.forward_into(&self.uz_on_r_exp, &mut self.g);
        for k in 0..nr_exp {
            let op_d = self.ddy_k[k];
            let op_s = self.y_shift_neg[k];
            Zip::from(self.ak.slice_mut(s![.., k]))
                .and(self.g.slice(s![.., k]))
                .and(self.kappa_2d.slice(s![.., k]))
                .for_each(|a, &gv, &kap| {
                    *a = (op_d * *a + gv) * op_s * kap;
                });
        }
        plan.inverse_complex_inplace(&mut self.ak);
        Zip::from(&mut self.duzdr)
            .and(self.ak.slice(s![.., 0..nr]))
            .par_for_each(|o, v| *o = v.re);
    }

    // ---- Domain expansion -- associated functions ------------------------

    /// WS (whole-sample symmetric) expansion: a (nx,nr) into out (nx,4*nr).
    pub fn ws_expand(a: &Array2<f64>, out: &mut Array2<f64>, nr: usize) {
        out.fill(0.0);
        out.slice_mut(s![.., 0..nr]).assign(a);
        for k in 0..nr - 1 {
            let src = nr - 1 - k;
            let dst = nr + 1 + k;
            Zip::from(out.slice_mut(s![.., dst]))
                .and(a.slice(s![.., src]))
                .for_each(|o, &v| *o = -v);
        }
        Zip::from(out.slice_mut(s![.., 2 * nr..3 * nr]))
            .and(a)
            .for_each(|o, &v| *o = -v);
        for k in 0..nr - 1 {
            let src = nr - 1 - k;
            let dst = 3 * nr + 1 + k;
            Zip::from(out.slice_mut(s![.., dst]))
                .and(a.slice(s![.., src]))
                .for_each(|o, &v| *o = v);
        }
    }

    /// HAHS expansion (radial velocity): a (nx,nr) into out (nx,4*nr).
    pub fn hahs_expand(a: &Array2<f64>, out: &mut Array2<f64>, nr: usize) {
        out.fill(0.0);
        out.slice_mut(s![.., 0..nr]).assign(a);
        for k in 0..nr {
            let src = nr - 1 - k;
            Zip::from(out.slice_mut(s![.., nr + k]))
                .and(a.slice(s![.., src]))
                .for_each(|o, &v| *o = v);
        }
        Zip::from(out.slice_mut(s![.., 2 * nr..3 * nr]))
            .and(a)
            .for_each(|o, &v| *o = -v);
        for k in 0..nr {
            let src = nr - 1 - k;
            Zip::from(out.slice_mut(s![.., 3 * nr + k]))
                .and(a.slice(s![.., src]))
                .for_each(|o, &v| *o = -v);
        }
    }

    /// HSHA expansion (ur/r term): a (nx,nr) into out (nx,4*nr).
    pub fn hsha_expand(a: &Array2<f64>, out: &mut Array2<f64>, nr: usize) {
        out.fill(0.0);
        out.slice_mut(s![.., 0..nr]).assign(a);
        for k in 0..nr {
            let src = nr - 1 - k;
            Zip::from(out.slice_mut(s![.., nr + k]))
                .and(a.slice(s![.., src]))
                .for_each(|o, &v| *o = -v);
        }
        Zip::from(out.slice_mut(s![.., 2 * nr..3 * nr]))
            .and(a)
            .for_each(|o, &v| *o = -v);
        for k in 0..nr {
            let src = nr - 1 - k;
            Zip::from(out.slice_mut(s![.., 3 * nr + k]))
                .and(a.slice(s![.., src]))
                .for_each(|o, &v| *o = v);
        }
    }

    #[inline]
    pub fn expand_ws(&self, a: &Array2<f64>, out: &mut Array2<f64>) {
        Self::ws_expand(a, out, self.nr);
    }

    #[inline]
    pub fn expand_hahs(&self, a: &Array2<f64>, out: &mut Array2<f64>) {
        Self::hahs_expand(a, out, self.nr);
    }

    #[inline]
    pub fn expand_hsha(&self, a: &Array2<f64>, out: &mut Array2<f64>) {
        Self::hsha_expand(a, out, self.nr);
    }
}
