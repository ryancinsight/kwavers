//! Axisymmetric (CylindricalAS) PSTD propagator -- WSWA-FFT radial operators.
//!
//! All per-time-step intermediate arrays are pre-allocated in AsContext::new
//! and reused on every step, eliminating heap allocation on the hot path.
//! Apollo owns the 2-D FFT plan cache for the real and complex transforms.
//!
//! References:
//!   k-Wave MATLAB source kspaceFirstOrderAS.m, WSWA-FFT case.
//!   Treeby et al. (2012). k-Wave axisymmetric documentation.

use apollo::{fft_2d_array_into, ifft_2d_complex_inplace, Complex64 as ApolloComplex64};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array2 as LetoArray2;
use moirai_parallel::{enumerate_mut_with, Adaptive};
use ndarray::{s, Array1, Array2, ArrayView2};
use kwavers_math::fft::Complex64;

#[inline]
fn dense_indices(index: usize, ncols: usize) -> (usize, usize) {
    (index / ncols, index % ncols)
}

fn fft_forward_into_nd(field: &Array2<f64>, out: &mut Array2<Complex64>) {
    let field_leto = LetoArray2::from_shape_vec(
        [field.nrows(), field.ncols()],
        field.iter().copied().collect(),
    )
    .expect("axisymmetric real field length must match shape");
    let mut out_leto = to_apollo_array2(out);
    fft_2d_array_into(&field_leto, &mut out_leto);
    for (dst, src) in out.iter_mut().zip(out_leto.iter()) {
        *dst = Complex64::new(src.re, src.im);
    }
}

fn ifft_2d_complex_inplace_nd(data: &mut Array2<Complex64>) {
    let mut leto_data = to_apollo_array2(data);
    ifft_2d_complex_inplace(&mut leto_data);
    for (dst, src) in data.iter_mut().zip(leto_data.iter()) {
        *dst = Complex64::new(src.re, src.im);
    }
}

fn to_apollo_array2(input: &Array2<Complex64>) -> LetoArray2<ApolloComplex64> {
    LetoArray2::from_shape_vec(
        [input.nrows(), input.ncols()],
        input
            .iter()
            .map(|value| ApolloComplex64::new(value.re, value.im))
            .collect(),
    )
    .expect("axisymmetric complex field length must match shape")
}

fn multiply_by_real(field: &mut Array2<Complex64>, factors: &Array2<f64>) {
    assert_eq!(
        field.shape(),
        factors.shape(),
        "invariant: AS spectral field shape matches real multiplier"
    );

    if let (Some(field_values), Some(factor_values)) = (
        field.as_slice_memory_order_mut(),
        factors.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(field_values, |index, value| {
            *value *= factor_values[index];
        });
        return;
    }

    let (nx, nr) = field.dim();
    for k in 0..nr {
        for i in 0..nx {
            field[[i, k]] *= factors[[i, k]];
        }
    }
}

fn fill_row_operator(
    output: &mut Array2<Complex64>,
    input: &Array2<Complex64>,
    operators: &Array1<Complex64>,
) {
    assert_eq!(
        output.shape(),
        input.shape(),
        "invariant: AS spectral operator output shape matches input"
    );

    let (_nx, nr) = output.dim();
    if let (Some(output_values), Some(input_values), Some(operator_values)) = (
        output.as_slice_memory_order_mut(),
        input.as_slice_memory_order(),
        operators.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(output_values, |index, output| {
            let (i, _k) = dense_indices(index, nr);
            *output = operator_values[i] * input_values[index];
        });
        return;
    }

    let (nx, nr) = output.dim();
    for k in 0..nr {
        for i in 0..nx {
            output[[i, k]] = operators[i] * input[[i, k]];
        }
    }
}

fn fill_column_operator(
    output: &mut Array2<Complex64>,
    input: &Array2<Complex64>,
    operators: &Array1<Complex64>,
) {
    assert_eq!(
        output.shape(),
        input.shape(),
        "invariant: AS spectral operator output shape matches input"
    );

    let (_nx, nr) = output.dim();
    if let (Some(output_values), Some(input_values), Some(operator_values)) = (
        output.as_slice_memory_order_mut(),
        input.as_slice_memory_order(),
        operators.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(output_values, |index, output| {
            let (_i, k) = dense_indices(index, nr);
            *output = operator_values[k] * input_values[index];
        });
        return;
    }

    let (nx, nr) = output.dim();
    for k in 0..nr {
        for i in 0..nx {
            output[[i, k]] = operators[k] * input[[i, k]];
        }
    }
}

fn copy_real_window(output: &mut Array2<f64>, input: &Array2<Complex64>, cols: usize) {
    assert_eq!(
        output.nrows(),
        input.nrows(),
        "invariant: AS real output row count matches spectral input"
    );
    assert_eq!(
        output.ncols(),
        cols,
        "invariant: AS real output column count matches selected spectral window"
    );
    assert!(
        cols <= input.ncols(),
        "invariant: AS real output window is inside spectral input"
    );

    if let Some(output_values) = output.as_slice_memory_order_mut() {
        enumerate_mut_with::<Adaptive, _, _>(output_values, |index, output| {
            let (i, k) = dense_indices(index, cols);
            *output = input[[i, k]].re;
        });
        return;
    }

    let nx = output.nrows();
    for k in 0..cols {
        for i in 0..nx {
            output[[i, k]] = input[[i, k]].re;
        }
    }
}

fn divide_by_radial_position(
    output: &mut Array2<f64>,
    velocity: &Array2<f64>,
    radial_positions: &Array1<f64>,
) {
    assert_eq!(
        output.shape(),
        velocity.shape(),
        "invariant: AS radial velocity quotient shape matches velocity"
    );

    let (_nx, nr) = output.dim();
    if let (Some(output_values), Some(velocity_values), Some(radial_values)) = (
        output.as_slice_memory_order_mut(),
        velocity.as_slice_memory_order(),
        radial_positions.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(output_values, |index, output| {
            let (_i, k) = dense_indices(index, nr);
            *output = velocity_values[index] / radial_values[k];
        });
        return;
    }

    let (nx, nr) = output.dim();
    for k in 0..nr {
        for i in 0..nx {
            output[[i, k]] = velocity[[i, k]] / radial_positions[k];
        }
    }
}

fn apply_radial_divergence_operator(
    radial_spectrum: &mut Array2<Complex64>,
    quotient_spectrum: &Array2<Complex64>,
    derivative_operator: &Array1<Complex64>,
    shift_operator: &Array1<Complex64>,
    kappa: &Array2<f64>,
) {
    assert_eq!(
        radial_spectrum.shape(),
        quotient_spectrum.shape(),
        "invariant: AS radial spectrum shape matches quotient spectrum"
    );
    assert_eq!(
        radial_spectrum.shape(),
        kappa.shape(),
        "invariant: AS radial spectrum shape matches kappa"
    );

    let (_nx, nr) = radial_spectrum.dim();
    if let (
        Some(radial_values),
        Some(quotient_values),
        Some(kappa_values),
        Some(derivative_values),
        Some(shift_values),
    ) = (
        radial_spectrum.as_slice_memory_order_mut(),
        quotient_spectrum.as_slice_memory_order(),
        kappa.as_slice_memory_order(),
        derivative_operator.as_slice(),
        shift_operator.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(radial_values, |index, radial| {
            let (_i, k) = dense_indices(index, nr);
            *radial = (derivative_values[k] * *radial + quotient_values[index])
                * shift_values[k]
                * kappa_values[index];
        });
        return;
    }

    let (nx, nr) = radial_spectrum.dim();
    for k in 0..nr {
        let derivative = derivative_operator[k];
        let shift = shift_operator[k];
        for i in 0..nx {
            radial_spectrum[[i, k]] = (derivative * radial_spectrum[[i, k]]
                + quotient_spectrum[[i, k]])
                * shift
                * kappa[[i, k]];
        }
    }
}

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
    #[allow(clippy::too_many_arguments)]
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
        let dk_z = TWO_PI / (nr_exp as f64 * dr);
        let dk_x = TWO_PI / (nx as f64 * dx);

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

        let ze = || Array2::<f64>::zeros((nx, nr_exp));
        let zn = || Array2::<f64>::zeros((nx, nr));
        let zc = || Array2::<Complex64>::from_elem((nx, nr_exp), Complex64::default());

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
    /// No extra 1/N factor: the Kwavers FFT facade preserves Apollo's FFTW-compatible
    /// 1/N normalisation, so IFFT(FFT(x)) = x without additional scaling.
    pub fn compute_vel_grads(&mut self, p: ArrayView2<'_, f64>) {
        let nr = self.nr;
        self.p_2d.assign(&p);
        Self::ws_expand(&self.p_2d, &mut self.p_exp, nr);
        fft_forward_into_nd(&self.p_exp, &mut self.ak);
        multiply_by_real(&mut self.ak, &self.kappa_2d);

        // grad_x
        fill_row_operator(&mut self.g, &self.ak, &self.ddx_k_shift_pos);
        ifft_2d_complex_inplace_nd(&mut self.g);
        copy_real_window(&mut self.dpdx, &self.g, nr);

        // grad_r
        fill_column_operator(&mut self.g, &self.ak, &self.ddy_k_shift_pos);
        ifft_2d_complex_inplace_nd(&mut self.g);
        copy_real_window(&mut self.dpdr, &self.g, nr);
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
    /// No extra 1/N factor: the Kwavers FFT facade preserves Apollo's FFTW-compatible
    /// 1/N normalisation, so IFFT(FFT(x)) = x without additional scaling.
    pub fn compute_density_divs(&mut self, ux: ArrayView2<'_, f64>, uz: ArrayView2<'_, f64>) {
        let nr = self.nr;
        self.ux_2d.assign(&ux);
        self.uz_2d.assign(&uz);

        // uz_on_r = uz / r_sg (staggered radial positions)
        divide_by_radial_position(&mut self.uz_on_r, &self.uz_2d, &self.r_sg);

        // div_x
        Self::ws_expand(&self.ux_2d, &mut self.ux_exp, nr);
        fft_forward_into_nd(&self.ux_exp, &mut self.ak);
        multiply_by_real(&mut self.ak, &self.kappa_2d);
        fill_row_operator(&mut self.g, &self.ak, &self.ddx_k_shift_neg);
        ifft_2d_complex_inplace_nd(&mut self.g);
        copy_real_window(&mut self.duxdx, &self.g, nr);

        // div_r_cylindrical
        Self::hahs_expand(&self.uz_2d, &mut self.uz_exp, nr);
        Self::hsha_expand(&self.uz_on_r, &mut self.uz_on_r_exp, nr);
        fft_forward_into_nd(&self.uz_exp, &mut self.ak);
        fft_forward_into_nd(&self.uz_on_r_exp, &mut self.g);
        apply_radial_divergence_operator(
            &mut self.ak,
            &self.g,
            &self.ddy_k,
            &self.y_shift_neg,
            &self.kappa_2d,
        );
        ifft_2d_complex_inplace_nd(&mut self.ak);
        copy_real_window(&mut self.duzdr, &self.ak, nr);
    }

    // ---- Domain expansion -- associated functions ------------------------

    /// WS (whole-sample symmetric) expansion: a (nx,nr) into out (nx,4*nr).
    pub fn ws_expand(a: &Array2<f64>, out: &mut Array2<f64>, nr: usize) {
        out.fill(0.0);
        out.slice_mut(s![.., 0..nr]).assign(a);
        for k in 0..nr - 1 {
            let src = nr - 1 - k;
            let dst = nr + 1 + k;
            for i in 0..out.nrows() {
                out[[i, dst]] = -a[[i, src]];
            }
        }
        for k in 0..nr {
            for i in 0..out.nrows() {
                out[[i, 2 * nr + k]] = -a[[i, k]];
            }
        }
        for k in 0..nr - 1 {
            let src = nr - 1 - k;
            let dst = 3 * nr + 1 + k;
            for i in 0..out.nrows() {
                out[[i, dst]] = a[[i, src]];
            }
        }
    }

    /// HAHS expansion (radial velocity): a (nx,nr) into out (nx,4*nr).
    pub fn hahs_expand(a: &Array2<f64>, out: &mut Array2<f64>, nr: usize) {
        out.fill(0.0);
        out.slice_mut(s![.., 0..nr]).assign(a);
        for k in 0..nr {
            let src = nr - 1 - k;
            for i in 0..out.nrows() {
                out[[i, nr + k]] = a[[i, src]];
            }
        }
        for k in 0..nr {
            for i in 0..out.nrows() {
                out[[i, 2 * nr + k]] = -a[[i, k]];
            }
        }
        for k in 0..nr {
            let src = nr - 1 - k;
            for i in 0..out.nrows() {
                out[[i, 3 * nr + k]] = -a[[i, src]];
            }
        }
    }

    /// HSHA expansion (ur/r term): a (nx,nr) into out (nx,4*nr).
    pub fn hsha_expand(a: &Array2<f64>, out: &mut Array2<f64>, nr: usize) {
        out.fill(0.0);
        out.slice_mut(s![.., 0..nr]).assign(a);
        for k in 0..nr {
            let src = nr - 1 - k;
            for i in 0..out.nrows() {
                out[[i, nr + k]] = -a[[i, src]];
            }
        }
        for k in 0..nr {
            for i in 0..out.nrows() {
                out[[i, 2 * nr + k]] = -a[[i, k]];
            }
        }
        for k in 0..nr {
            let src = nr - 1 - k;
            for i in 0..out.nrows() {
                out[[i, 3 * nr + k]] = a[[i, src]];
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn context(nx: usize, nr: usize) -> AsContext {
        let zero_operator = Array1::from_elem(nx, Complex64::new(0.0, 0.0));
        AsContext::new(
            nx,
            nr,
            1.0e-3,
            1.0e-3,
            1_500.0,
            1.0e-7,
            zero_operator.clone(),
            zero_operator,
        )
        .expect("valid axisymmetric context")
    }

    #[test]
    fn axisymmetric_apollo_fft_path_preserves_zero_gradients() {
        let mut ctx = context(4, 3);
        let pressure = Array2::<f64>::zeros((ctx.nx, ctx.nr));

        ctx.compute_vel_grads(pressure.view());

        assert_eq!(ctx.dpdx, Array2::<f64>::zeros((ctx.nx, ctx.nr)));
        assert_eq!(ctx.dpdr, Array2::<f64>::zeros((ctx.nx, ctx.nr)));
    }

    #[test]
    fn axisymmetric_apollo_fft_path_preserves_zero_divergences() {
        let mut ctx = context(4, 3);
        let velocity = Array2::<f64>::zeros((ctx.nx, ctx.nr));

        ctx.compute_density_divs(velocity.view(), velocity.view());

        assert_eq!(ctx.duxdx, Array2::<f64>::zeros((ctx.nx, ctx.nr)));
        assert_eq!(ctx.duzdr, Array2::<f64>::zeros((ctx.nx, ctx.nr)));
    }
}
