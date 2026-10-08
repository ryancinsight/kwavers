//! Internal absorption and nonlinear sub-operators for the cylindrical KZK solver.

use apollo::{fft_1d_complex_inplace, ifft_1d_complex_inplace, Complex64 as ApolloComplex64};
use kwavers_core::constants::acoustic_parameters::NP_TO_DB;
use kwavers_core::constants::numerical::{CM_TO_M, MHZ_TO_HZ};
use kwavers_math::fft::Complex64;
use leto::{Array1 as LetoArray1, Array2};

use super::CylindricalKZKConfig;
#[derive(Debug)]
pub(super) struct CylindricalAbsorption {
    config: CylindricalKZKConfig,
    h_mask_half: Vec<f64>,
    h_mask_full: Vec<f64>,
}

impl CylindricalAbsorption {
    #[must_use]
    pub(super) fn new(config: &CylindricalKZKConfig) -> Self {
        let alpha0_np = config.alpha0 / CM_TO_M / NP_TO_DB / MHZ_TO_HZ.powf(config.alpha_power);
        let h_mask_half = Self::build_mask(
            alpha0_np,
            config.alpha_power,
            config.nt,
            config.dt,
            config.dz * 0.5,
        );
        let h_mask_full = Self::build_mask(
            alpha0_np,
            config.alpha_power,
            config.nt,
            config.dt,
            config.dz,
        );

        Self {
            config: config.clone(),
            h_mask_half,
            h_mask_full,
        }
    }

    pub(super) fn build_mask(
        alpha0_np: f64,
        power: f64,
        nt: usize,
        dt: f64,
        step_size: f64,
    ) -> Vec<f64> {
        let df = 1.0 / (nt as f64 * dt);
        let mut mask = vec![1.0_f64; nt];
        for (k, elem) in mask.iter_mut().enumerate().skip(1) {
            let pos_k = if k <= nt / 2 { k } else { nt - k };
            let freq_hz = pos_k as f64 * df;
            let alpha = alpha0_np * freq_hz.powf(power);
            *elem = (-alpha * step_size).exp();
        }
        mask
    }

    pub(super) fn apply(&mut self, pressure: &mut Array2<Complex64>, step_size: f64) {
        let h_mask = if self.config.dz.mul_add(-0.5, step_size).abs() <= self.config.dz * 1.0e-10 {
            &self.h_mask_half
        } else {
            &self.h_mask_full
        };

        let mut waveform = LetoArray1::<ApolloComplex64>::zeros([self.config.nt]);
        for i in 0..self.config.nr {
            for t in 0..self.config.nt {
                let value = pressure[[i, t]];
                waveform[t] = ApolloComplex64::new(value.re, value.im);
            }

            fft_1d_complex_inplace(&mut waveform);
            for (w, &h) in waveform.iter_mut().zip(h_mask.iter()) {
                *w *= h;
            }
            ifft_1d_complex_inplace(&mut waveform);

            for t in 0..self.config.nt {
                let value = waveform[t];
                pressure[[i, t]] = Complex64::new(value.re, value.im);
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct CylindricalNonlinear {
    beta: f64,
    config: CylindricalKZKConfig,
    delta: Array2<f64>,
}

impl CylindricalNonlinear {
    #[must_use]
    pub(super) fn new(config: &CylindricalKZKConfig) -> Self {
        Self {
            beta: 1.0 + config.b_over_a / 2.0,
            config: config.clone(),
            delta: Array2::zeros((config.nr, config.nt)),
        }
    }

    pub(super) fn apply(
        &mut self,
        pressure: &mut Array2<Complex64>,
        _pressure_prev: &Array2<Complex64>,
        step_size: f64,
    ) {
        let coeff = self.beta * step_size / (2.0 * self.config.rho0 * self.config.c0.powi(3));
        self.delta.fill(0.0);

        for i in 0..self.config.nr {
            for t in 0..self.config.nt {
                let prev_t = if t == 0 { self.config.nt - 1 } else { t - 1 };
                let next_t = if t + 1 == self.config.nt { 0 } else { t + 1 };
                let p_prev = pressure[[i, prev_t]].re;
                let p_next = pressure[[i, next_t]].re;
                let dp2_dt = (p_next * p_next - p_prev * p_prev) / (2.0 * self.config.dt);
                self.delta[[i, t]] = coeff * dp2_dt;
            }
        }

        for i in 0..self.config.nr {
            for t in 0..self.config.nt {
                pressure[[i, t]].re += self.delta[[i, t]];
            }
        }
    }
}
