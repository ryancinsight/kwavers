//! One-dimensional CPML profile kernels.
//!
//! The kernels are axis-agnostic. Callers provide the target arrays, axis
//! length, spacing, PML thickness, k-Wave `pml_alpha`, reference sound speed,
//! and time step. This keeps x/y/z variation data-driven and prevents cloned
//! axis-specific implementations.

use ndarray::Array1;

pub(super) struct CollocatedProfileMut<'a> {
    sigma: &'a mut Array1<f64>,
    kappa: &'a mut Array1<f64>,
    alpha: &'a mut Array1<f64>,
    a_coeff: &'a mut Array1<f64>,
    b_coeff: &'a mut Array1<f64>,
}

impl<'a> CollocatedProfileMut<'a> {
    pub(super) fn new(
        sigma: &'a mut Array1<f64>,
        kappa: &'a mut Array1<f64>,
        alpha: &'a mut Array1<f64>,
        a_coeff: &'a mut Array1<f64>,
        b_coeff: &'a mut Array1<f64>,
    ) -> Self {
        Self {
            sigma,
            kappa,
            alpha,
            a_coeff,
            b_coeff,
        }
    }
}

/// Exact k-Wave collocated PML profile.
///
/// For the left PML, index `i = 0` is the outer wall and uses
/// `q = (thickness - i) / thickness`. For the right PML, `q` increases from
/// `1 / thickness` at the physical-domain interface to `1` at the outer wall.
pub(super) fn compute_collocated_profile(
    profile: CollocatedProfileMut<'_>,
    n: usize,
    dx: f64,
    thickness: usize,
    pml_alpha: f64,
    sound_speed: f64,
    dt: f64,
) {
    if n <= 1 || thickness == 0 {
        set_neutral(profile);
        return;
    }

    let mut assign = |idx: usize, sigma_val: f64| {
        profile.sigma[idx] = sigma_val;
        profile.kappa[idx] = 1.0;
        profile.alpha[idx] = 0.0;
        let b = (-sigma_val * dt).exp();
        profile.b_coeff[idx] = b;
        profile.a_coeff[idx] = b - 1.0;
    };

    for i in 0..thickness.min(n) {
        let q = (thickness - i) as f64 / thickness as f64;
        assign(i, pml_alpha * (sound_speed / dx) * q.powi(4));
    }

    let right_start = n.saturating_sub(thickness);
    for i in right_start..n {
        let q = (i - right_start + 1) as f64 / thickness as f64;
        assign(i, pml_alpha * (sound_speed / dx) * q.powi(4));
    }
}

/// Exact k-Wave staggered PML profile for velocity components.
///
/// k-Wave shifts the profile by a half cell. The rightmost staggered cell can
/// exceed the collocated wall value because the half-cell point lies outside
/// the physical-domain sample centers; this is required for parity.
pub(super) fn compute_staggered_profile(
    sigma_sg: &mut Array1<f64>,
    n: usize,
    dx: f64,
    thickness: usize,
    pml_alpha: f64,
    sound_speed: f64,
) {
    if n <= 1 || thickness == 0 {
        sigma_sg.fill(0.0);
        return;
    }

    let t = thickness as f64;

    for i in 0..thickness.min(n) {
        let q = (t - i as f64 - 0.5) / t;
        sigma_sg[i] = pml_alpha * (sound_speed / dx) * q.abs().powi(4);
    }

    let right_start = n.saturating_sub(thickness);
    for i in right_start..n {
        let j = (i - right_start) as f64;
        let q = (j + 1.5) / t;
        sigma_sg[i] = pml_alpha * (sound_speed / dx) * q.powi(4);
    }
}

fn set_neutral(profile: CollocatedProfileMut<'_>) {
    profile.sigma.fill(0.0);
    profile.kappa.fill(1.0);
    profile.alpha.fill(0.0);
    profile.a_coeff.fill(0.0);
    profile.b_coeff.fill(1.0);
}
