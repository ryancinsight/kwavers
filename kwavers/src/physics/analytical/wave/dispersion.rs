/// Relative phase-velocity error of the 1-D FDTD scheme.
///
/// The staggered-grid leap-frog FDTD modified wavenumber is:
/// ```text
/// k'h = 2 · arcsin(CFL · sin(kh / (2·CFL))) / CFL
///                               (where kh = k·Δx)
/// ```
/// Relative error returned: `(k' − k) / k`.
///
/// # Reference
/// Taflove & Hagness (2005) *Computational Electrodynamics*, §4.5.
#[must_use]
pub fn fdtd_phase_error_1d(kh_arr: &[f64], cfl: f64) -> Vec<f64> {
    kh_arr
        .iter()
        .map(|&kh| {
            if kh == 0.0 {
                return 0.0;
            }
            let arg = cfl * (kh / 2.0).sin();
            let arg_clamped = arg.clamp(-1.0, 1.0);
            let kp_h = 2.0 * arg_clamped.asin() / cfl;
            (kp_h - kh) / kh
        })
        .collect()
}

/// Relative phase-velocity error of the PSTD / k-space scheme.
///
/// PSTD is spectrally exact up to the Nyquist limit; returns zeros for all
/// spatial frequencies within that range. Values outside the Nyquist limit
/// (kh > π) are undefined and also returned as 0.
///
/// # Reference
/// Liu (1997), *J. Comput. Phys.* 131, 306.
#[must_use]
#[inline]
pub fn pstd_phase_error(kh_arr: &[f64]) -> Vec<f64> {
    vec![0.0; kh_arr.len()]
}

/// Relative temporal dispersion error of the k-space correction.
///
/// The k-space method applies a temporal sinc correction; the remaining
/// relative error compared with the exact continuous dispersion relation is:
/// ```text
/// ε(kh) = sinc(CFL·kh/2) / sinc(kh/2) − 1
/// ```
/// where sinc(x) = sin(x)/x (normalized sinc is NOT used here).
///
/// # Reference
/// Tabei et al. (2002), *J. Acoust. Soc. Am.* 111, 53.
#[must_use]
pub fn kspace_correction_error(kh_arr: &[f64], cfl: f64) -> Vec<f64> {
    let sinc = |x: f64| -> f64 {
        if x.abs() < 1e-12 {
            1.0
        } else {
            x.sin() / x
        }
    };
    kh_arr
        .iter()
        .map(|&kh| {
            let numerator = sinc(cfl * kh / 2.0);
            let denominator = sinc(kh / 2.0);
            if denominator.abs() < 1e-15 {
                0.0
            } else {
                numerator / denominator - 1.0
            }
        })
        .collect()
}

/// CFL stability limit for the explicit FDTD scheme in `ndim` spatial dimensions.
///
/// ```text
/// CFL_max = 1 / √(ndim)
/// ```
///
/// # Reference
/// Courant, Friedrichs & Lewy (1928).
#[must_use]
#[inline]
pub fn fdtd_cfl_limit(ndim: u32) -> f64 {
    1.0 / (ndim as f64).sqrt()
}
