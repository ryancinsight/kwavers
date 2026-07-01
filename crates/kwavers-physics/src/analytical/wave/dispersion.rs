/// Relative phase-velocity error of the 1-D FDTD scheme (Chapter 2 §2.4,
/// Theorem 2.3).
///
/// The staggered-grid leap-frog FDTD modified wavenumber (`kh = k·Δx`,
/// `CFL = c₀Δt/Δx`) is
/// ```text
/// k'h = 2 · arcsin(CFL · sin(kh / 2)) / CFL
/// ```
/// so the relative error returned, `(k' − k) / k = arcsin(CFL·sin(kh/2))/(CFL·kh/2) − 1`,
/// equals the relative phase-velocity error `(c̃ − c₀)/c₀` of Theorem 2.3. It
/// vanishes at `CFL = 1` (the 1-D magic time step) and grows as `kh → π`.
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

/// Centered finite-difference first-derivative modified wavenumber.
///
/// Returns `k* dx` for the second-, fourth-, or sixth-order centered stencil.
///
/// # Errors
/// Returns an error when `order` is not one of `2`, `4`, or `6`, or when any
/// sampled `kh` value is non-finite.
pub fn centered_fd_modified_wavenumber(kh_arr: &[f64], order: u32) -> Result<Vec<f64>, String> {
    if !matches!(order, 2 | 4 | 6) {
        return Err("centered finite-difference order must be 2, 4, or 6".to_owned());
    }
    if !kh_arr.iter().all(|value| value.is_finite()) {
        return Err("modified-wavenumber samples must be finite".to_owned());
    }
    Ok(kh_arr
        .iter()
        .map(|&kh| match order {
            2 => kh.sin(),
            4 => (8.0 * kh.sin() - (2.0 * kh).sin()) / 6.0,
            6 => (45.0 * kh.sin() - 9.0 * (2.0 * kh).sin() + (3.0 * kh).sin()) / 30.0,
            _ => unreachable!("invariant: order was validated"),
        })
        .collect())
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

/// Temporal sinc correction factor used by k-space PSTD updates.
///
/// Returns `sinc(CFL * kh / 2)`, where `sinc(x) = sin(x) / x`.
///
/// # Errors
/// Returns an error when `cfl` or any sampled `kh` value is non-finite.
pub fn kspace_temporal_correction(kh_arr: &[f64], cfl: f64) -> Result<Vec<f64>, String> {
    if !cfl.is_finite() {
        return Err("k-space temporal correction CFL must be finite".to_owned());
    }
    if !kh_arr.iter().all(|value| value.is_finite()) {
        return Err("k-space temporal correction samples must be finite".to_owned());
    }
    Ok(kh_arr.iter().map(|&kh| sinc(0.5 * cfl * kh)).collect())
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

/// 2-D explicit acoustic FDTD CFL stability region over component Courant axes.
///
/// The returned vector is row-major over `(cfl_x, cfl_z)` and contains `1.0`
/// for stable samples and `0.0` for unstable samples.
///
/// # Errors
/// Returns an error when either axis contains a non-finite value.
pub fn fdtd_cfl_stability_region_2d(cfl_x: &[f64], cfl_z: &[f64]) -> Result<Vec<f64>, String> {
    if !cfl_x.iter().all(|value| value.is_finite()) || !cfl_z.iter().all(|value| value.is_finite())
    {
        return Err("FDTD CFL axes must be finite".to_owned());
    }
    Ok(cfl_x
        .iter()
        .flat_map(|&x| {
            cfl_z
                .iter()
                .map(move |&z| if x.mul_add(x, z * z) <= 1.0 { 1.0 } else { 0.0 })
        })
        .collect())
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1.0e-12 {
        1.0
    } else {
        x.sin() / x
    }
}
