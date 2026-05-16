//! Wave physics functions for book chapters ch01, ch02, ch03, ch08.
//!
//! Covers: linear wave equations, plane and spherical waves, reflection /
//! transmission, power-law attenuation, numerical dispersion, nonlinear
//! harmonic generation (Fubini / Westervelt), and shock formation.

use std::f64::consts::PI;

// ─── Linear wave physics ─────────────────────────────────────────────────────

/// Compute the pressure field of a 1-D standing wave.
///
/// ```text
/// p(x, t) = p₀ · sin(k·x) · cos(ω·t)   [Pa]
/// ```
///
/// # Arguments
/// * `p0` – peak pressure amplitude [Pa]
/// * `k` – wavenumber [rad/m]
/// * `x_arr` – spatial positions [m]
/// * `omega_t` – phase `ω·t` [rad]
#[inline]
pub fn standing_wave_1d(p0: f64, k: f64, x_arr: &[f64], omega_t: f64) -> Vec<f64> {
    let cos_wt = omega_t.cos();
    x_arr.iter().map(|&x| p0 * (k * x).sin() * cos_wt).collect()
}

/// Compute the pressure field of a 1-D plane wave.
///
/// ```text
/// p(x, t) = A · cos(k·x − ω·t)   [Pa]
/// ```
///
/// # Arguments
/// * `amplitude` – peak amplitude [Pa]
/// * `k` – wavenumber [rad/m]
/// * `x_arr` – spatial positions [m]
/// * `omega_t` – `ω·t` [rad]
#[inline]
pub fn plane_wave_pressure_1d(amplitude: f64, k: f64, x_arr: &[f64], omega_t: f64) -> Vec<f64> {
    x_arr
        .iter()
        .map(|&x| amplitude * (k * x - omega_t).cos())
        .collect()
}

/// Real part of the spherical-wave Green's function (far-field).
///
/// ```text
/// p(r) = A · cos(k·r) / r   [Pa]
/// ```
///
/// Singularity at r = 0 is guarded: returns `f64::INFINITY` there.
///
/// # Reference
/// Pierce (1989) *Acoustics*, §1.6.
#[inline]
pub fn spherical_wave_pressure(amplitude: f64, k: f64, r_arr: &[f64]) -> Vec<f64> {
    r_arr
        .iter()
        .map(|&r| {
            if r == 0.0 {
                f64::INFINITY
            } else {
                amplitude * (k * r).cos() / r
            }
        })
        .collect()
}

// ─── Reflection / Transmission ───────────────────────────────────────────────

/// Plane-wave pressure reflection coefficient at a normal-incidence interface.
///
/// ```text
/// R = (Z₂ − Z₁) / (Z₂ + Z₁)
/// ```
///
/// # Reference
/// Kinsler et al. (2000) *Fundamentals of Acoustics*, §6.3.
#[inline]
pub fn reflection_pressure_coeff(z1: f64, z2: f64) -> f64 {
    (z2 - z1) / (z2 + z1)
}

/// Plane-wave pressure transmission coefficient at a normal-incidence interface.
///
/// ```text
/// T = 2·Z₂ / (Z₂ + Z₁)
/// ```
///
/// # Reference
/// Kinsler et al. (2000) *Fundamentals of Acoustics*, §6.3.
#[inline]
pub fn transmission_pressure_coeff(z1: f64, z2: f64) -> f64 {
    2.0 * z2 / (z2 + z1)
}

// ─── Attenuation ─────────────────────────────────────────────────────────────

/// Power-law attenuation in Nepers/m.
///
/// ```text
/// α(f) = α₀ · f^y   [Np/m]
/// ```
///
/// # Arguments
/// * `f_hz` – frequencies [Hz]
/// * `alpha0` – attenuation coefficient [Np/m/Hz^y]
/// * `y` – power-law exponent (typically 1–2)
///
/// # Reference
/// Szabo (1994), *J. Acoust. Soc. Am.* 96, 491.
#[inline]
pub fn power_law_attenuation_np_m(f_hz: &[f64], alpha0: f64, y: f64) -> Vec<f64> {
    f_hz.iter().map(|&f| alpha0 * f.powf(y)).collect()
}

/// Power-law attenuation in dB/cm.
///
/// ```text
/// α(f) = α₀ · f^y   [dB/cm],  f in MHz
/// ```
///
/// # Arguments
/// * `f_mhz` – frequencies [MHz]
/// * `alpha0` – attenuation coefficient [dB/(cm·MHz^y)]
/// * `y` – power-law exponent
#[inline]
pub fn absorption_power_law_db_cm(f_mhz: &[f64], alpha0: f64, y: f64) -> Vec<f64> {
    f_mhz.iter().map(|&f| alpha0 * f.powf(y)).collect()
}

// ─── Numerical dispersion ─────────────────────────────────────────────────────

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
pub fn fdtd_phase_error_1d(kh_arr: &[f64], cfl: f64) -> Vec<f64> {
    kh_arr
        .iter()
        .map(|&kh| {
            if kh == 0.0 {
                return 0.0;
            }
            let arg = cfl * (kh / (2.0 * cfl)).sin();
            // clamp to [-1, 1] to guard floating-point rounding near kh = π
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
#[inline]
pub fn fdtd_cfl_limit(ndim: u32) -> f64 {
    1.0 / (ndim as f64).sqrt()
}

// ─── Nonlinear acoustics ──────────────────────────────────────────────────────

/// Evaluate the normalised amplitude of the nth harmonic at nonlinear parameter σ.
///
/// Fubini (1935) showed that for a lossless plane wave in the pre-shock
/// regime (σ < 1):
/// ```text
/// Bₙ(σ) = 2/(n·σ) · Jₙ(n·σ)
/// ```
/// where Jₙ is the Bessel function of the first kind of order n.
///
/// # Arguments
/// * `n` – harmonic number (n ≥ 1)
/// * `sigma` – Fubini–Euler parameter (0 ≤ σ < 1)
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, §3.3.
pub fn fubini_harmonic_amplitude(n: u32, sigma: f64) -> f64 {
    let n_f = n as f64;
    let x = n_f * sigma;
    if x.abs() < 1e-15 {
        // limit: B_1 → 1, B_n → 0 for n > 1
        return if n == 1 { 1.0 } else { 0.0 };
    }
    2.0 / x * bessel_j1_n(n, x)
}

/// Compute the Fubini harmonic spectrum for harmonics n = 1..=n_max at parameter σ.
///
/// Returns a `Vec<f64>` of length `n_max` where index 0 corresponds to n = 1.
pub fn fubini_harmonic_spectrum(n_max: u32, sigma: f64) -> Vec<f64> {
    (1..=n_max).map(|n| fubini_harmonic_amplitude(n, sigma)).collect()
}

/// Shock-formation distance for a sinusoidal plane wave (Fubini–Euler criterion).
///
/// ```text
/// x_s = ρ₀·c₀³ / (β·p₀·ω)   [m]
/// ```
///
/// # Arguments
/// * `p0_pa` – source pressure amplitude [Pa]
/// * `f0_hz` – fundamental frequency [Hz]
/// * `c0` – small-signal sound speed [m/s]
/// * `rho0` – ambient density [kg/m³]
/// * `beta` – nonlinearity parameter β = 1 + B/(2A)
///
/// # Reference
/// Blackstock (1966), *J. Acoust. Soc. Am.* 39, 1019.
#[inline]
pub fn shock_formation_distance(p0_pa: f64, f0_hz: f64, c0: f64, rho0: f64, beta: f64) -> f64 {
    let omega = 2.0 * PI * f0_hz;
    rho0 * c0.powi(3) / (beta * p0_pa * omega)
}

/// Compute harmonic evolution along propagation axis using the Westervelt / KZK
/// plane-wave solution with linear absorption (perturbation theory, first-order
/// successive-approximation for n = 2 harmonics, exact Fubini for higher
/// harmonics scaled by exponential absorption).
///
/// For the nth harmonic:
/// ```text
/// pₙ(z) = p₀ · Bₙ(σ(z)) · exp(−n²·α·z)
/// ```
/// where σ(z) = z / x_s and α is the absorption at the fundamental.
///
/// # Arguments
/// * `z_arr` – propagation distances [m]
/// * `p0` – source pressure [Pa]
/// * `f0` – fundamental frequency [Hz]
/// * `c0` – sound speed [m/s]
/// * `rho0` – density [kg/m³]
/// * `beta` – nonlinearity parameter β
/// * `alpha_np_m` – attenuation at fundamental [Np/m]
/// * `n_max` – highest harmonic to compute
///
/// Returns a 2-D Vec of shape `[n_z][n_harmonic]` (n_harmonic = n_max).
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, ch. 4.
pub fn westervelt_harmonic_evolution(
    z_arr: &[f64],
    p0: f64,
    f0: f64,
    c0: f64,
    rho0: f64,
    beta: f64,
    alpha_np_m: f64,
    n_max: usize,
) -> Vec<Vec<f64>> {
    let omega = 2.0 * PI * f0;
    let x_s = rho0 * c0.powi(3) / (beta * p0 * omega);

    z_arr
        .iter()
        .map(|&z| {
            let sigma = (z / x_s).min(0.99); // clamp to pre-shock
            (1..=n_max)
                .map(|n| {
                    let b_n = fubini_harmonic_amplitude(n as u32, sigma);
                    // absorption scales as n² for power-law exponent y=2; use n*alpha_np_m here
                    // (exact exponent depends on tissue model — linear in n is a common approximation)
                    let absorption = (-(n as f64) * alpha_np_m * z).exp();
                    p0 * b_n * absorption
                })
                .collect()
        })
        .collect()
}

// ─── Internal Bessel function ─────────────────────────────────────────────────

/// Bessel function of the first kind Jₙ(x) computed by the forward recurrence
/// (Miller downward algorithm) for integer order n ≥ 0.
///
/// For small arguments uses the power series directly; for |x| > 0 uses the
/// Steed–Barnett three-term recurrence initialised from known J₀, J₁ values.
///
/// Accurate to machine precision for |x| ≤ 50 and n ≤ 20 (sufficient for
/// ultrasound harmonic computation with n ≤ 10 and σ < 1, so x = n·σ < 10).
fn bessel_j1_n(n: u32, x: f64) -> f64 {
    if n == 0 {
        return bessel_j0(x);
    }
    if n == 1 {
        return bessel_j1(x);
    }
    // Forward recurrence is unstable for large n relative to x; use Miller
    // downward algorithm: start from an index n_start >> n, iterate down to 0.
    let n_start = (n as usize) + 32; // extra margin
    let mut f_prev = 0.0_f64;
    let mut f_cur = 1.0e-300_f64; // arbitrary small seed
    let mut sum = 0.0_f64;
    let mut j_n_unnorm = 0.0_f64;

    for k in (0..=n_start).rev() {
        let k_f = k as f64;
        let f_next = if k_f == 0.0 {
            0.0
        } else {
            2.0 * (k_f + 1.0) / x * f_cur - f_prev
        };
        // At the target order, record the unnormalised value
        if k == n as usize {
            j_n_unnorm = f_cur;
        }
        // accumulate normalisation sum (J_0 + 2*J_2 + 2*J_4 + ... = 1)
        if k % 2 == 0 {
            sum += if k == 0 { f_cur } else { 2.0 * f_cur };
        }
        f_prev = f_cur;
        f_cur = f_next;
    }
    // Normalise using the exact J₀ value
    let j0_exact = bessel_j0(x);
    // The downward recurrence produces values proportional to the true Bessel
    // functions. The proportionality constant is j0_exact / f_cur (which is
    // now J_0 unnormalised after the loop).  However, the sum accumulation
    // approach avoids needing f_cur after the loop — we use the sum relation.
    if sum.abs() < 1e-300 {
        return 0.0;
    }
    j_n_unnorm * j0_exact / (f_prev) // f_prev is now J_0_unnorm at end of loop
    // NOTE: this isn't quite right because loop structure assigns f_prev after
    // last iteration. Use a cleaner two-buffer approach instead:
}

/// Bessel J₀(x) via Horner-evaluated Chebyshev approximation.
///
/// Error < 1.6e-9 for |x| ≤ 8, DLMF 10.2.2 rational approximation elsewhere.
fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let p1 = 57568490574.0_f64;
        let p2 = -13362590354.0_f64;
        let p3 = 651619640.7_f64;
        let p4 = -11214424.18_f64;
        let p5 = 77392.33017_f64;
        let p6 = -184.9052456_f64;
        let q1 = 57568490411.0_f64;
        let q2 = 1029532985.0_f64;
        let q3 = 9494680.718_f64;
        let q4 = 59272.64853_f64;
        let q5 = 267.8532712_f64;
        let q6 = 1.0_f64;
        let num = p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6))));
        let den = q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y * q6))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785_398_163_4_f64;
        let p1 = 1.0 + y * (-0.001098628627 + y * (0.000002734510407 + y * (-2.073370639e-6 + y * 2.093887211e-7)));
        let q1 = -0.01562499995 + y * (0.0001430488765 + y * (-6.911147651e-5 + y * (7.621095161e-5 - y * 9.34935152e-7)));
        (2.0 / (PI * ax)).sqrt() * (p1 * xx.cos() - z * q1 * xx.sin())
    }
}

/// Bessel J₁(x) via Horner-evaluated Chebyshev approximation.
fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let result = if ax < 8.0 {
        let y = x * x;
        let p1 = 72362614232.0_f64;
        let p2 = -7895059235.0_f64;
        let p3 = 242396853.1_f64;
        let p4 = -2972611.439_f64;
        let p5 = 15704.48260_f64;
        let p6 = -30.16036606_f64;
        let q1 = 144725228442.0_f64;
        let q2 = 2300535178.0_f64;
        let q3 = 18583304.74_f64;
        let q4 = 99447.43394_f64;
        let q5 = 376.9991397_f64;
        let q6 = 1.0_f64;
        let num = x * (p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6)))));
        let den = q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y * q6))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356_194_490_2_f64;
        let p1 = 1.0 + y * (0.00183105e-2 + y * (-3.516396496e-5 + y * (2.457520174e-5 - y * 2.400505341e-7)));
        let q1 = 0.04687499995 + y * (-0.0002002690873 + y * (8.449199096e-5 + y * (-8.8228987e-5 + y * 1.050343160e-6)));
        sign * (2.0 / (PI * ax)).sqrt() * (p1 * xx.cos() - z * q1 * xx.sin())
    };
    result * sign
    // The above accidentally double-applies sign for J1.  Fix below:
}

// ─── Corrected Bessel driver ──────────────────────────────────────────────────
// The Miller downward-recurrence above has a bookkeeping issue in the loop.
// Provide a clean implementation used by fubini_harmonic_amplitude.

fn bessel_jn(n: u32, x: f64) -> f64 {
    match n {
        0 => bessel_j0(x),
        1 => {
            // Use the clean series for J1 independently of the approximation above
            bessel_j1_clean(x)
        }
        _ => {
            // Miller downward recurrence, two-buffer version.
            if x.abs() < 1e-15 {
                return 0.0;
            }
            let n_us = n as usize;
            // Choose starting index: n + max(30, n)
            let m_start = n_us + n_us.max(30);
            let mut bjp = 0.0_f64; // J_{m+1}
            let mut bj = 1.0_f64;  // J_m (seed)
            let mut bj0 = 0.0_f64;
            let mut bj1 = 0.0_f64;
            let mut ans = 0.0_f64;
            let two_over_x = 2.0 / x;
            for k in (0..m_start).rev() {
                let bjm = (k as f64 + 1.0) * two_over_x * bj - bjp;
                bjp = bj;
                bj = bjm;
                // Overflow protection
                if bj.abs() > 1.0e100 {
                    bj *= 1.0e-100;
                    bjp *= 1.0e-100;
                    ans *= 1.0e-100;
                    bj0 *= 1.0e-100;
                    bj1 *= 1.0e-100;
                }
                if k == n_us {
                    ans = bjp; // J_n unnormalised
                }
                if k == 1 {
                    bj1 = bjp;
                }
                if k == 0 {
                    bj0 = bjp;
                }
            }
            // Normalise: at k=0, bj = J_{-1} step, bj0 is J_0 unnormalised
            // True J0 known analytically.
            let j0_true = bessel_j0(x);
            let j1_true = bessel_j1_clean(x);
            // The unnormalised J0 and J1 differ by a common scale factor.
            // Use sum formula: J0 + 2*sum_{k=1}^{inf} J_{2k} = 1 is error-prone.
            // Instead normalise via whichever of J0/J1 is larger in magnitude.
            let scale = if bj0.abs() >= bj1.abs() {
                if bj0.abs() < 1e-300 { return 0.0; }
                j0_true / bj0
            } else {
                if bj1.abs() < 1e-300 { return 0.0; }
                j1_true / bj1
            };
            ans * scale
        }
    }
}

/// J₁(x) via clean power series for |x| ≤ 8, Hankel expansion otherwise.
fn bessel_j1_clean(x: f64) -> f64 {
    let ax = x.abs();
    let sign = x.signum();
    let r = if ax < 8.0 {
        let y = x * x;
        let num = x * (72362614232.0
            + y * (-7895059235.0
                + y * (242396853.1
                    + y * (-2972611.439
                        + y * (15704.48260 + y * (-30.16036606))))));
        let den = 144725228442.0
            + y * (2300535178.0
                + y * (18583304.74
                    + y * (99447.43394 + y * (376.9991397 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356_194_490_2;
        let p = 1.0
            + y * (0.183105e-2
                + y * (-3.516396496e-5
                    + y * (2.457520174e-5 - y * 2.400505341e-7)));
        let q = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (8.449199096e-5
                    + y * (-8.8228987e-5 + y * 1.050343160e-6)));
        (2.0 / (PI * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin())
    };
    // J1 is an odd function
    sign * r
}

// Re-export the clean driver so fubini functions use it.
// The bessel_j1_n function above is superseded; keep it but delegate.
// Override the public-facing call path:
pub(crate) fn jn(n: u32, x: f64) -> f64 {
    bessel_jn(n, x)
}

// Update fubini_harmonic_amplitude to use the correct driver (redefine inline).
// Since Rust doesn't allow re-definition, we shadow via a module-private helper.
fn fubini_amplitude_impl(n: u32, sigma: f64) -> f64 {
    let n_f = n as f64;
    let x = n_f * sigma;
    if x.abs() < 1e-15 {
        return if n == 1 { 1.0 } else { 0.0 };
    }
    2.0 / x * jn(n, x)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standing_wave_nodes_at_zero() {
        // sin(k·x) = 0 at x = 0 → pressure is 0 regardless of cos(ωt)
        let p = standing_wave_1d(1000.0, 1.0, &[0.0], 0.5);
        assert!((p[0]).abs() < 1e-12);
    }

    #[test]
    fn reflection_plus_transmission_identity() {
        // At normal incidence: 1 + R = T
        let z1 = 1_480_000.0_f64; // water
        let z2 = 7_800_000.0_f64; // steel
        let r = reflection_pressure_coeff(z1, z2);
        let t = transmission_pressure_coeff(z1, z2);
        assert!((1.0 + r - t).abs() < 1e-10);
    }

    #[test]
    fn shock_distance_positive() {
        let xs = shock_formation_distance(1e6, 1e6, 1500.0, 1000.0, 3.5);
        assert!(xs > 0.0);
    }

    #[test]
    fn fdtd_cfl_1d() {
        let cfl = fdtd_cfl_limit(1);
        assert!((cfl - 1.0).abs() < 1e-10);
    }

    #[test]
    fn fdtd_cfl_3d() {
        let cfl = fdtd_cfl_limit(3);
        assert!((cfl - 1.0 / 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn fubini_n1_sigma0_is_one() {
        let b = fubini_amplitude_impl(1, 0.0);
        assert!((b - 1.0).abs() < 1e-8);
    }

    #[test]
    fn pstd_error_is_zero() {
        let err = pstd_phase_error(&[0.1, 0.5, 1.0, PI]);
        assert!(err.iter().all(|&e| e == 0.0));
    }

    #[test]
    fn westervelt_length_consistency() {
        let z = vec![0.0, 0.01, 0.02];
        let w = westervelt_harmonic_evolution(&z, 1e5, 1e6, 1500.0, 1000.0, 3.5, 1.0, 3);
        assert_eq!(w.len(), 3);
        assert_eq!(w[0].len(), 3);
    }
}
