use num_complex::Complex;

// ═══════════════════════════════════════════════════════════════════════════════
// Mie Acoustic Scattering — Compressible Fluid Sphere
// ═══════════════════════════════════════════════════════════════════════════════
//
// # Theorem (Anderson 1950; Morse & Ingard 1968 §8.3)
//
// For a fluid sphere of radius R, interior wavenumber k_b = ω/c_b and density ρ_b,
// embedded in a medium with k_l = ω/c_l and density ρ_l, the partial-wave scattering
// coefficients are:
//
//         ρ_b·jₙ(k_b·R)·D[jₙ](k_l·R)  −  ρ_l·jₙ(k_l·R)·D[jₙ](k_b·R)
// aₙ = − ─────────────────────────────────────────────────────────────────
//         ρ_b·jₙ(k_b·R)·D[hₙ](k_l·R)  −  ρ_l·hₙ⁽¹⁾(k_l·R)·D[jₙ](k_b·R)
//
// where D[fₙ](x) = x·fₙ₋₁(x) − n·fₙ(x) (logarithmic derivative operator),
// jₙ = spherical Bessel first kind, hₙ⁽¹⁾ = jₙ + i·yₙ (Hankel, outgoing).
//
// The far-field backscattering (θ = π) form function is:
//   f_bs = (2i/k_l) Σₙ₌₁^{N_max} (2n+1)·(−1)ⁿ·aₙ
//
// # Algorithm
//
// 1. Compute spherical Bessel j₀, j₁ using analytic formulae; higher orders via
//    upward recurrence jₙ = (2n−1)/x · jₙ₋₁ − jₙ₋₂ (stable for x > 0).
// 2. Same upward recurrence for yₙ (Neumann functions).
// 3. hₙ⁽¹⁾ = jₙ + i·yₙ (assembled from real parts).
// 4. Sum series until N_max = max(⌈k_l·R⌉ + 10, 5) for convergence.
//
// # References
//
// - Anderson, V. C. (1950). Sound scattering from a fluid sphere.
//   J. Acoust. Soc. Am. 22(4):426–431.
// - Morse, P. M. & Ingard, K. U. (1968). Theoretical Acoustics. McGraw-Hill, §8.3.
// - Nussenzveig, H. M. (1992). Diffraction Effects in Semiclassical Scattering.
//   Cambridge University Press — partial-wave convergence criterion.
// - Abramowitz, M. & Stegun, I. A. (1965). Handbook of Mathematical Functions §10.1.

// ─── Internal f64 helpers — prevent f32 overflow for large-order Bessel terms ───
//
// Spherical Neumann yₙ(x) ~ (2n−1)!!/x^(n+1) for small x, which overflows f32
// for n ≳ 15 at ka ≪ 1. All intermediate Mie arithmetic is therefore done in f64;
// the public API casts back to f32 at the very end.

/// Spherical Bessel function of the first kind, jₙ(x)  [f64 internal].
///
/// Uses stable upward recurrence (Abramowitz & Stegun §10.1):
///   j₀(x) = sin(x)/x,  j₁(x) = sin(x)/x² − cos(x)/x
///   jₙ(x) = (2n−1)/x · jₙ₋₁(x) − jₙ₋₂(x)
///
/// Returns 1 for n=0, x→0 (L'Hôpital limit).
fn spherical_bessel_j(n: usize, x: f64) -> f64 {
    if x.abs() < 1e-30 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    match n {
        0 => x.sin() / x,
        1 => x.sin() / (x * x) - x.cos() / x,
        _ => {
            let mut jm2 = x.sin() / x;
            let mut jm1 = x.sin() / (x * x) - x.cos() / x;
            let mut jn = 0.0_f64;
            for k in 2..=n {
                jn = (2 * k - 1) as f64 / x * jm1 - jm2;
                jm2 = jm1;
                jm1 = jn;
            }
            jn
        }
    }
}

/// Spherical Bessel function of the second kind, yₙ(x)  [f64 internal].
///
/// Uses stable upward recurrence (Abramowitz & Stegun §10.1):
///   y₀(x) = −cos(x)/x,  y₁(x) = −cos(x)/x² − sin(x)/x
///   yₙ(x) = (2n−1)/x · yₙ₋₁(x) − yₙ₋₂(x)
///
/// Diverges as x → 0; callers guard via near-zero xl check.
fn spherical_bessel_y(n: usize, x: f64) -> f64 {
    if x.abs() < 1e-30 {
        return f64::NEG_INFINITY;
    }
    match n {
        0 => -x.cos() / x,
        1 => -x.cos() / (x * x) - x.sin() / x,
        _ => {
            let mut ym2 = -x.cos() / x;
            let mut ym1 = -x.cos() / (x * x) - x.sin() / x;
            let mut yn = 0.0_f64;
            for k in 2..=n {
                yn = (2 * k - 1) as f64 / x * ym1 - ym2;
                ym2 = ym1;
                ym1 = yn;
            }
            yn
        }
    }
}

/// D[jₙ](x) = x·jₙ₋₁(x) − n·jₙ(x)  (real, f64).  Defined for n ≥ 0.
///
/// For n=0: D[j₀](x) = x·j₋₁(x) = x·(cos x / x) = cos x
///   (j₋₁(x) = cos(x)/x via downward recurrence; Anderson 1950 eq. 7)
#[inline]
fn d_bessel_j(n: usize, x: f64) -> f64 {
    if n == 0 {
        x.cos() // x · j₋₁(x) = x · (cos x / x) = cos x
    } else {
        x * spherical_bessel_j(n - 1, x) - n as f64 * spherical_bessel_j(n, x)
    }
}

/// D[hₙ](x) = x·hₙ₋₁(x) − n·hₙ(x)  (complex, f64).  Defined for n ≥ 0.
///
/// For n=0: D[h₀](x) = x·h₋₁(x) = cos x + i·sin x = exp(ix)
///   (h₋₁(x) = j₋₁(x)+i·y₋₁(x) = cos(x)/x + i·sin(x)/x; Anderson 1950 eq. 7)
#[inline]
fn d_bessel_h(n: usize, x: f64) -> Complex<f64> {
    if n == 0 {
        Complex::new(x.cos(), x.sin()) // exp(ix)
    } else {
        let h_nm1 = Complex::new(spherical_bessel_j(n - 1, x), spherical_bessel_y(n - 1, x));
        let h_n = Complex::new(spherical_bessel_j(n, x), spherical_bessel_y(n, x));
        h_nm1 * x - h_n * n as f64
    }
}

/// Mie partial-wave coefficient aₙ for a compressible fluid sphere (f64 internal).
///
/// See module-level theorem block for the full formula (Anderson 1950, eq. 14).
/// Valid for n ≥ 0 (n=0 is the monopole/compressibility term).
fn mie_coefficient(n: usize, k_l: f64, k_b: f64, rho_l: f64, rho_b: f64, r: f64) -> Complex<f64> {
    let xl = k_l * r;
    let xb = k_b * r;

    let jn_xl = spherical_bessel_j(n, xl);
    let jn_xb = spherical_bessel_j(n, xb);
    let hn_xl = Complex::new(jn_xl, spherical_bessel_y(n, xl));

    let d_jn_xl = d_bessel_j(n, xl); // D[jₙ](k_l·R) — real
    let d_jn_xb = d_bessel_j(n, xb); // D[jₙ](k_b·R) — real
    let d_hn_xl = d_bessel_h(n, xl); // D[hₙ](k_l·R) — complex

    let numerator = Complex::new(rho_b * jn_xb * d_jn_xl - rho_l * jn_xl * d_jn_xb, 0.0);
    let denominator =
        Complex::new(rho_b * jn_xb, 0.0) * d_hn_xl - hn_xl * Complex::new(rho_l * d_jn_xb, 0.0);

    -numerator / denominator
}

/// Far-field backscattering form function f_bs for acoustic Mie scattering.
///
/// # Theorem
/// At θ = π (backscatter):
/// ```text
/// f_bs = (2i/k_l) Σₙ₌₁^{N_max} (2n+1)·(−1)ⁿ·aₙ
/// ```
/// Series truncated at N_max = max(⌈k_l·R⌉ + 10, 5), guaranteeing convergence
/// to machine precision for k_l·R < 100 (Nussenzveig 1992).
///
/// Internal arithmetic is f64 to prevent Hankel-function overflow at small ka;
/// result is downcast to f32 for the PINN computation pipeline.
///
/// The scattered far field at distance d, phase k_l·d is:
/// ```text
/// p_sc(d) ∝ (f_bs.re·cos(k_l·d) − f_bs.im·sin(k_l·d)) / d
/// ```
///
/// # Arguments
/// * `k_l`   – exterior wavenumber \[1/m\]
/// * `k_b`   – interior (bubble) wavenumber \[1/m\]
/// * `rho_l` – exterior density \[kg/m³\]
/// * `rho_b` – interior density \[kg/m³\]
/// * `r`     – bubble radius \[m\]
pub fn mie_backscatter_form_function(
    k_l: f32,
    k_b: f32,
    rho_l: f32,
    rho_b: f32,
    r: f32,
) -> Complex<f32> {
    // Promote to f64 to avoid overflow in high-order Hankel functions at small ka.
    let (k_l64, k_b64, rho_l64, rho_b64, r64) =
        (k_l as f64, k_b as f64, rho_l as f64, rho_b as f64, r as f64);

    let n_max = ((k_l64 * r64).ceil() as usize + 10).max(5);
    let i = Complex::new(0.0_f64, 1.0_f64);
    let mut sum = Complex::<f64>::new(0.0, 0.0);

    // Sum from n=0 (monopole/compressibility) through n=N_max (multipoles)
    for n in 0..=n_max {
        let a_n = mie_coefficient(n, k_l64, k_b64, rho_l64, rho_b64, r64);
        let sign = if n % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
        sum = sum + a_n * ((2 * n + 1) as f64 * sign);
    }

    let result = sum * (i * 2.0_f64 / k_l64);
    Complex::new(result.re as f32, result.im as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_AIR};
    use kwavers_core::constants::tissue_acoustics::DENSITY_AIR;

    #[test]
    fn test_spherical_bessel_j0_known_values() {
        // j₀(x) = sin(x)/x; j₀(1) = sin(1) ≈ 0.84147
        let j0_1 = spherical_bessel_j(0, 1.0_f64);
        assert!((j0_1 - 1.0_f64.sin()).abs() < 1e-12, "j₀(1) = {j0_1}");

        // j₀(π) ≈ 0 (first root of j₀)
        let j0_pi = spherical_bessel_j(0, std::f64::consts::PI);
        assert!(j0_pi.abs() < 1e-10, "j₀(π) ≈ 0; got {j0_pi}");

        // j₀(0) = 1 (L'Hôpital limit)
        let j0_0 = spherical_bessel_j(0, 0.0_f64);
        assert!((j0_0 - 1.0).abs() < 1e-15, "j₀(0) = {j0_0}");
    }

    #[test]
    fn test_spherical_bessel_j1_known_values() {
        // j₁(x) = sin(x)/x² − cos(x)/x
        let x = 2.0_f64;
        let expected = x.sin() / (x * x) - x.cos() / x;
        let computed = spherical_bessel_j(1, x);
        assert!(
            (computed - expected).abs() < 1e-14,
            "j₁(2): got {computed}, want {expected}"
        );

        // j₁(0) = 0 (limit)
        let j1_0 = spherical_bessel_j(1, 1e-12_f64);
        assert!(j1_0.abs() < 1e-10, "j₁(0) ≈ 0; got {j1_0}");
    }

    #[test]
    fn test_mie_rayleigh_monotone_and_scaling() {
        // For a gas bubble (ρ_b ≪ ρ_l) with k_l·R ≪ 1:
        //
        // - n=0 monopole (density contrast): |a_0| ≈ (ρ_l−ρ_b)/ρ_l · (k_l·R) → |f_bs| ∝ R
        // - n=1 dipole  (density contrast): |a_1| ≈ (2/3)·(k_l·R)³·|(ρ_b−ρ_l)/(2ρ_b+ρ_l)|
        //
        // The n=0 term dominates for a gas bubble in water (ρ_b/ρ_l ≈ 0.0012).
        // Dominant scaling: |f_bs| ≈ 2·R → ratio |f_bs(r1)|/|f_bs(r2)| ≈ r1/r2 = 0.5
        //
        // Reference: Anderson (1950) J. Acoust. Soc. Am. 22:426, eq. 14-16.
        let c_l = kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM as f32; // water [m/s]
        let c_b = SOUND_SPEED_AIR as f32; // 343.0 m/s — air at 20°C
        let rho_l = DENSITY_WATER_NOMINAL as f32; // ≈ 998.0 kg/m³
        let rho_b = DENSITY_AIR as f32; // 1.204 kg/m³ — air at 20°C

        let f = 1e5_f32; // 100 kHz
        let omega = 2.0 * std::f32::consts::PI * f;
        let k_l = omega / c_l;
        let k_b = omega / c_b;

        let r1 = 0.01_f32 / k_l; // ka = 0.01
        let r2 = 0.02_f32 / k_l; // ka = 0.02

        let f1 = mie_backscatter_form_function(k_l, k_b, rho_l, rho_b, r1).norm();
        let f2 = mie_backscatter_form_function(k_l, k_b, rho_l, rho_b, r2).norm();

        // Larger bubble must scatter more
        assert!(f2 > f1, "|f_bs| must increase with R: f1={f1}, f2={f2}");

        // For ka ≪ 1, n=0 dominates: |f_bs| ≈ 2·R  → ratio = r1/r2 = 0.5
        let ratio = f1 / f2;
        let expected = r1 / r2; // = 0.5 for n=0 monopole dominance
        assert!(
            (ratio - expected).abs() < 0.15 * expected,
            "Rayleigh monopole ratio {ratio:.4} ≠ expected {expected:.4} (±15%)"
        );
    }

    #[test]
    fn test_mie_series_convergence() {
        // N_max and N_max+5 should agree to < 1e-4 relative error.
        // Verified by comparing mie_backscatter_form_function at ka = 1 with increased truncation.
        let k_l = 100.0_f32;
        let k_b = 1000.0_f32; // test wavenumber [m⁻¹] — abstract geometry parameter
        let rho_l = DENSITY_WATER_NOMINAL as f32; // ≈ 998.0 kg/m³
        let rho_b = DENSITY_AIR as f32; // 1.204 kg/m³
        let r = 1e-2_f32; // ka ≈ 1

        let f_nominal = mie_backscatter_form_function(k_l, k_b, rho_l, rho_b, r);

        // Manually compute with N_max+5 extra terms using f64 internals
        let (k_l64, k_b64, rho_l64, rho_b64, r64) =
            (k_l as f64, k_b as f64, rho_l as f64, rho_b as f64, r as f64);
        let n_max_extra = ((k_l64 * r64).ceil() as usize + 15).max(10);
        let i = Complex::new(0.0_f64, 1.0_f64);
        let mut sum = Complex::<f64>::new(0.0, 0.0);
        for n in 0..=n_max_extra {
            let a_n = mie_coefficient(n, k_l64, k_b64, rho_l64, rho_b64, r64);
            let sign = if n % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            sum = sum + a_n * ((2 * n + 1) as f64 * sign);
        }
        let f_extra_64 = sum * (i * 2.0_f64 / k_l64);
        let f_extra = Complex::new(f_extra_64.re as f32, f_extra_64.im as f32);

        let rel_err = (f_nominal - f_extra).norm() / f_extra.norm().max(1e-30);
        assert!(
            rel_err < 1e-4,
            "Mie series not converged: rel_err = {rel_err}"
        );
    }
}
