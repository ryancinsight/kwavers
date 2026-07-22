//! SIMD-accelerated wave pressure field operations using `hermes-simd`.
//!
//! This example demonstrates the `hermes_simd` runtime-dispatch free functions
//! that replace hand-rolled SIMD loops in the kwavers wave solver kernel:
//!
//! | Operation             | hermes function         | kwavers use |
//! |-----------------------|-------------------------|-------------|
//! | pressure superposition | `elementwise_add`      | multi-source IVP |
//! | PML attenuation       | `scale`                 | PML σ·Δt damping |
//! | velocity update       | `axpy`                  | u += −Δt/ρ·∇p |
//! | energy diagnostics    | `dot`, `sum`            | field monitoring |
//!
//! ## Architecture
//!
//! ```text
//! hermes_simd dispatch layer (all safe at call site)
//!   ├── AVX-512  8 × f64 per cycle  (512-bit SIMD, Intel Ice Lake+)
//!   ├── AVX2     4 × f64 per cycle  (256-bit SIMD, Intel Haswell+)
//!   ├── NEON     2 × f64 per cycle  (128-bit SIMD, AArch64)
//!   └── Scalar   1 × f64            (portable, identical semantics)
//! ```
//!
//! `kwavers` stays `#[forbid(unsafe_code)]` — all `unsafe` is encapsulated in
//! `hermes_simd_intrinsics` and exposed only through the safe dispatch facade.
//!
//! ## Theorem — AXPY velocity update exactness
//!
//! Given velocity `u = 0` and pressure gradient `∇p`, one step of the
//! staggered-grid velocity update is:
//!
//! ```text
//! u_new[i] = u[i] + α · ∇p[i]   where  α = −Δt/ρ₀
//! ```
//!
//! Therefore `||u_new||² = α² · ||∇p||²`, which the `axpy` test verifies
//! to machine precision.
//!
//! # Chapter reference
//!
//! Part VI — Atlas Stack Integration, §SIMD: Hermes for Vectorized Operations.

use hermes_simd::{axpy, dot, elementwise_add, scale, sum};

// ── Physical constants ────────────────────────────────────────────────────
const N: usize = 1024; // 1-D field length
const DT: f64 = 1.0e-8; // time step [s]
const RHO0: f64 = 1000.0; // water density [kg/m³]
const SIGMA_PML: f64 = 1.0e6; // PML attenuation coefficient [Np/s]

fn main() {
    println!("SIMD wave kernel demo (hermes-simd runtime dispatch)");
    println!("field size: {N}, dt: {DT:.1e} s, ρ₀: {RHO0:.0} kg/m³");
    println!("backend: {}", runtime_backend());
    println!();

    // ── 1. Pressure superposition: p_out = p_a + p_b ─────────────────────
    //
    // In multi-source problems (e.g. two focused transducers), the total
    // initial pressure is the sum of individual source contributions.

    let sigma = N as f64 * 0.08;
    let center_a = N as f64 / 2.0;
    let center_b = N as f64 / 4.0;

    let p_a: Vec<f64> = (0..N).map(|i| gauss(i as f64, center_a, sigma)).collect();
    let p_b: Vec<f64> = (0..N)
        .map(|i| 0.5 * gauss(i as f64, center_b, sigma))
        .collect();

    let energy_a = dot(&p_a, &p_a).expect("equal lengths");
    let energy_b = dot(&p_b, &p_b).expect("equal lengths");

    let mut p_super = vec![0.0_f64; N];
    elementwise_add(&p_a, &p_b, &mut p_super).expect("equal lengths");

    let _total_before = energy_a + energy_b;
    let energy_super = dot(&p_super, &p_super).expect("equal lengths");

    // Cross-term means energy_super ≠ total_before; check linearity of sum instead
    let sum_super = sum(&p_super);
    let sum_ref: f64 = p_a.iter().zip(&p_b).map(|(a, b)| a + b).sum();
    let add_error = (sum_super - sum_ref).abs() / sum_ref.abs().max(f64::EPSILON);

    println!("1. elementwise_add — pressure superposition");
    println!("   ||p_a||² = {energy_a:.8e}");
    println!("   ||p_b||² = {energy_b:.8e}");
    println!("   Σ(p_a+p_b) = {sum_super:.8e}  (ref: {sum_ref:.8e}, err: {add_error:.2e})");
    assert!(
        add_error < 1e-12,
        "elementwise_add error {add_error:.2e} exceeds tolerance"
    );
    println!("   → PASS");
    println!();

    // ── 2. PML damping: p *= exp(−σ·Δt) ─────────────────────────────────
    //
    // The perfectly matched layer attenuates outgoing waves each step by
    // multiplying the pressure field by a spatially-varying coefficient.
    // Here we apply a uniform coefficient for illustration.

    let damping = (-SIGMA_PML * DT).exp();
    let mut p_damped = p_super.clone();
    scale(&mut p_damped, damping);

    let energy_before = energy_super;
    let energy_after = dot(&p_damped, &p_damped).expect("equal lengths");
    let expected_ratio = damping * damping;
    let actual_ratio = energy_after / energy_before;
    let pml_error = (actual_ratio - expected_ratio).abs();

    println!("2. scale — PML attenuation (exp(−σΔt) = {damping:.14})");
    println!("   E_before = {energy_before:.8e}");
    println!("   E_after  = {energy_after:.8e}");
    println!("   ratio    = {actual_ratio:.14}  (expected {expected_ratio:.14})");
    println!("   error    = {pml_error:.2e}");
    assert!(
        pml_error < 1e-12,
        "PML energy ratio error {pml_error:.2e} exceeds tolerance"
    );
    println!("   → PASS");
    println!();

    // ── 3. Velocity update: u += α · ∇p  (AXPY, α = −Δt/ρ₀) ─────────────
    //
    // The staggered-grid acoustic update is:
    //   u^{n+1/2} = u^{n-1/2} − (Δt/ρ₀) · ∇p^n
    // which is AXPY with α = −Δt/ρ₀.

    // Approximate ∇p with forward differences on p_damped
    let mut grad_p = vec![0.0_f64; N];
    for i in 0..N - 1 {
        grad_p[i] = p_damped[i + 1] - p_damped[i];
    }

    let mut u = vec![0.0_f64; N];
    let alpha = -DT / RHO0;
    axpy(alpha, &grad_p, &mut u).expect("equal lengths");

    // Theorem: u = α·∇p  ⟹  ||u||² = α²·||∇p||²
    let vel_energy = dot(&u, &u).expect("equal lengths");
    let grad_energy = dot(&grad_p, &grad_p).expect("equal lengths");
    let expected_vel_energy = alpha * alpha * grad_energy;
    let axpy_error =
        (vel_energy - expected_vel_energy).abs() / expected_vel_energy.max(f64::EPSILON);

    println!("3. axpy — velocity update (α = {alpha:.4e})");
    println!("   ||∇p||² = {grad_energy:.8e}");
    println!("   ||u||²  = {vel_energy:.8e}  (expected {expected_vel_energy:.8e})");
    println!("   relative error = {axpy_error:.2e}");
    assert!(
        axpy_error < 1e-12,
        "AXPY velocity energy error {axpy_error:.2e} exceeds tolerance"
    );
    println!("   → PASS");
    println!();

    // ── 4. Field diagnostics: sum and dot ─────────────────────────────────

    let integral_p = sum(&p_damped);
    let l2_norm = dot(&p_damped, &p_damped).map(|e| e.sqrt()).unwrap();

    println!("4. sum + dot — field diagnostics");
    println!("   ∫p dx     = {integral_p:.8e}");
    println!("   ||p||_L2  = {l2_norm:.8e}");

    // Peak detection via argmax
    let (peak_idx, peak_val) = hermes_simd::argmax(&p_a).expect("non-empty");
    let expected_peak_idx = center_a as usize;
    assert!(
        (peak_idx as isize - expected_peak_idx as isize).unsigned_abs() <= 1,
        "peak index {peak_idx} should be near {expected_peak_idx}"
    );
    println!("   peak p_a at idx {peak_idx} = {peak_val:.8e}  (expected near {expected_peak_idx})");
    println!("   → PASS");
    println!();

    println!("All SIMD correctness checks PASS");
    println!("Runtime backend: {}", runtime_backend());
}

/// Gaussian pulse.
#[inline]
fn gauss(x: f64, center: f64, sigma: f64) -> f64 {
    let d = x - center;
    (-d * d / (2.0 * sigma * sigma)).exp()
}

/// Report the SIMD width that will be used at runtime.
fn runtime_backend() -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return "AVX-512 (8×f64)";
        }
        if std::is_x86_feature_detected!("avx2") {
            return "AVX2 (4×f64)";
        }
        "Scalar fallback"
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "NEON (2×f64)";
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return "Scalar fallback";
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_simd::{elementwise_mul, elementwise_sub};

    #[test]
    fn elementwise_add_identity_with_zeros() {
        let a: Vec<f64> = (0..64).map(|i| i as f64 * 1.5).collect();
        let b = vec![0.0_f64; 64];
        let mut out = vec![0.0_f64; 64];
        elementwise_add(&a, &b, &mut out).unwrap();
        assert_eq!(out, a);
    }

    #[test]
    fn scale_zero_clears_field() {
        let mut data: Vec<f64> = (0..128).map(|i| i as f64).collect();
        scale(&mut data, 0.0);
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn scale_two_doubles_energy() {
        let original: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
        let mut scaled = original.clone();
        scale(&mut scaled, 2.0);
        let e_orig = dot(&original, &original).unwrap();
        let e_scaled = dot(&scaled, &scaled).unwrap();
        assert!((e_scaled - 4.0 * e_orig).abs() < 1e-10 * e_orig);
    }

    #[test]
    fn axpy_zero_alpha_is_identity() {
        let x: Vec<f64> = (0..128).map(|i| i as f64 * 0.01).collect();
        let mut out: Vec<f64> = (0..128).map(|i| (i as f64).sin()).collect();
        let expected = out.clone();
        axpy(0.0, &x, &mut out).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn elementwise_sub_self_is_zero() {
        let data: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let mut out = vec![0.0_f64; 64];
        elementwise_sub(&data, &data, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn dot_with_self_equals_squared_sum() {
        let v: Vec<f64> = (0..64).map(|i| (i as f64 + 1.0).recip()).collect();
        let d = dot(&v, &v).unwrap();
        let ref_d: f64 = v.iter().map(|x| x * x).sum();
        assert!((d - ref_d).abs() < 1e-12 * ref_d.max(1.0));
    }

    #[test]
    fn elementwise_mul_distributes_over_sum() {
        let a: Vec<f64> = (0..64).map(|i| i as f64 * 0.25).collect();
        let b: Vec<f64> = (0..64).map(|i| (i as f64).cos()).collect();
        let mut out = vec![0.0_f64; 64];
        elementwise_mul(&a, &b, &mut out).unwrap();
        let ref_out: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        for (actual, expected) in out.iter().zip(&ref_out) {
            assert!((actual - expected).abs() < 1e-14);
        }
    }
}
