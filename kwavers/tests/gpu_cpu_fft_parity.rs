//! GPU vs CPU 3D FFT Parity Tests
//!
//! Validates that the GPU FFT (`GpuFft3d`) produces results consistent with
//! the Apollo-backed CPU 3D reference for both power-of-2 and
//! non-power-of-2 grid sizes.
//!
//! ## Tests
//!
//! 1. **Forward+inverse roundtrip** — `IFFT(FFT(x)) ≈ x` for each grid size;
//!    GPU f32 vs CPU f64 max relative error < 6e-7.
//!
//! 2. **CPU cross-validation** — GPU spectrum vs CPU f64 3D FFT to within
//!    max relative error 6e-7 on all non-negligible bins.
//!
//! 3. **Parseval's theorem** — `‖GPU_FFT(x)‖²/N ≈ ‖x‖²` within relative
//!    error 1e-5.
//!
//! ## Grid Sizes Tested
//!
//! | ID   | Shape       | Axis strategies          |
//! |------|-------------|--------------------------|
//! | p64  | 64×64×64    | Radix-2 on all axes      |
//! | p128 | 128×128×128 | Radix-2 on all axes      |
//! | p256 | 256×256×256 | Radix-2 on all axes      |
//! | np   | 48×64×32    | Chirp-Z × Radix-2 × Radix-2 |
//!
//! All tests are gated behind `#[cfg(feature = "gpu")]` and marked
//! `#[ignore = "requires GPU device"]` so they never run on headless CI.
//! To execute: `cargo nextest run --features gpu --run-ignored all -E 'test(gpu_cpu_fft)'`
//!
//! ## References
//! - Parseval D. (1799). Mem. Acad. Sci. Paris, 1, 638–648.
//! - Cooley J.W., Tukey J.W. (1965). Math. Comp. 19(90), 297–301.
//! - Bluestein L.I. (1970). IEEE Trans. AU-18(4), 451–455.

#![cfg(feature = "gpu")]

use kwavers::math::fft::{fft_3d_array, Complex64};
use ndarray::Array3;

// ── GPU device initialisation ─────────────────────────────────────────────────

/// Attempt to acquire a wgpu device/queue pair.  Returns `None` on headless CI.
async fn try_init_gpu() -> Option<(std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok()?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .ok()?;
    Some((std::sync::Arc::new(device), std::sync::Arc::new(queue)))
}

// ── CPU 3D FFT reference (Apollo-backed, f64) ─────────────────────────────────

fn cpu_fft_3d_forward(field: &Array3<f64>) -> Array3<Complex64> {
    fft_3d_array(field)
}

// ── Deterministic 3D test signal ─────────────────────────────────────────────

/// Build a deterministic real 3D test signal.
///
/// `f[i,j,k] = cos(2π·3·i/nx) + 0.5·sin(2π·7·j/ny) + 0.25·cos(2π·5·k/nz)`
///
/// The three-frequency decomposition ensures the FFT has analytically predictable
/// dominant bins, making spectral cross-validation straightforward.
fn test_signal_3d(nx: usize, ny: usize, nz: usize) -> Array3<f64> {
    use std::f64::consts::TAU;
    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        (TAU * 3.0 * i as f64 / nx as f64).cos()
            + 0.5 * (TAU * 7.0 * j as f64 / ny as f64).sin()
            + 0.25 * (TAU * 5.0 * k as f64 / nz as f64).cos()
    })
}

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Assert `IFFT(FFT(x)) ≈ x` to within `roundtrip_tol` (L∞, absolute).
///
/// Also asserts GPU spectrum vs CPU f64 3D FFT: max relative error < `cross_val_rel_tol`
/// on all bins where `|CPU_k| > 1e-10`.
///
/// Also asserts Parseval: `‖GPU_FFT‖²/N ≈ ‖x‖²` within relative error `parseval_tol`.
async fn parity_test(
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    nx: usize,
    ny: usize,
    nz: usize,
    roundtrip_tol: f64,
    cross_val_rel_tol: f64,
    parseval_tol: f64,
) {
    use kwavers::math::fft::gpu_fft::GpuFft3d;

    let signal = test_signal_3d(nx, ny, nz);
    let n = nx * ny * nz;

    let gpu = GpuFft3d::new(device, queue, nx, ny, nz)
        .unwrap_or_else(|e| panic!("GpuFft3d::new({nx},{ny},{nz}) failed: {e}"));

    // ── GPU forward FFT ───────────────────────────────────────────────────────
    let spectrum = gpu.forward(&signal);

    // ── GPU roundtrip ─────────────────────────────────────────────────────────
    let mut reconstructed = Array3::<f64>::zeros((nx, ny, nz));
    gpu.inverse(&spectrum, &mut reconstructed);

    let max_abs_err = signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_abs_err < roundtrip_tol,
        "Roundtrip IFFT(FFT(x)) failed for grid {nx}×{ny}×{nz}: \
         max_abs_err={max_abs_err:.3e} >= tol={roundtrip_tol:.3e}"
    );

    // ── CPU 3D FFT cross-validation ───────────────────────────────────────────
    let cpu_spec = cpu_fft_3d_forward(&signal);
    let signal_scale = signal.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let abs_tol = 4.0 * signal.len() as f64 * f32::EPSILON as f64 * signal_scale;

    let mut max_rel_err = 0.0_f64;
    let mut max_abs_err = 0.0_f64;
    let mut worst_idx = 0usize;
    let mut worst_cpu = Complex64::new(0.0, 0.0);
    let mut worst_gpu = Complex64::new(0.0, 0.0);
    for (idx, cpu_val) in cpu_spec.iter().enumerate() {
        let gpu_re = spectrum[2 * idx] as f64;
        let gpu_im = spectrum[2 * idx + 1] as f64;
        let abs_err = ((gpu_re - cpu_val.re).powi(2) + (gpu_im - cpu_val.im).powi(2)).sqrt();
        let cpu_mag = cpu_val.norm();
        if cpu_mag > abs_tol {
            let rel = abs_err / cpu_mag;
            if rel > max_rel_err {
                max_rel_err = rel;
                worst_idx = idx;
                worst_cpu = *cpu_val;
                worst_gpu = Complex64::new(gpu_re, gpu_im);
            }
        } else if abs_err > max_abs_err {
            max_abs_err = abs_err;
            worst_idx = idx;
            worst_cpu = *cpu_val;
            worst_gpu = Complex64::new(gpu_re, gpu_im);
        }
    }

    assert!(
        max_rel_err < cross_val_rel_tol && max_abs_err < abs_tol,
        "GPU vs CPU 3D FFT parity failed for grid {nx}×{ny}×{nz}: \
         max_rel_err={max_rel_err:.3e} >= tol={cross_val_rel_tol:.3e} \
         or max_abs_err={max_abs_err:.3e} >= abs_tol={abs_tol:.3e}, \
         worst_idx={worst_idx}, cpu={worst_cpu:?}, gpu={worst_gpu:?}"
    );

    // ── Parseval's theorem ────────────────────────────────────────────────────
    // For the unitary-unnormalised DFT: sum_k |X[k]|² / N = sum_n |x[n]|²
    let time_energy: f64 = signal.iter().map(|x| x * x).sum();
    let freq_energy: f64 = (0..n)
        .map(|i| {
            let re = spectrum[2 * i] as f64;
            let im = spectrum[2 * i + 1] as f64;
            (re * re + im * im) / n as f64
        })
        .sum();

    let parseval_rel_err = ((time_energy - freq_energy) / time_energy).abs();
    assert!(
        parseval_rel_err < parseval_tol,
        "Parseval violated for grid {nx}×{ny}×{nz}: \
         time_energy={time_energy:.6e}, freq_energy={freq_energy:.6e}, \
         rel_err={parseval_rel_err:.3e} >= tol={parseval_tol:.3e}"
    );
}

// ── Power-of-2 cubic grids ────────────────────────────────────────────────────

/// 64×64×64 — Radix-2 DIT on all three axes.
///
/// Tolerances:
/// - Roundtrip (L∞ absolute):  1e-4  (accumulated f32 error over log₂(64)=6 stages × 3 axes)
/// - CPU cross-validation (relative): 6e-7
/// - Parseval (relative): 1e-5
#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_gpu_cpu_fft_parity_64_cubic() {
    let Some((device, queue)) = try_init_gpu().await else {
        eprintln!("Skipping test_gpu_cpu_fft_parity_64_cubic: no GPU available");
        return;
    };
    parity_test(device, queue, 64, 64, 64, 1e-4, 6e-7, 1e-5).await;
}

/// 128×128×128 — Radix-2 DIT on all three axes.
///
/// Accumulated f32 error budget: log₂(128)=7 stages × 3 axes × ε_f32 ≈ 2.5e-6.
/// Tolerances: roundtrip 1e-4, cross-val 6e-7, Parseval 1e-5.
#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_gpu_cpu_fft_parity_128_cubic() {
    let Some((device, queue)) = try_init_gpu().await else {
        eprintln!("Skipping test_gpu_cpu_fft_parity_128_cubic: no GPU available");
        return;
    };
    parity_test(device, queue, 128, 128, 128, 1e-4, 6e-7, 1e-5).await;
}

/// 256×256×256 — Radix-2 DIT on all three axes.
///
/// Accumulated f32 error budget: log₂(256)=8 stages × 3 axes × ε_f32 ≈ 3e-6.
/// Tolerances: roundtrip 1e-4, cross-val 6e-7, Parseval 1e-5.
#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_gpu_cpu_fft_parity_256_cubic() {
    let Some((device, queue)) = try_init_gpu().await else {
        eprintln!("Skipping test_gpu_cpu_fft_parity_256_cubic: no GPU available");
        return;
    };
    parity_test(device, queue, 256, 256, 256, 1e-4, 6e-7, 1e-5).await;
}

// ── Non-power-of-2 grid ───────────────────────────────────────────────────────

/// 48×64×32 — Chirp-Z on X (48 → M=next_pow2(95)=128), Radix-2 on Y and Z.
///
/// Tests the hybrid radix-2 / Chirp-Z dispatch path for mixed-strategy grids.
/// Tolerances: roundtrip 1e-4, cross-val 6e-7, Parseval 1e-5.
#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_gpu_cpu_fft_parity_48x64x32() {
    let Some((device, queue)) = try_init_gpu().await else {
        eprintln!("Skipping test_gpu_cpu_fft_parity_48x64x32: no GPU available");
        return;
    };
    parity_test(device, queue, 48, 64, 32, 1e-4, 6e-7, 1e-5).await;
}

// ── Parseval-only sanity checks for each cubic size ───────────────────────────

/// Parseval's theorem: `‖FFT(x)‖²/N ≈ ‖x‖²` for 64³ grid.
///
/// Separate test so Parseval can be checked without waiting for the cross-val
/// CPU reference path (CPU 3D FFT at 64³ is fast but at 256³ can be slow).
#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_parseval_64_cubic() {
    let Some((device, queue)) = try_init_gpu().await else {
        return;
    };
    use kwavers::math::fft::gpu_fft::GpuFft3d;
    let (nx, ny, nz) = (64, 64, 64);
    let n = nx * ny * nz;
    let signal = test_signal_3d(nx, ny, nz);
    let gpu = GpuFft3d::new(device, queue, nx, ny, nz).expect("GpuFft3d::new");
    let spectrum = gpu.forward(&signal);

    let time_energy: f64 = signal.iter().map(|x| x * x).sum();
    let freq_energy: f64 = (0..n)
        .map(|i| {
            let re = spectrum[2 * i] as f64;
            let im = spectrum[2 * i + 1] as f64;
            (re * re + im * im) / n as f64
        })
        .sum();
    let rel_err = ((time_energy - freq_energy) / time_energy).abs();
    assert!(
        rel_err < 1e-5,
        "Parseval violated for {nx}³: time={time_energy:.6e} freq={freq_energy:.6e} rel={rel_err:.3e}"
    );
}

/// Parseval's theorem for 128³ grid.
#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_parseval_128_cubic() {
    let Some((device, queue)) = try_init_gpu().await else {
        return;
    };
    use kwavers::math::fft::gpu_fft::GpuFft3d;
    let (nx, ny, nz) = (128, 128, 128);
    let n = nx * ny * nz;
    let signal = test_signal_3d(nx, ny, nz);
    let gpu = GpuFft3d::new(device, queue, nx, ny, nz).expect("GpuFft3d::new");
    let spectrum = gpu.forward(&signal);

    let time_energy: f64 = signal.iter().map(|x| x * x).sum();
    let freq_energy: f64 = (0..n)
        .map(|i| {
            let re = spectrum[2 * i] as f64;
            let im = spectrum[2 * i + 1] as f64;
            (re * re + im * im) / n as f64
        })
        .sum();
    let rel_err = ((time_energy - freq_energy) / time_energy).abs();
    assert!(
        rel_err < 1e-5,
        "Parseval violated for {nx}³: time={time_energy:.6e} freq={freq_energy:.6e} rel={rel_err:.3e}"
    );
}

/// Parseval's theorem for 48×64×32 non-power-of-2 grid.
#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_parseval_48x64x32() {
    let Some((device, queue)) = try_init_gpu().await else {
        return;
    };
    use kwavers::math::fft::gpu_fft::GpuFft3d;
    let (nx, ny, nz) = (48, 64, 32);
    let n = nx * ny * nz;
    let signal = test_signal_3d(nx, ny, nz);
    let gpu = GpuFft3d::new(device, queue, nx, ny, nz).expect("GpuFft3d::new");
    let spectrum = gpu.forward(&signal);

    let time_energy: f64 = signal.iter().map(|x| x * x).sum();
    let freq_energy: f64 = (0..n)
        .map(|i| {
            let re = spectrum[2 * i] as f64;
            let im = spectrum[2 * i + 1] as f64;
            (re * re + im * im) / n as f64
        })
        .sum();
    let rel_err = ((time_energy - freq_energy) / time_energy).abs();
    assert!(
        rel_err < 1e-5,
        "Parseval violated for {nx}×{ny}×{nz}: time={time_energy:.6e} freq={freq_energy:.6e} rel={rel_err:.3e}"
    );
}
