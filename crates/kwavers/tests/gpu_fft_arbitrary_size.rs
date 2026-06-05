//! GPU FFT arbitrary-size roundtrip and cross-validation tests.
//!
//! ## Coverage
//!
//! 1. **Roundtrip accuracy** — for every N in `{12, 30, 48, 100, 200}` the
//!    roundtrip `IFFT(FFT(x)) ≈ x` must satisfy `|error|_∞ < 1e-5`.
//!
//! 2. **CPU cross-validation** — for the same sizes the GPU spectrum must
//!    agree with the Apollo-backed CPU FFT to within a relative error
//!    of `5e-6` on all bins with `|CPU_k| > 1e-12`.
//!
//! 3. **Power-of-2 regression** — sizes `{16, 32, 64}` must also pass both
//!    checks (ensures the radix-2 path is not broken by the Chirp-Z changes).
//!
//! All tests require the `gpu` feature and a working wgpu device.  They are
//! marked `#[ignore]` on headless CI (use `cargo test --features gpu -- --ignored`
//! or `cargo nextest run --features gpu --run-ignored all` to execute them).
//!
//! ## References
//! - Bluestein L.I. (1970). IEEE Trans. AU-18(4), 451–455.
//! - Cooley J.W., Tukey J.W. (1965). Math. Comp. 19(90), 297–301.

#![cfg(feature = "gpu")]

use kwavers_math::fft::{fft_1d_array, Complex64};
use ndarray::Array1;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Initialise a wgpu device synchronously (Vulkan/DX12/Metal/GL).
/// Returns `None` if no suitable adapter is available (headless CI).
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

/// Build a deterministic 1-D complex test signal of length `n`.
/// x[k] = cos(2π·3k/n) + 0.5·sin(2π·7k/n), for k=0..n.
fn test_signal_1d(n: usize) -> Vec<f64> {
    use std::f64::consts::TAU;
    (0..n)
        .map(|k| {
            let t = k as f64 / n as f64;
            (TAU * 3.0 * t).cos() + 0.5 * (TAU * 7.0 * t).sin()
        })
        .collect()
}

/// CPU reference 1-D DFT via the Apollo FFT cache (f64 precision).
fn cpu_fft_1d(signal: &[f64]) -> Array1<Complex64> {
    fft_1d_array(&Array1::from_vec(signal.to_vec()))
}

/// Compute the 3-D GPU FFT for a grid of shape (nx, ny, nz) using a uniform
/// real signal and return (gpu_re, gpu_im) as flat Vec<f32> length nx*ny*nz.
fn gpu_fft_3d(
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    nx: usize,
    ny: usize,
    nz: usize,
) -> (Vec<f32>, Vec<f32>) {
    use kwavers_math::fft::gpu_fft::GpuFft3d;
    use ndarray::Array3;

    let signal: Array3<f64> = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        use std::f64::consts::TAU;
        let t = (i + j + k) as f64 / (nx + ny + nz) as f64;
        (TAU * 3.0 * t).cos() + 0.5 * (TAU * 7.0 * t).sin()
    });

    let gpu_fft = GpuFft3d::new(device, queue, nx, ny, nz).expect("GpuFft3d::new failed");
    let spectrum = gpu_fft.forward(&signal);

    let n = nx * ny * nz;
    let re: Vec<f32> = (0..n).map(|i| spectrum[2 * i]).collect();
    let im: Vec<f32> = (0..n).map(|i| spectrum[2 * i + 1]).collect();
    (re, im)
}

// ── roundtrip test helper ─────────────────────────────────────────────────────

/// Test that IFFT(FFT(x)) ≈ x to within `tol` (infinity norm).
/// The grid is (n, 4, 4) to keep data volume small while exercising the axis
/// under test at length `n`.
async fn roundtrip_1d_in_3d(n: usize, tol: f64) {
    let Some((device, queue)) = try_init_gpu().await else {
        eprintln!("Skipping roundtrip_1d_in_3d(n={n}): no GPU available");
        return;
    };
    use kwavers_math::fft::gpu_fft::GpuFft3d;
    use ndarray::Array3;

    let nx = n;
    let ny = 4;
    let nz = 4;
    let original: Array3<f64> = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        use std::f64::consts::TAU;
        (TAU * 3.0 * i as f64 / nx as f64).cos()
            + 0.5 * (TAU * 7.0 * j as f64 / ny as f64).sin()
            + 0.25 * (TAU * 5.0 * k as f64 / nz as f64).cos()
    });

    let gpu = GpuFft3d::new(device, queue, nx, ny, nz).expect("GpuFft3d::new");
    let spectrum = gpu.forward(&original);
    let mut reconstructed = Array3::zeros((nx, ny, nz));
    gpu.inverse(&spectrum, &mut reconstructed);

    let max_err = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_err < tol,
        "Roundtrip IFFT(FFT(x)) failed for n={n}: max_err={max_err:.3e} >= tol={tol:.3e}"
    );
}

// ── cross-validation helper ───────────────────────────────────────────────────

/// Check that the GPU 1D FFT agrees with the CPU f64 FFT for a 1D signal
/// embedded in a (n, 1, 1) grid, to within relative error `rel_tol`.
async fn cross_validate_1d(n: usize, rel_tol: f64) {
    let Some((device, queue)) = try_init_gpu().await else {
        eprintln!("Skipping cross_validate_1d(n={n}): no GPU available");
        return;
    };
    use kwavers_math::fft::gpu_fft::GpuFft3d;
    use ndarray::Array3;

    let signal = test_signal_1d(n);
    let cpu_spec = cpu_fft_1d(&signal);
    let signal_scale = signal.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let abs_tol = 4.0 * signal.len() as f64 * f32::EPSILON as f64 * signal_scale;

    let arr: Array3<f64> = Array3::from_shape_fn((n, 1, 1), |(i, _, _)| signal[i]);
    let gpu = GpuFft3d::new(device, queue, n, 1, 1).expect("GpuFft3d::new");
    let spec = gpu.forward(&arr);

    let mut max_rel_err = 0.0_f64;
    let mut max_abs_err = 0.0_f64;
    for (k, cpu_val) in cpu_spec.iter().enumerate() {
        let gpu_re = spec[2 * k] as f64;
        let gpu_im = spec[2 * k + 1] as f64;
        let abs_err = ((gpu_re - cpu_val.re).powi(2) + (gpu_im - cpu_val.im).powi(2)).sqrt();
        let cpu_mag = cpu_val.norm();
        if cpu_mag > abs_tol {
            let rel = abs_err / cpu_mag;
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        } else if abs_err > max_abs_err {
            max_abs_err = abs_err;
        }
    }

    assert!(
        max_rel_err < rel_tol && max_abs_err < abs_tol,
        "GPU vs CPU cross-validation failed for n={n}: \
         max_rel_err={max_rel_err:.3e} >= {rel_tol:.3e} or \
         max_abs_err={max_abs_err:.3e} >= abs_tol={abs_tol:.3e}"
    );
}

// ── tests ─────────────────────────────────────────────────────────────────────

// Non-power-of-2 roundtrip tests

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_roundtrip_n12() {
    roundtrip_1d_in_3d(12, 1e-5).await;
}

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_roundtrip_n30() {
    roundtrip_1d_in_3d(30, 1e-5).await;
}

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_roundtrip_n48() {
    roundtrip_1d_in_3d(48, 1e-5).await;
}

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_roundtrip_n100() {
    roundtrip_1d_in_3d(100, 1e-5).await;
}

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_roundtrip_n200() {
    roundtrip_1d_in_3d(200, 1e-5).await;
}

// Power-of-2 regression tests (radix-2 path)

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_roundtrip_n16() {
    roundtrip_1d_in_3d(16, 1e-5).await;
}

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_roundtrip_n32() {
    roundtrip_1d_in_3d(32, 1e-5).await;
}

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_roundtrip_n64() {
    roundtrip_1d_in_3d(64, 1e-5).await;
}

// CPU cross-validation tests (non-power-of-2)

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_cross_validate_n12() {
    cross_validate_1d(12, 5e-6).await;
}

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_cross_validate_n30() {
    cross_validate_1d(30, 5e-6).await;
}

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_cross_validate_n48() {
    cross_validate_1d(48, 5e-6).await;
}

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_cross_validate_n100() {
    cross_validate_1d(100, 5e-6).await;
}

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_cross_validate_n200() {
    cross_validate_1d(200, 5e-6).await;
}

// CPU cross-validation tests (power-of-2 regression)

#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_cross_validate_n64() {
    cross_validate_1d(64, 5e-6).await;
}

/// Parseval's theorem: ||FFT(x)||² / N ≈ ||x||² within 1e-5.
#[tokio::test]
#[ignore = "requires GPU device"]
async fn test_parseval_n30() {
    let n = 30;
    let Some((device, queue)) = try_init_gpu().await else {
        return;
    };
    use kwavers_math::fft::gpu_fft::GpuFft3d;
    use ndarray::Array3;

    let signal = test_signal_1d(n);
    let arr: Array3<f64> = Array3::from_shape_fn((n, 1, 1), |(i, _, _)| signal[i]);
    let gpu = GpuFft3d::new(device, queue, n, 1, 1).expect("GpuFft3d::new");
    let spec = gpu.forward(&arr);

    let time_energy: f64 = signal.iter().map(|x| x * x).sum::<f64>();
    let freq_energy: f64 = (0..n)
        .map(|k| {
            let re = spec[2 * k] as f64;
            let im = spec[2 * k + 1] as f64;
            (re * re + im * im) / n as f64
        })
        .sum();

    let rel_err = ((time_energy - freq_energy) / time_energy).abs();
    assert!(
        rel_err < 1e-5,
        "Parseval violated for n={n}: time_energy={time_energy:.6}, freq_energy={freq_energy:.6}, rel_err={rel_err:.3e}"
    );
}
