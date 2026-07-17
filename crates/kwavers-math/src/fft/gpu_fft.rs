//! GPU-accelerated 3D FFT facade backed by Apollo's FFT backend trait.
//!
//! The single source of truth for GPU FFT execution is Apollo. `kwavers`
//! exposes this narrow facade so solver code depends on the spectral contract
//! rather than on Apollo's internal module tree. Apollo currently exposes a
//! WGPU implementation for this contract; CUDA FFT remains an upstream backend
//! gap rather than a Kwavers-side placeholder.
//!
//! # Theorem
//! Let `F_N` be the unnormalised length-`N` DFT and let each 3D transform be
//! the tensor product `F_nx (*) F_ny (*) F_nz`. Applying the inverse Apollo
//! plan after the forward Apollo plan returns the input field because Apollo's
//! inverse kernels implement FFTW-compatible `1 / (nx ny nz)` normalisation.
//!
//! # Proof sketch
//! The 3D DFT matrix factorises into the Kronecker product of one-dimensional
//! DFT matrices. Each factor satisfies `F_N^{-1} F_N = I_N` under the inverse
//! `1 / N` scaling, so the tensor product satisfies
//! `(F_nx^{-1} (*) F_ny^{-1} (*) F_nz^{-1})(F_nx (*) F_ny (*) F_nz) = I`.
//! Apollo's current GPU plan evaluates the same separable transforms as the CPU path:
//! radix-2 Cooley-Tukey for power-of-two axes and Bluestein convolution for
//! non-power-of-two axes. The GPU facade therefore has one acceptance criterion:
//! its interleaved complex spectrum and inverse round trip must match Apollo
//! CPU results within the `f64 -> f32 -> f64` storage contract.
//!
//! # Algorithm
//! For arbitrary `N`, Apollo uses Bluestein's chirp-z transform:
//! ```text
//! X[k] = W^{k^2/2} sum_n (x[n] * W^{n^2/2}) * W^{-(k-n)^2/2}
//! ```
//! where `W = exp(-2*pi*i/N)`. It evaluates this as a length-`M` circular
//! convolution with `M = next_power_of_two(2N - 1)`, giving `O(N log N)` for
//! every positive axis length.

#[cfg(feature = "gpu")]
pub use apollo::{FftBackend, GpuFft3d, GpuFft3dBuffers, WgpuBackend};

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{FftBackend, WgpuBackend};
    use crate::fft::{fft_3d_array, Shape3D};
    use leto::Array3;

    fn try_plan(shape: Shape3D) -> Option<apollo::GpuFft3d> {
        let backend = match WgpuBackend::try_default() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("Apollo WGPU FFT backend unavailable in this environment: {error}");
                return None;
            }
        };

        let capabilities = backend.capabilities();
        assert!(capabilities.supports_3d);
        assert!(!capabilities.supports_1d);
        assert!(!capabilities.supports_2d);
        assert!(capabilities.supports_mixed_precision);

        Some(
            backend
                .plan_3d(shape)
                .expect("validated positive shape must produce a WGPU 3D FFT plan"),
        )
    }

    #[test]
    fn apollo_wgpu_fft_matches_cpu_spectrum_for_power_of_two_shape() {
        let shape = Shape3D::new(2, 4, 2).unwrap();
        let Some(plan) = try_plan(shape) else {
            return;
        };
        let field = Array3::from_shape_fn([shape.nx, shape.ny, shape.nz], |[i, j, k]| {
            (i as f64 + 1.0) - 0.25 * (j as f64) + 0.125 * (k as f64)
        });

        let gpu = plan.forward(&field);
        let cpu = fft_3d_array(&field);

        assert_eq!(gpu.len(), 2 * shape.volume());
        for (idx, (actual, expected)) in gpu.chunks_exact(2).zip(cpu.iter()).enumerate() {
            assert!(
                (actual[0] as f64 - expected.re).abs() <= 2.0e-4,
                "real spectrum mismatch at {idx}: actual={} expected={}",
                actual[0],
                expected.re
            );
            assert!(
                (actual[1] as f64 - expected.im).abs() <= 2.0e-4,
                "imag spectrum mismatch at {idx}: actual={} expected={}",
                actual[1],
                expected.im
            );
        }
    }

    #[test]
    fn apollo_wgpu_fft_round_trips_non_power_of_two_shape_with_reusable_buffers() {
        let shape = Shape3D::new(3, 2, 5).unwrap();
        let Some(plan) = try_plan(shape) else {
            return;
        };
        let field = Array3::from_shape_fn([shape.nx, shape.ny, shape.nz], |[i, j, k]| {
            ((i * 7 + j * 3 + k) as f64).sin()
        });
        let mut spectrum = vec![0.0_f32; 2 * shape.volume()];
        let mut buffers = apollo::GpuFft3dBuffers::new(&plan);
        let mut out = Array3::<f64>::from_shape_vec(
            [shape.nx, shape.ny, shape.nz],
            vec![0.0; shape.volume()],
        )
        .expect("zeroed Leto output must match the plan shape");

        plan.forward_into_with_buffers(&field, &mut spectrum, &mut buffers);
        plan.inverse_with_buffers(&spectrum, &mut out, &mut buffers);

        for (idx, (actual, expected)) in out
            .as_slice_memory_order()
            .expect("Leto output must be contiguous")
            .iter()
            .zip(field.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 2.0e-4,
                "round-trip mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
    }
}
