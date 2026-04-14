//! GPU-accelerated 3D FFT — thin re-export of `apollofft-wgpu`.
//!
//! The canonical implementation lives in `apollofft-wgpu::infrastructure::gpu_fft`.
//! It supports arbitrary grid dimensions via Bluestein Chirp-Z Transform for
//! non-power-of-2 axes and iterative Cooley-Tukey radix-2 DIT for power-of-2 axes.
//!
//! # Algorithm
//! For arbitrary N (Bluestein 1970, IEEE Trans. AU-18):
//! ```text
//! X[k] = W^{k²/2} Σₙ (x[n]·W^{n²/2}) · W^{-(k-n)²/2}
//! ```
//! where W = exp(-2πi/N).  Evaluated as a length-M circular convolution via
//! M-point radix-2 FFT, M = next_pow2(2N-1), giving O(N log N) for all N ≥ 1.

/// Returns `true` when the `gpu` feature is enabled.
pub fn gpu_fft_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        apollofft_wgpu::gpu_fft_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

#[cfg(feature = "gpu")]
pub use apollofft_wgpu::GpuFft3d;
