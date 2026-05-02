//! K-space correction operators for FDTD — spectral gradient/divergence.
//!
//! # Theory: K-Space Corrected FDTD (Treeby & Cox 2010, §II.A)
//!
//! Standard FDTD uses finite-difference spatial gradients with phase-velocity
//! error O(kΔx)². k-Wave eliminates this error by replacing finite-difference
//! gradients with spectral (FFT) derivatives and adding a temporal sinc
//! correction factor κ.
//!
//! ## Update equations
//!
//! ```text
//!   u^{n+½} = u^{n-½} − (dt/ρ₀) · IFFT[ ddx_k_shift_pos · κ · FFT(p^n) ]
//!   p^{n+1} = p^n     − dt·ρ₀c₀² · IFFT[ ddx_k_shift_neg · κ · FFT(u^{n+½}) ]
//! ```
//!
//! ## References
//!
//! - Treeby, B.E. & Cox, B.T. (2010). J. Biomed. Opt. 15(2), 021314.
//! - Liu, Q.-H. (1998). Microwave Opt. Technol. Lett. 15(3), 158–165.

mod operators;
#[cfg(test)]
mod tests;

pub use operators::KSpaceFdtdOperators;
