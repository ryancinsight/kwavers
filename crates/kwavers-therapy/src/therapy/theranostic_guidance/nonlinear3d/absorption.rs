//! Treeby & Cox 2010 fractional-Laplacian power-law absorption for the
//! nonlinear 3-D Westervelt FDTD.
//!
//! # Physics contract
//!
//! The lossless Westervelt recurrence used by `forward::update_cells` is
//!
//! ```text
//!   p[n+1] = sponge · (2 p[n] − p[n−1] + (c·dt)² ∇²p + q · ∂²(p²)/∂t²)
//! ```
//!
//! Tissue absorbs acoustic energy with a power-law frequency dependence
//! `α(f) = α₀·|f|^y`. Treeby & Cox 2010 §III.B Eq. 11 expresses this in
//! the wave-equation form:
//!
//! ```text
//!   (1/c² · ∂²/∂t² − ∇²) p = τ̃·∂L_y(p)/∂t + η̃·L_{y+1}(p)
//!   τ̃ = −2 α₀ c^(y−1)           (Eq. 9)
//!   η̃ = −2 α₀ c^y · tan(π y / 2) (Eq. 10)
//! ```
//!
//! Multiplying by `c²` and discretising `∂L_y/∂t` with a one-sided
//! difference (consistent with the leapfrog Westervelt stencil) gives
//! the FDTD absorption contribution
//!
//! ```text
//!   Δp_abs[n+1] = −dt · τ · ( L_y(p[n]) − L_y(p[n−1]) )
//!   τ  = 2 α₀ c^(y+1)
//! ```
//!
//! with `L_a(p) = IFFT( |k|^a · FFT(p) )` evaluated on the 3-D periodic
//! spatial grid via full-spectrum complex FFTs through the kwavers FFT facade
//! (`math::fft::fft_3d_array_into` / `ifft_3d_array_into`).
//!
//! # Why η is omitted (Kramers-Kronig dispersion)
//!
//! Von-Neumann analysis on the Westervelt leapfrog with an explicit
//! `−dt²·η·L_{y+1}(p[n])` term shows a Nyquist-mode growth factor
//!
//! ```text
//!   |z|² ≈ 1 + dt²·|η|·k_max^(y+1)
//! ```
//!
//! which exceeds unity for any `y < 2` at clinically realistic `α₀` and
//! `dt`. For `y = 2` (Stokes-Kirchhoff, cortical skull) `tan(π y / 2) = 0`
//! so `η ≡ 0` analytically; for `y ≈ 1.05` (soft tissue) the explicit
//! form is conditionally unstable. We therefore drop the η term and keep
//! only the τ absorption: the magnitude is matched to Treeby-Cox 2010
//! exactly (verified by the discrete dispersion analysis
//! `α(ω) = α₀_ω·ω^y` in the unit tests below), at the cost of a
//! second-order phase-velocity correction that is sub-leading versus the
//! FDTD's intrinsic numerical dispersion for the 1–10 pts/λ regime used
//! by `forward_with_schedule`.
//!
//! # Heterogeneity scope
//!
//! - **α₀ (per voxel)**: full heterogeneity supported via the per-voxel
//!   `dt_tau` array. A skull voxel (α₀ ≈ 150 Np/m at 1 MHz) and a
//!   soft-tissue voxel (α₀ ≈ 5.8 Np/m) contribute their own local
//!   absorption magnitude.
//! - **y (one scalar per simulation)**: matches the canonical kwavers
//!   PSTD convention.  The spectral filter `|k|^y` is global in k-space.
//!   The volume-area-weighted median `y` selected inside
//!   [`construction`] yields `y ≈ 2.0` for a brain volume and
//!   `y ≈ 1.05` for an abdominal volume.
//!
//! # Adjoint
//!
//! `L_y` is self-adjoint on the periodic spatial grid (real symmetric
//! multiplier in k-space) and the per-voxel `dt·τ` factor is diagonal in
//! real space and therefore self-adjoint. Consequently the discrete
//! adjoint of [`FractionalLaplacianAbsorption::apply`] has the same
//! structure with the source and destination roles of `next/curr/prev`
//! permuted; see [`FractionalLaplacianAbsorption::apply_transpose`] for
//! the chain-rule transpose used by `adjoint::gradient`.
//!
//! # References
//!
//! - Treeby & Cox 2010. *J. Biomed. Opt.* 15(2), 021314, Eqs. 9–11.
//! - Hamilton & Blackstock 1998 §3.6 (Stokes-Kirchhoff limit `y = 2`).
//! - Connor & Hynynen 2002 (skull `y ≈ 1.9 – 2.0`).

use leto::Array3;

mod apply;
mod construction;
mod spectrum;

#[cfg(test)]
mod tests;

/// Precomputed fractional-Laplacian absorption operator for one homogeneous
/// power-law exponent `y`.
#[derive(Debug)]
pub(super) struct FractionalLaplacianAbsorption {
    n: usize,
    /// Per-voxel `dt · τ` factor in flat row-major layout (`n³` entries).
    dt_tau: Vec<f64>,
    /// Half-spectrum `|k|^y` weights for `L_y` (`n × n × (n/2+1)`).
    k_pow_y: Array3<f64>,
    /// Cached `L_y(p[n-1])` from the previous step.  Empty until the
    /// solver primes it after the first absorption application.
    prev_l_y: Option<Vec<f64>>,
}

/// Inputs for constructing a fractional-Laplacian absorption operator.
pub(super) struct AbsorptionBuilder<'a> {
    pub n: usize,
    pub spacing_m: f64,
    pub dt_s: f64,
    pub speed_m_s: &'a [f64],
    pub attenuation_np_per_m_mhz: &'a [f64],
    pub attenuation_power_law_y: &'a [f64],
}
