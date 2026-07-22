//! Real-space exponential PML for the elastic PSTD orchestrator.
//!
//! # Theorem (PML attenuation)
//!
//! Let `σ_α(x_α) ≥ 0` be the per-axis damping profile (zero in the
//! interior, monotonically increasing through a thickness `L_α` at the
//! boundary). The PML modification of the elastic equations
//!
//! ```text
//!   ∂_t v_α + σ_α v_α = (1/ρ) (∇·σ)_α
//!   ∂_t σ_αβ + (σ_α + σ_β)/2 · σ_αβ = C_αβγδ ε̇_γδ
//! ```
//!
//! admits the asymptotic solution `v_α(x_α) ∝ exp(−∫₀^x_α σ_α(x') dx')`
//! for an outgoing wave. Hence a wave that traverses a PML of thickness
//! `L_α` and integrated absorption `Σ_α = ∫₀^L_α σ_α(x) dx` is attenuated
//! by `exp(−Σ_α)` in amplitude (twice in round-trip, so reflectivity
//! `R ≈ exp(−2 Σ_α)`).
//!
//! For a polynomial profile `σ_α(x) = σ_max · (x/L_α)^p` with `p = 4` and
//! `σ_max` chosen to give theoretical reflection coefficient `R₀`,
//!
//! ```text
//!   σ_max = − (p + 1) c_max ln(R₀) / (2 L_α)
//! ```
//!
//! follows from Roden & Gedney (2000) eq. 25 with the standard
//! cell-thickness normalisation. For `R₀ = 10⁻⁴`, `p = 4`,
//! `c_max = 1500 m/s`, `L_α = 10·dx`, this gives
//! `σ_max ≈ 4.6 · 10⁵ s⁻¹` in air-water acoustic units.
//!
//! # Implementation strategy
//!
//! The orchestrator applies the PML in real space immediately after the
//! IFFT-back-to-velocity step:
//!
//! ```text
//!   v_α(x, t+dt)  ←  v_α(x, t+dt) · exp(−σ_α(x_α)·dt)
//!   σ_αβ(x, t+dt) ←  σ_αβ(x, t+dt) · exp(−(σ_α(x_α) + σ_β(x_β))/2 · dt)
//! ```
//!
//! The exponential factor is precomputed once at PML construction
//! (`damping_x`i` = exp(−σ_α(x_i)·dt)`) and applied each step as a
//! per-cell multiply. This is simpler than Berenger's split-field PML —
//! it doesn't require splitting velocity and stress into directional
//! sub-fields — but achieves the same exponential absorption at the cost
//! of slightly higher residual reflection. For the dominant
//! pykwavers/KWave.jl parity use cases (short propagation, sensor
//! recording within ≥1 wavelength of the boundary) this is sufficient;
//! Berenger split-field PML is on the backlog for long-range elastic.
//!
//! # Stability
//!
//! Real-space exponential damping is unconditionally stable for any
//! `σ_α · dt ≥ 0` since the multiplier `exp(−σ_α·dt) ∈ (0, 1]`. No
//! additional CFL constraint is introduced.

use leto::{Array1, Array3};

/// Per-axis exponential damping coefficients precomputed for the
/// orchestrator's grid. All arrays are real-valued and shaped `(N_α,)`
/// since the damping depends on the boundary-distance index alone.
#[derive(Debug, Clone)]
pub struct ElasticPml {
    /// Number of grid cells in the absorbing layer per side along each
    /// axis. `0` ⇒ no PML on that axis.
    thickness_cells: (usize, usize, usize),
    /// `damping_x`i` = exp(−σ_x(i)·dt)` ∈ (0, 1].  Outside the absorbing
    /// layer this is exactly 1.0.
    damping_x: Array1<f64>,
    damping_y: Array1<f64>,
    damping_z: Array1<f64>,
}

/// Immutable construction parameters shared by real-space and split-field
/// elastic PMLs.
///
/// The grid shape, spacing, layer thickness, time step, wave-speed bound, and
/// target reflection coefficient form one Roden-Gedney profile contract. Both
/// PML variants consume this same specification so their per-axis damping
/// profiles cannot diverge.
#[derive(Debug, Clone, Copy)]
pub struct ElasticPmlSpec {
    /// Grid shape `(nx, ny, nz)`.
    pub shape: (usize, usize, usize),
    /// Absorbing-layer thickness in cells per side along each axis.
    pub thickness_cells: (usize, usize, usize),
    /// Grid spacing `(dx, dy, dz)` in metres.
    pub spacing: (f64, f64, f64),
    /// Maximum elastic wave speed in metres per second.
    pub c_max: f64,
    /// Time step in seconds.
    pub dt: f64,
    /// Target theoretical reflection coefficient at normal incidence.
    pub r0: f64,
}

impl ElasticPml {
    /// Build a PML for a `(nx, ny, nz)` grid with the given per-axis
    /// thickness in cells.
    ///
    /// `c_max` is the maximum sound speed in the medium (used to set
    /// `σ_max` for a target theoretical reflection coefficient `R0`).
    /// `dt` is the simulation time step. `dx, dy, dz` are the grid
    /// spacings.
    ///
    /// `r0` is the target theoretical reflection coefficient at normal
    /// incidence (Roden & Gedney 2000); `1e-4` is a standard choice.
    /// The polynomial order `p = 4` is hard-coded as the literature
    /// optimum for spectral solvers (Treeby & Cox 2010 Appendix B).
    #[must_use]
    pub fn new(spec: ElasticPmlSpec) -> Self {
        const P: f64 = 4.0;
        let (nx, ny, nz) = spec.shape;
        let (dx, dy, dz) = spec.spacing;
        let damping_x = build_axis_damping(
            nx,
            spec.thickness_cells.0,
            dx,
            spec.c_max,
            spec.dt,
            spec.r0,
            P,
        );
        let damping_y = build_axis_damping(
            ny,
            spec.thickness_cells.1,
            dy,
            spec.c_max,
            spec.dt,
            spec.r0,
            P,
        );
        let damping_z = build_axis_damping(
            nz,
            spec.thickness_cells.2,
            dz,
            spec.c_max,
            spec.dt,
            spec.r0,
            P,
        );
        Self {
            thickness_cells: spec.thickness_cells,
            damping_x,
            damping_y,
            damping_z,
        }
    }

    /// Apply the PML damping to a real-space velocity / displacement
    /// field, in place.  `damping_field[i, j, k] *= damping_x`i` *
    /// damping_y`J` * damping_z`K``.
    pub fn apply_to_field(&self, field: &mut Array3<f64>) {
        let dx_s = self.damping_x.as_slice().expect("damping_x contiguous");
        let dy_s = self.damping_y.as_slice().expect("damping_y contiguous");
        let dz_s = self.damping_z.as_slice().expect("damping_z contiguous");
        let [nx, ny, nz] = field.shape();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    field[[i, j, k]] *= dx_s[i] * dy_s[j] * dz_s[k];
                }
            }
        }
    }

    /// Borrow the per-axis damping coefficients (for tests / diagnostics).
    #[must_use]
    pub fn damping_axes(&self) -> (&Array1<f64>, &Array1<f64>, &Array1<f64>) {
        (&self.damping_x, &self.damping_y, &self.damping_z)
    }

    /// Borrow the per-axis thickness in cells.
    #[must_use]
    pub fn thickness_cells(&self) -> (usize, usize, usize) {
        self.thickness_cells
    }
}

pub(super) fn build_axis_damping(
    n: usize,
    thickness: usize,
    dx: f64,
    c_max: f64,
    dt: f64,
    r0: f64,
    p: f64,
) -> Array1<f64> {
    let mut damping = Array1::<f64>::ones([n]);
    if thickness == 0 || n < 2 * thickness + 1 {
        return damping;
    }
    let layer_len = thickness as f64 * dx;
    // Roden & Gedney (2000) eq. 25 — optimal σ_max for a polynomial profile
    // of order p giving theoretical reflection coefficient r0.
    let sigma_max = -(p + 1.0) * c_max * r0.ln() / (2.0 * layer_len);
    debug_assert!(
        sigma_max.is_finite() && sigma_max > 0.0,
        "PML σ_max must be positive and finite (r0 = {r0}, c_max = {c_max}, \
         layer_len = {layer_len})"
    );
    // Lower-side absorbing layer: indices 0..thickness, normalised distance
    // from the inner boundary x = 0 outward.
    for i in 0..thickness {
        let x = (thickness - i) as f64 / thickness as f64;
        let sigma = sigma_max * x.powf(p);
        damping[i] = (-sigma * dt).exp();
    }
    // Upper-side absorbing layer: indices n-thickness..n.
    for i in (n - thickness)..n {
        let x = (i - (n - thickness - 1)) as f64 / thickness as f64;
        let sigma = sigma_max * x.powf(p);
        damping[i] = (-sigma * dt).exp();
    }
    damping
}

/// Build per-cell exponential-decay (`alpha`) and RHS-integration (`beta`)
/// coefficient arrays for one axis of the split-field PML.
///
/// The profile matches [`build_axis_damping`]:
/// `σ_i = σ_max · ((distance from interior edge) / L)^p`,
/// `σ_max = −(p+1) c_max ln(r0) / (2L)` (Roden & Gedney 2000 eq. 25).
///
/// Returned arrays (length `n`):
/// - `alpha`i` = exp(−σ_i · dt)` ∈ (0, 1].  Interior: `alpha = 1`.
/// - `beta`i`  = (1 − alpha`i`) / σ_i` for σ_i > 0; otherwise `dt`.
///   Absorbs `dt` so the caller's RHS need not carry the time-step factor
///   (exact integrator: `f^{n+1} = alpha·f^n + beta·g`, where in the
///   interior α=1, β=dt → standard leapfrog).
pub(super) fn build_axis_alpha_beta(
    n: usize,
    thickness: usize,
    dx: f64,
    c_max: f64,
    dt: f64,
    r0: f64,
    p: f64,
) -> (Array1<f64>, Array1<f64>) {
    let mut alpha = Array1::<f64>::ones([n]);
    let mut beta = Array1::<f64>::from_elem([n], dt);
    if thickness == 0 || n < 2 * thickness + 1 {
        return (alpha, beta);
    }
    let layer_len = thickness as f64 * dx;
    let sigma_max = -(p + 1.0) * c_max * r0.ln() / (2.0 * layer_len);
    debug_assert!(
        sigma_max.is_finite() && sigma_max > 0.0,
        "PML σ_max must be positive and finite (r0 = {r0}, c_max = {c_max}, \
         layer_len = {layer_len})"
    );
    for i in 0..thickness {
        let x = (thickness - i) as f64 / thickness as f64;
        let sigma = sigma_max * x.powf(p);
        let a = (-sigma * dt).exp();
        alpha[i] = a;
        beta[i] = (1.0 - a) / sigma;
    }
    for i in (n - thickness)..n {
        let x = (i - (n - thickness - 1)) as f64 / thickness as f64;
        let sigma = sigma_max * x.powf(p);
        let a = (-sigma * dt).exp();
        alpha[i] = a;
        beta[i] = (1.0 - a) / sigma;
    }
    (alpha, beta)
}

#[cfg(test)]
mod tests;
