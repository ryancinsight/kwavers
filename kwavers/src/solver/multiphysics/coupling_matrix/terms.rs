//! Per-interface-cell coupling correction vectors.

/// Coupling corrections at a single interface cell (Zienkiewicz et al. 2013, §12.3).
#[derive(Debug, Clone)]
pub struct CouplingTerms {
    /// Fluid velocity correction (3 components): Δv_f = −(dt/ρ_f) · ü_solid · n̂
    pub delta_fluid_velocity: [f64; 3],
    /// Solid stress correction (6 Voigt components: σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz):
    /// Δσ = −p_fluid · (n̂ ⊗ n̂)
    pub delta_solid_stress: [f64; 6],
}
