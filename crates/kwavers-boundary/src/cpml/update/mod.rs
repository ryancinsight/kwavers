//! CPML field update implementation
//!
//! # Theorem: CPML Recursive Convolution (Roden & Gedney 2000)
//!
//! The Convolutional PML (CPML) replaces the standard PML gradient `∂f/∂x` with
//! an effective gradient that includes a memory (auxiliary field) ψ:
//!
//! ```text
//!   ∂f/∂x_eff = (1/κ) · ∂f/∂x + ψ
//! ```
//!
//! The memory field ψ satisfies the recursive convolution update:
//! ```text
//!   ψ^{n+1} = b · ψ^n + a · (∂f/∂x)^n
//! ```
//!
//! where:
//! - `b = exp(−σ·Δt)` (decay factor, `b ∈ (0,1)` for σ > 0)
//! - `a = b − 1 = exp(−σ·Δt) − 1` (amplitude coefficient, always ≤ 0)
//! - `κ` = stretch factor (κ = 1 in the basic CPML formulation used here)
//!
//! ## References
//! - Roden, J.A. & Gedney, S.D. (2000). Convolution PML (CPML): An efficient FDTD
//!   implementation of the CFS-PML for arbitrary media. Microwave Opt. Tech. Lett. 27(5), 334–339.

use super::memory::CPMLMemory;
use super::profiles::CPMLProfiles;

use leto::Array3;

mod axis;

/// CPML field updater
#[derive(Debug, Clone)]
pub struct CPMLUpdater {}

impl CPMLUpdater {
    /// Create new updater
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// Update memory component for acoustic gradients (used in velocity update).
    /// `component`: 0=x, 1=y, 2=z (out-of-range is a no-op).
    pub fn update_p_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        component: usize,
        profiles: &CPMLProfiles,
    ) {
        self.update_p_memory_axis(memory, gradient, component, profiles);
    }

    /// Update memory component for velocity gradients (used in pressure update).
    /// `component`: 0=x, 1=y, 2=z (out-of-range is a no-op).
    pub fn update_v_memory(
        &self,
        memory: &mut CPMLMemory,
        v_gradient: &Array3<f64>,
        component: usize,
        profiles: &CPMLProfiles,
    ) {
        self.update_v_memory_axis(memory, v_gradient, component, profiles);
    }

    /// Apply gradient correction from CPML memory.
    /// `component`: 0=x, 1=y, 2=z (out-of-range is a no-op).
    pub fn apply_p_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        component: usize,
        profiles: &CPMLProfiles,
    ) {
        self.apply_p_correction_axis(gradient, memory, component, profiles);
    }

    /// Apply velocity gradient correction from CPML memory.
    /// `component`: 0=x, 1=y, 2=z (out-of-range is a no-op).
    pub fn apply_v_correction(
        &self,
        v_gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        component: usize,
        profiles: &CPMLProfiles,
    ) {
        self.apply_v_correction_axis(v_gradient, memory, component, profiles);
    }
}

impl Default for CPMLUpdater {
    fn default() -> Self {
        Self::new()
    }
}
