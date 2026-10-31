//! Grid Source Definition
//!
//! Data structures for defining sources on the computational grid.

use leto::{Array2, Array3};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SourceMode {
    #[default]
    Additive,
    /// Additive without k-space source correction (k-Wave `additive-no-correction`)
    AdditiveNoCorrection,
    Dirichlet, // Enforce value (hard source)
}

/// Container for grid-distributed source definitions
#[derive(Default, Clone, Debug)]
pub struct GridSource {
    // Initial conditions
    pub p0: Option<Array3<f64>>,
    pub u0: Option<(Array3<f64>, Array3<f64>, Array3<f64>)>,

    // Time varying pressure source
    // If p_mask is defined, p_signal must be defined
    pub p_mask: Option<Array3<f64>>,
    pub p_signal: Option<Array2<f64>>, // [num_sources, time_steps]
    pub p_mode: SourceMode,

    // Time varying velocity source
    pub u_mask: Option<Array3<f64>>,
    pub u_signal: Option<Array3<f64>>, // [3, num_sources, time_steps]
    pub u_mode: SourceMode,
}

impl GridSource {
    #[must_use]
    pub fn new_empty() -> Self {
        Self::default()
    }
}

/// Per-source-point velocity-source scale `2·c₀·Δt/Δα` along one axis.
///
/// For an additive velocity source on a leapfrog grid the per-step injection
/// that reproduces the canonical analytical mass-flow boundary condition is
/// `u_α(x_s, t_n) ← u_α(x_s, t_n) + (2·c₀(x_s)·Δt/Δα) · u_α^src(t_n)`,
/// matching `kwave/solvers/kspace_solver.py` (`scale = 2 * c0_src * dt / di`)
/// and the C++ k-Wave binary's `velocity_source_input` term. `Dirichlet`
/// mode is normalised differently (replacement, not addition), so its scale
/// is fixed at 1.0 and the user-supplied signal is the velocity itself.
///
/// This is the one owner of the formula: the CPU FDTD source handler and the
/// GPU PSTD schedule assembly both call it, so the providers cannot drift.
#[must_use]
pub fn velocity_source_scale(sound_speed: f64, dt: f64, spacing: f64, mode: SourceMode) -> f64 {
    match mode {
        SourceMode::Dirichlet => 1.0,
        SourceMode::Additive | SourceMode::AdditiveNoCorrection => 2.0 * sound_speed * dt / spacing,
    }
}

#[cfg(test)]
mod tests {
    use super::{velocity_source_scale, SourceMode};

    #[test]
    fn additive_scale_matches_the_k_wave_quotient() {
        assert_eq!(
            velocity_source_scale(1500.0, 1e-7, 1e-3, SourceMode::Additive),
            0.3
        );
        assert_eq!(
            velocity_source_scale(1500.0, 1e-7, 1e-3, SourceMode::AdditiveNoCorrection),
            0.3
        );
    }

    #[test]
    fn dirichlet_scale_is_unity_whatever_the_grid() {
        assert_eq!(
            velocity_source_scale(1500.0, 1e-7, 1e-3, SourceMode::Dirichlet),
            1.0
        );
        assert_eq!(
            velocity_source_scale(0.0, 0.0, 1.0, SourceMode::Dirichlet),
            1.0
        );
    }
}
