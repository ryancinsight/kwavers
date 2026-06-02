use std::collections::HashMap;

use ndarray::Array3;

use kwavers_physics::acoustics::bubble_dynamics::{
    bubble_field::BubbleField, bubble_state::BubbleState, gilmore::GilmoreSolver,
};

/// Per-model runtime engine, created lazily in [`Plugin::initialize`].
///
/// `KmOrRp` covers both Keller-Miksis and Rayleigh-Plesset because the
/// existing [`BubbleField`] code path handles both: when
/// `BubbleParameters::use_compressibility = false` the KM O(Mach) correction
/// factors collapse to unity, recovering the incompressible RP equation.
///
/// `Gilmore` drives per-voxel integration via [`GilmoreSolver::step_rk4`] —
/// the RK4 loop lives inside the solver where it belongs (SRP), not here.
/// The Gilmore path does **not** store a `prev_pressure` field because the
/// Gilmore ODE receives the instantaneous pressure at each voxel directly from
/// the field array; no dp/dt estimate is required.
pub(super) enum BubbleEngine {
    /// Keller-Miksis or Rayleigh-Plesset via existing adaptive BubbleField.
    KmOrRp {
        field: Box<BubbleField>,
        /// Previous-step pressure, used to estimate dp/dt via backward difference.
        ///
        /// The [`BubbleField::update`] signature requires `dp_dt_field` for the
        /// KM radiation-damping term.  A first-order backward difference
        /// `dp_dt[i,j,k] ≈ (p_n − p_{n-1}) / dt` is sufficient for the
        /// O(Mach) accuracy level of the KM equation.
        prev_pressure: Array3<f64>,
    },
    /// Gilmore equation (Tait EOS) via per-voxel [`GilmoreSolver::step_rk4`].
    ///
    /// State carries only the live solver and the per-voxel `BubbleState`
    /// map. `BubbleParameters` are owned by the surrounding `BubblePluginConfig`
    /// (`self.config.params`) and read from there when needed; carrying a copy
    /// in this variant would duplicate the SSOT held by the plugin.
    Gilmore {
        solver: GilmoreSolver,
        /// Per-voxel bubble states, keyed by grid index (i, j, k).
        states: HashMap<(usize, usize, usize), BubbleState>,
    },
}

impl std::fmt::Debug for BubbleEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KmOrRp { .. } => write!(f, "BubbleEngine::KmOrRp"),
            Self::Gilmore { .. } => write!(f, "BubbleEngine::Gilmore"),
        }
    }
}
