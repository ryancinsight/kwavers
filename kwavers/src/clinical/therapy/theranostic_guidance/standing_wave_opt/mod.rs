//! Iterative standing-wave suppression via Green's function precomputation
//! and gradient-descent phase optimization.
//!
//! # Algorithm
//!
//! 1. **Precompute** — one linearised 2D FDTD run per element (Rayon parallel)
//!    produces the monochromatic complex Green's function `G_i(x,y)`.
//! 2. **Reconstruct** — `p(x,y; φ) = Σ_i exp(iφ_i) G_i(x,y)` (Born operator).
//! 3. **Analyse** — SWI from spectral λ/2 fringe; `p_focal` from peak |p|.
//! 4. **Optimise** — gradient-descent update of phases with Armijo backtracking.
//! 5. **Repeat** steps 2–4 for `n_opt_iter` iterations.

mod config;
mod fdtd;
mod medium;
mod optimizer;
mod result;
mod swi;

pub use config::StandingWaveOptConfig;
pub use result::StandingWaveOptResult;

use fdtd::compute_all_green_functions;
use medium::{build_sound_speed, pml_damping};
use optimizer::run_optimization;

/// Run the standing-wave suppression pipeline and return the full result.
///
/// # Example
/// ```no_run
/// use kwavers::clinical::therapy::theranostic_guidance::standing_wave_opt::{
///     run_standing_wave_suppression, StandingWaveOptConfig,
/// };
/// let config = StandingWaveOptConfig::default();
/// let result = run_standing_wave_suppression(&config);
/// println!("SWI {:.3} → {:.3}", result.swi_history[0], result.swi_history.last().unwrap());
/// ```
pub fn run_standing_wave_suppression(config: &StandingWaveOptConfig) -> StandingWaveOptResult {
    let c_map = build_sound_speed(config);
    let damp = pml_damping(config);
    let element_ys = config.element_ys();
    let (g_re, g_im) = compute_all_green_functions(&c_map, &damp, &element_ys, config);
    run_optimization(g_re, g_im, config)
}
