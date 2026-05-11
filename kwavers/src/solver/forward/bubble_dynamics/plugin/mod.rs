//! Plugin adapter for bubble dynamics.
//!
//! Bridges the [`Plugin`] contract to the three production bubble-equation
//! implementations:
//!
//! | [`BubbleModel`] variant | ODE / integrator used |
//! |-------------------------|-----------------------|
//! | `KellerMiksis`          | [`BubbleField`] (adaptive KM, `use_compressibility = true`) |
//! | `RayleighPlesset`       | [`BubbleField`] (adaptive KM, `use_compressibility = false` → O(Mach⁰) limit) |
//! | `Gilmore`               | [`GilmoreSolver`] with per-voxel classical RK4, Tait liquid EOS |
//!
//! ## Field contract
//!
//! | Direction | [`UnifiedFieldType`] | Physical meaning |
//! |-----------|----------------------|-----------------|
//! | reads     | `Pressure`           | far-field acoustic driving pressure (Pa) |
//! | writes    | `BubbleRadius`       | instantaneous bubble radius R(t) (m) |
//! | writes    | `BubbleVelocity`     | bubble-wall velocity Ṙ(t) (m/s) |
//!
//! ## Nucleation seeding
//!
//! When `nucleation = false` (default), a single bubble is seeded at the
//! grid centre.  When `nucleation = true`, eight additional bubbles are
//! seeded at ±¼-domain offsets from the centre, modelling a focal zone
//! nucleation cloud.  All bubbles share the same `BubbleParameters`.
//!
//! ## dp/dt computation
//!
//! The [`BubbleField`] update requires `dp_dt_field` (the time derivative of
//! acoustic pressure) for the Keller-Miksis radiation-damping term.  This
//! plugin stores the previous-step pressure and computes a first-order
//! backward-difference estimate:
//!
//! ```text
//! dp_dt[i,j,k] ≈ (p_n[i,j,k] − p_{n-1}[i,j,k]) / dt
//! ```
//!
//! On the first call the denominator is finite but the numerator is zero
//! (previous pressure initialised to the current pressure at `initialize`
//! time), giving `dp_dt = 0` for the first step — the correct cold-start
//! behaviour.
//!
//! ## References
//!
//! - Keller & Miksis (1980) J. Acoust. Soc. Am. 68(2):628–633.
//! - Gilmore (1952) Caltech Hydrodynamics Lab Report 26-4.
//! - Rayleigh (1917) Phil. Mag. 34:94.

mod config;
mod engine;
mod plugin;

#[cfg(test)]
mod tests;

pub use config::BubbleDynamicsConfig;
pub use plugin::BubbleDynamicsPlugin;
