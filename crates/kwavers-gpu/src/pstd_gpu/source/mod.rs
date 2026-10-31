//! Source-schedule assembly shared by GPU PSTD entry points.
//!
//! Both the direct runner and the simulation-trait adapter upload these
//! schedules. Pressure (`p_mask`/`p_signal`) and x-velocity
//! (`u_mask`/`u_signal`) schedules live in their own submodules so their
//! finite-aperture, local-medium, and source-mode contracts cannot drift;
//! the k-space-correction gate they share lives in `correction`. Keeping the
//! conversion here prevents the entry points' contracts from drifting.

pub mod correction;
pub mod pressure;
pub mod velocity;

pub(crate) use correction::source_mode_uses_kspace_correction;
pub use pressure::{prepare_pstd_pressure_source, PstdPressureSourceSchedule};
pub use velocity::{prepare_pstd_velocity_source, PstdVelocitySourceSchedule};
