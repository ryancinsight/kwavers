//! Current-carrying capacity (ampacity) and conductor resistance — IPC-2221.
//!
//! # Phase 1a migration roadmap
//!
//! `ipc2221_min_width` already returns `Nm` (length newtype). The remaining flat-`f64`
//! parameters (`current_a`, `dt_c`, `copper_oz`, `len_m`, `width_m`) stay as `f64` at Phase 1a —
//! Phase 2 will swap to `Amp` / `Celsius` / `Meter` (plain backticks — those newtypes are not yet
//! wrapped) as the ampacity slice migrates. The DFM helpers (annular ring, aspect ratio)
//! similarly stay `f64` for now.
//!
//! A track must be wide enough to carry its current without overheating, and its `I²R` loss is a
//! distributed heat source that couples into the [`crate::physics::thermal`] field (electro-thermal
//! co-analysis). For the HV ultrasound driver the relevant current is the **RMS** of the pulsed
//! waveform (Joule heating is thermal, set by RMS, not the 1.5 A peak).
//!
//! IPC-2221 cross-section for a temperature rise `ΔT`:
//! ```text
//! A[mil²] = ( I / (k · ΔT^0.44) )^(1/0.725),   k = 0.048 external, 0.024 internal
//! width   = A / (copper thickness)
//! ```
//! Validated against the standard's canonical point: 1 A, 10 °C rise, 1 oz, external ⇒ ≈ 0.30 mm.

/// Black's electromigration MTTF relative-to-baseline degradation as a function of current
/// density `J` and absolute temperature `T` (the `J²·exp(-E_a/kT)` activation law).
pub mod electromigration;
/// IPC-2221 cross-section A = (I / (k·ΔT^0.44))^(1/0.725) → minimum copper width for an external
/// `k=0.048` / internal `k=0.024` layer given a steady-state current, temperature rise, and copper
/// weight. The 1 A / 10 °C / 1 oz external canonical point is the differential oracle.
pub mod ipc2221;
/// Skin-effect (`δ = √(2ρ/(ωμ))`) and proximity/film-effect AC resistance penalty so a track's
/// copper utilisation regime (thin / partial-skin / deep-skin) can be selected at design time.
pub mod skin_and_film;
/// DC track resistance `R = ρ·L/(W·t)` plus the SOLE Tier-2 upstream enabler for both
/// `thermal::joule_source` and `pdn::ir_drop` (Phase 3a cut-over kept this here).
pub mod track_resistance;
/// Annular ring floor (mm around a via drill) plus through-hole aspect-ratio check (drill length
/// vs drill diameter); the manufacturing-readiness gates a fab rejects first.
pub mod via_mechanics;

#[cfg(test)]
mod tests;

pub use electromigration::*;
pub use ipc2221::*;
pub use skin_and_film::*;
pub use track_resistance::*;
pub use via_mechanics::*;
