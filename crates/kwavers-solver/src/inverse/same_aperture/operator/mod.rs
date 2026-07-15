//! Matrix-free finite-frequency same-aperture operators.
//!
//! # Performance contract
//!
//! `FiniteFrequencyOperator` precomputes one `PitchCatchRow` or `PassiveRow`
//! record per output row at construction so the hot `matvec`, `t_matvec`,
//! `normal_diag`, and `materialize` loops never recompute the row index
//! `divmod`, the source/receiver pair, the angular wavenumber, or the
//! frequency-MHz factor on a per-cell basis. Inverse row norms are cached
//! alongside the row norms so the inner loops never recompute `1 / norm`.
//! Outer loops over rows or columns dispatch through Moirai for cache-aware
//! parallelism on the SPD normal equations driven by PCG.
//!
//! # Module layout
//!
//! - [`types`] — the public operator struct, the row-spec enum, and the
//!   `PitchCatchRow` / `PassiveRow` value-types with their `unscaled_value`
//!   closures.
//! - [`linear_op`] — the [`LinearOperator`] impl on `FiniteFrequencyOperator`.
//! - [`rows`] — row-spec construction (`pitch_catch_rows`, `passive_rows`),
//!   per-row norms, and row-writers used by `materialize`.
//! - [`dot`] — matvec / t_matvec / normal-diag inner kernels and the
//!   `column_lookup` / `scaled_input` / `distance` helpers reused by every
//!   kernel above.
//!
//! [`LinearOperator`]: super::linear_operator::LinearOperator

mod dot;
mod linear_op;
mod rows;
mod types;

pub use types::FiniteFrequencyOperator;
