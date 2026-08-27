#![doc = include_str!("../README.md")]

/// Internal re-exports for macro use.
#[doc(hidden)]
pub mod __private {
    pub use leto::SliceArg;
}

/// ndarray `s!` macro replacement for leto `SliceArg`.
///
/// Accepts native Rust range expressions (`start..end`, `..end`, `start..`, `..`)
/// and integer indices, separated by commas. Steps use semicolons: `start..end;step`.
/// Returns `[leto::SliceArg; N]`.
// Rule 3 -- a step on an element that is not the last -- exists so the grammar
// is symmetric: a caller who may write `s![a..b; 2]` may also write
// `s![a..b; 2, c..d]`. No call site in this crate needs it yet, so the lib
// build sees it as unused. `slice_macro_tests` does use it, which is why the
// expectation is scoped to builds where those tests are absent; unscoped, it
// would report itself unfulfilled against the test target.
#[cfg_attr(
    not(test),
    expect(
        unused_macro_rules,
        reason = "grammar completeness, covered by slice_macro_tests"
    )
)]
macro_rules! s {
    // Internal: convert expression to SliceArg via From trait.
    (@as_slicearg $r:expr) => {
        <$crate::__private::SliceArg as ::std::convert::From<_>>::from($r)
    };

    // Final item with step: expr;step
    (@parse [$($stack:tt)*] $r:expr; $s:expr) => {
        [$($stack)* s!(@as_slicearg $r).step($s as isize)]
    };
    // Not-final item with step: expr;step,
    (@parse [$($stack:tt)*] $r:expr; $s:expr, $($t:tt)*) => {
        s!(@parse [$($stack)* s!(@as_slicearg $r).step($s as isize),] $($t)*)
    };
    // Final item without step: expr
    (@parse [$($stack:tt)*] $r:expr) => {
        [$($stack)* s!(@as_slicearg $r)]
    };
    // Not-final item without step: expr,
    (@parse [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        s!(@parse [$($stack)* s!(@as_slicearg $r),] $($t)*)
    };

    // Entry point: delegate to internal parser.
    ($($t:tt)*) => {
        s!(@parse [] $($t)*)
    };
}

#[cfg(test)]
mod slice_macro_tests {
    use crate::__private::SliceArg;

    /// `start..end` with no step.
    const fn range(start: isize, end: isize, step: isize) -> SliceArg {
        SliceArg::Range {
            start: Some(start),
            end: Some(end),
            step,
        }
    }

    #[test]
    fn a_single_range_carries_a_unit_step() {
        assert_eq!(s![0..10], [range(0, 10, 1)]);
    }

    #[test]
    fn several_elements_keep_their_order() {
        assert_eq!(
            s![0..10, .., 3],
            [range(0, 10, 1), SliceArg::All, SliceArg::Index(3)]
        );
    }

    #[test]
    fn a_step_applies_to_the_last_element() {
        assert_eq!(s![0..10; 2], [range(0, 10, 2)]);
    }

    /// The stepped-and-followed arm, which no call site in this crate uses.
    ///
    /// Without this test the rule is unreachable, `-D unused` reports it, and
    /// the obvious response -- deleting it -- would leave the macro's grammar
    /// asymmetric: a step would be accepted in the last position and rejected
    /// in every other, for no reason a caller could infer from the error.
    #[test]
    fn a_step_applies_to_an_element_that_is_not_the_last() {
        assert_eq!(s![0..10; 2, 1..5], [range(0, 10, 2), range(1, 5, 1)]);
        assert_eq!(
            s![0..10; 2, 1..8; 3, 4],
            [range(0, 10, 2), range(1, 8, 3), SliceArg::Index(4)]
        );
    }

    #[test]
    fn open_ended_ranges_leave_the_open_side_unset() {
        assert_eq!(
            s![..5, 2..],
            [
                SliceArg::Range {
                    start: None,
                    end: Some(5),
                    step: 1
                },
                SliceArg::Range {
                    start: Some(2),
                    end: None,
                    step: 1
                },
            ]
        );
    }
}

// src/solver/mod.rs
// Clean module structure focusing only on the plugin-based architecture

// Hierarchical solver module structure
pub mod backend;
pub mod config;
pub mod factory;
pub mod feature;
pub mod forward;

pub mod analytical;
pub mod integration;
pub mod interface;
pub mod inverse;
pub mod krylov;
pub mod multiphysics;
pub mod plugin;
pub mod utilities;
pub mod workspace;

// Safety: single source of truth for the Zip-migration layout preconditions
// (assert standard-layout + assert `as_slice{_mut,}` unwrap). Consolidates the
// 30 inline assert sites across the 6 migrated files (struct_impl.rs +
// diffusion.rs + model_impl.rs (fixup) + nonlinear.rs + operator_splitting/mod.rs
// + rhs.rs) into a single DRY helper.
pub mod safety;

// Re-export ScratchArena trait for ergonomic use across solver crates
pub use workspace::ScratchArena;

// Re-export canonical error types for doc-link resolution across the solver crate
pub use kwavers_core::error::KwaversError;
pub use kwavers_core::error::KwaversResult;

// Re-export field indices from the single source of truth
pub use kwavers_field::indices::{
    PRESSURE_IDX as P_IDX, STRESS_XX_IDX as SXX_IDX, STRESS_XY_IDX as SXY_IDX,
    STRESS_XZ_IDX as SXZ_IDX, STRESS_YY_IDX as SYY_IDX, STRESS_YZ_IDX as SYZ_IDX,
    STRESS_ZZ_IDX as SZZ_IDX, TOTAL_FIELDS, VX_IDX, VY_IDX, VZ_IDX,
};

// Re-export commonly used types from hierarchical modules
pub use config::{FftBackend, SolverConfiguration, SolverType};
pub use forward::fdtd::FdtdSolver;
pub use forward::hybrid::HybridSolver;
pub use forward::plugin_based::PluginBasedSolver;
pub use forward::pstd::PSTDSolver;
pub use interface::Solver;
pub use inverse::{
    ReconstructionConfig, Reconstructor, TimeReversalConfig, TimeReversalReconstructor,
};
pub use multiphysics::{
    CoupledMultiPhysicsSolver, FieldCouplingStrategy, MultiphysicsFieldCoupler,
};

// Constants module remains at root level for easy access
pub mod constants;

// Progress reporting - use types from interface module
pub use interface::{
    ConsoleProgressReporter, FieldsSummary, ProgressData, ProgressReporter, ProgressUpdate,
};

pub mod geometry;
pub use geometry::SolverGeometry;

// Backward-compatible re-exports for commonly used submodules
pub use forward::fdtd;
pub use forward::hybrid;
pub use forward::plugin_based;
pub use forward::pstd;
pub use integration::time_integration;

// Concrete GPU execution lives in `kwavers-gpu`; this crate owns only the
// solver-facing compute backend trait surface.
pub use inverse::reconstruction;
pub use inverse::time_reversal;
pub use utilities::amr;
pub mod validation;
