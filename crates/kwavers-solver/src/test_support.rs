//! Test-only diagnostics support.
//!
//! Test modules and test-only files report through [`tracing`] instead of
//! `println!` so the crate holds the workspace clippy floor
//! (`clippy::print_stdout`) under `--all-targets`. The macro mirrors
//! `println!`'s shape (`test_info!("...: {}", x)`), so a conversion is a
//! rename, not a rewrite; it emits at `INFO` through the global subscriber
//! and is free when no subscriber collects test output.

/// Test-progress diagnostic at `INFO` level.
macro_rules! test_info {
    ($($arg:tt)*) => {
        tracing::info!($($arg)*)
    };
}

pub(crate) use test_info;

/// Assert a rejection is the expected error variant and names its cause.
///
/// A bare `is_err` assertion passes whenever the call fails for *any* reason,
/// so a test written to check a dimension bound also passes when the host
/// lacks the ISA the constructor probes for. That is an assertion which cannot
/// fail on the defect it was written for. This checks both the variant and a
/// distinguishing fragment of its message, so a rejection arriving through a
/// different path fails the test.
///
/// # Panics
///
/// Panics when `result` is `Ok`, when the error is a different variant, or
/// when its rendered message does not contain `fragment`.
#[track_caller]
pub(crate) fn assert_invalid_input<T: std::fmt::Debug>(
    result: kwavers_core::error::KwaversResult<T>,
    fragment: &str,
) {
    match result {
        Err(kwavers_core::error::KwaversError::InvalidInput(message)) => assert!(
            message.contains(fragment),
            "InvalidInput must name the violated constraint: expected a message \
             containing {fragment:?}, got {message:?}"
        ),
        Err(other) => panic!("expected InvalidInput({fragment:?}), got {other:?}"),
        Ok(value) => panic!("expected InvalidInput({fragment:?}), got Ok({value:?})"),
    }
}

/// Assert a rejection carries the expected cause in its rendered message.
///
/// The variant-checked [`assert_invalid_input`] is preferred where the
/// contract is `InvalidInput`. This is its counterpart for contracts that
/// reject through another typed variant (`Validation`, `InternalError`): the
/// claim is still value-semantic — the error must name *this* cause — without
/// pinning a variant the contract does not fix.
///
/// # Panics
///
/// Panics when `result` is `Ok`, or when the rendered error does not contain
/// `fragment`.
#[track_caller]
pub(crate) fn assert_rejects<T: std::fmt::Debug, E: std::fmt::Display>(
    result: Result<T, E>,
    fragment: &str,
) {
    match result {
        Err(error) => {
            let rendered = error.to_string();
            assert!(
                rendered.contains(fragment),
                "rejection must name its cause: expected a message containing                  {fragment:?}, got {rendered:?}"
            );
        }
        Ok(value) => panic!("expected a rejection naming {fragment:?}, got Ok({value:?})"),
    }
}
