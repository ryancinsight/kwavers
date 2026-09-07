//! Rejection assertions shared by the workspace's test modules.
//!
//! A bare `is_err` assertion passes whenever a call fails for *any* reason, so
//! a test written for one precondition also passes when an unrelated guard
//! fires first — an assertion that cannot fail on the defect it was written
//! for. The measured instance: an AVX-512 constructor rejecting both a
//! too-small grid and an unsupported host returned different variants, and the
//! dimension test passed on every machine without AVX-512.
//!
//! These helpers make the claim value-semantic: the rejection must be the
//! expected variant and must name its cause. They live here rather than in
//! each crate because eleven copies of one helper is the duplication this
//! module exists to avoid; the `test-util` feature keeps them out of the
//! default build.

use crate::error::{KwaversError, KwaversResult};

/// Assert a rejection is [`KwaversError::InvalidInput`] and names its cause.
///
/// # Panics
///
/// Panics when `result` is `Ok`, when the error is a different variant, or
/// when its message does not contain `fragment`.
#[track_caller]
pub fn assert_invalid_input<T: std::fmt::Debug>(result: KwaversResult<T>, fragment: &str) {
    match result {
        Err(KwaversError::InvalidInput(message)) => assert!(
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
/// The variant-checked [`assert_invalid_input`] is preferred where the contract
/// is `InvalidInput`. This is its counterpart for contracts that reject through
/// another typed variant (`Validation`, `InternalError`): the claim stays
/// value-semantic — the error must name *this* cause — without pinning a
/// variant the contract does not fix.
///
/// # Panics
///
/// Panics when `result` is `Ok`, or when the rendered error does not contain
/// `fragment`.
#[track_caller]
pub fn assert_rejects<T: std::fmt::Debug, E: std::fmt::Display>(
    result: Result<T, E>,
    fragment: &str,
) {
    match result {
        Err(error) => {
            let rendered = error.to_string();
            assert!(
                rendered.contains(fragment),
                "rejection must name its cause: expected a message containing \
                 {fragment:?}, got {rendered:?}"
            );
        }
        Ok(value) => panic!("expected a rejection naming {fragment:?}, got Ok({value:?})"),
    }
}
