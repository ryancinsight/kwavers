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
