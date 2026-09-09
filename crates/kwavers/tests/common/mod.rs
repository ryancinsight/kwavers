//! Shared admission for the GPU-gated integration tests.
//!
//! A GPU test has two legitimate outcomes on a host without hardware -- run, or
//! skip -- and exactly one illegitimate one: skip because the GPU broke. The
//! hosted runners have no adapter and so take the skip path on every run, which
//! is what made the difference invisible: a driver fault or a regression inside
//! context construction read exactly like absent hardware and the suite
//! reported green.
//!
//! `kwavers_core::error::KwaversError::is_gpu_absent` is the oracle. It is true
//! only for the typed absence the acquisition path reports when no compatible
//! adapter exists; every other acquisition failure describes hardware that is
//! present and not working.

use kwavers_core::error::KwaversResult;

/// The acquired value, or `None` when this host genuinely has no GPU.
///
/// # Panics
///
/// When acquisition failed for any reason other than absent hardware. That is
/// the point: a present adapter that will not initialize is a defect, and a
/// test that skips over it proves nothing.
pub fn or_skip<T>(what: &str, outcome: KwaversResult<T>) -> Option<T> {
    match outcome {
        Ok(value) => Some(value),
        Err(error) if error.is_gpu_absent() => {
            eprintln!("{what}: no compatible GPU adapter on this host, skipping");
            None
        }
        Err(error) => panic!(
            "{what}: a GPU adapter is present but acquisition failed, which is a \
             defect rather than absent hardware: {error}"
        ),
    }
}
