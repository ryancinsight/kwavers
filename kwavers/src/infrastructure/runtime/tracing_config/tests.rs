use tracing::{debug, info, warn};

use super::{init_tracing, init_tracing_production, timed_span};

#[test]
fn tracing_initialization_is_idempotent() {
    init_tracing();
    init_tracing_production();

    info!("Test logging initialized");
    debug!(value = 42, "Debug value");
    warn!("Warning message");
}

#[test]
fn timed_span_enters_without_panicking() {
    init_tracing();
    let _span = timed_span("test_operation");
    std::thread::sleep(std::time::Duration::from_millis(10));
}
