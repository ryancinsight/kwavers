//! Deterministic test span capture, including isolated Miri execution.

pub(in super::super) fn subscriber() -> impl tracing::Subscriber + Send + Sync {
    tracing_subscriber::fmt()
        .with_test_writer()
        .without_time()
        .with_ansi(false)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .finish()
}
