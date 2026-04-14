/// Create a named span for timing a block of work.
pub fn timed_span(name: &'static str) -> tracing::span::EnteredSpan {
    tracing::info_span!("{}", name).entered()
}
