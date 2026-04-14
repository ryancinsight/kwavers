use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialize tracing with development-friendly formatting.
pub fn init_tracing() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("kwavers=info"));

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().pretty())
        .try_init();
}

/// Initialize tracing with compact production formatting.
pub fn init_tracing_production() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("kwavers=warn"));

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().compact())
        .try_init();
}
