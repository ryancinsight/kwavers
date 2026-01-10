// log/mod.rs
use log::{LevelFilter, Record};

pub mod file;

pub fn init_logging() -> Result<(), Box<dyn std::error::Error>> {
    let logger = file::configure_logger()?;
    log::set_logger(Box::leak(logger))?;
    log::set_max_level(LevelFilter::Info);
    Ok(())
}

#[must_use]
pub fn format_record(record: &Record) -> String {
    format!(
        "[{}] {}:{} - {}",
        record.level(),
        record.file().unwrap_or("unknown"),
        record.line().unwrap_or(0),
        record.args()
    )
}
