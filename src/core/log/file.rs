// log/file.rs
use crate::core::log::format_record;
use log::{LevelFilter, Log, Metadata, Record};
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::sync::Mutex;

#[derive(Debug)]
pub struct CombinedLogger {
    console: bool,
    file: Mutex<BufWriter<std::fs::File>>,
}

impl CombinedLogger {
    #[must_use]
    pub fn new(console: bool, file: std::fs::File) -> Self {
        Self {
            console,
            file: Mutex::new(BufWriter::new(file)),
        }
    }
}

impl Log for CombinedLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= LevelFilter::Debug
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let message = format_record(record);
            if let Ok(mut file) = self.file.lock() {
                let _ = writeln!(file, "{message}");
            }
            if self.console {
                println!("{message}");
            }
        }
    }

    fn flush(&self) {
        if let Ok(mut file) = self.file.lock() {
            let _ = file.flush();
        }
    }
}

pub fn configure_logger() -> io::Result<Box<CombinedLogger>> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("kwavers.log")?;
    Ok(Box::new(CombinedLogger::new(true, file)))
}
