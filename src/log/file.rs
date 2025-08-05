// log/file.rs
use crate::log::format_record;
use log::{Log, Metadata, Record, LevelFilter};
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::sync::Mutex;

pub struct CombinedLogger {
    console: bool,
    file: Mutex<std::fs::File>,
}

impl CombinedLogger {
    pub fn new(console: bool, file: std::fs::File) -> Self {
        Self {
            console,
            file: Mutex::new(file),
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
            let mut file = self.file.lock().unwrap();
            let _ = writeln!(file, "{}", message);
            if self.console {
                println!("{}", message);
            }
        }
    }

    fn flush(&self) {
        let _ = self.file.lock().unwrap().flush();
    }
}

pub fn configure_logger() -> io::Result<Box<CombinedLogger>> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("kwavers.log")?;
    Ok(Box::new(CombinedLogger::new(true, file)))
}
