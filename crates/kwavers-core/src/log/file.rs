// log/file.rs
use crate::log::format_record;
use log::{Level, LevelFilter, Log, Metadata, Record};
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
                if record.level() <= Level::Warn {
                    let _ = file.flush();
                }
            }
            if self.console {
                // This type is the console sink itself -- writing the record to
                // stdout is its contract, not incidental output from library
                // code. Ratchet: whether a log sink belongs on stdout rather
                // than stderr is tracked separately as KW-CORE-LOG-1.
                #[expect(clippy::print_stdout, reason = "console log sink")]
                {
                    println!("{message}");
                }
            }
        }
    }

    fn flush(&self) {
        if let Ok(mut file) = self.file.lock() {
            let _ = file.flush();
        }
    }
}

/// Configure logger.
/// # Errors
/// - Propagates any I/O error returned by called functions.
///
pub fn configure_logger() -> io::Result<Box<CombinedLogger>> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("kwavers.log")?;
    Ok(Box::new(CombinedLogger::new(true, file)))
}
