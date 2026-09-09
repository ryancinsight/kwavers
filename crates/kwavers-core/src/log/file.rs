// log/file.rs
use crate::log::format_record;
use log::{Level, LevelFilter, Log, Metadata, Record};
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::sync::Mutex;

/// A logger that writes every record to a file and, optionally, to the
/// console.
///
/// The console stream is stderr. Diagnostics share a process with whatever the
/// program actually produces, and a run whose results are piped or redirected
/// must not have log records interleaved into that data. The buffering settles
/// it beyond convention: stdout is block-buffered when it is not a terminal, so
/// a run that aborts loses precisely the records explaining why, while stderr
/// is unbuffered and each record has reached the descriptor before the next
/// line runs. That also keeps console records ordered against the file sink,
/// which flushes on `Warn` and above.
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
                eprintln!("{message}");
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
