// log/file.rs
use crate::log::format_record;
use log::{Level, LevelFilter, Log, Metadata, Record};
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::sync::Mutex;

/// A [`Log`] sink that writes every record to a file and, optionally, mirrors
/// it to the console.
///
/// # Console stream
///
/// The console mirror writes to **stderr**, not stdout. Diagnostics on stdout
/// interleave with a program's actual output, so a caller that pipes a kwavers
/// binary cannot separate the two; stderr is the conventional stream for log
/// sinks for exactly that reason. It also agrees with the workspace's other
/// logging entry point: the `kwavers` facade's `init_logging` installs
/// `env_logger`, which writes to stderr by default. Before this, the two
/// disagreed about where diagnostics belong.
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
                // Stderr, not stdout: see the type's `# Console stream` note.
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
