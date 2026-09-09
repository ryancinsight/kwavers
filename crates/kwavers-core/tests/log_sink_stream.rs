//! The console sink's stream is a decision, so it is asserted rather than
//! assumed.
//!
//! `CombinedLogger` wrote console records to stdout, where they interleaved
//! with whatever a run actually produced. Moving them to stderr is only worth
//! doing if it stays done, and nothing else in the suite would notice it
//! drifting back: `eprintln!` cannot be captured from inside the process that
//! calls it, so the assertion needs a child whose two streams are separate
//! pipes.

use kwavers_core::log::file::CombinedLogger;
use log::{Level, Log, Record};
use std::process::Command;

/// Set for the child, which emits one record and exits.
const CHILD_MARKER: &str = "KWAVERS_LOG_SINK_STREAM_CHILD";

/// Distinctive enough that finding it in a stream is not a coincidence.
const PROBE_RECORD: &str = "kwavers-console-sink-stream-probe";

#[test]
fn emits_one_console_record_as_a_child() {
    if std::env::var(CHILD_MARKER).is_err() {
        // The parent case is the real test below; as a direct run this is a
        // no-op rather than a failure.
        return;
    }

    let log_path =
        std::env::temp_dir().join(format!("kwavers-log-sink-probe-{}.log", std::process::id()));
    let file = std::fs::File::create(&log_path).expect("probe log file");
    let logger = CombinedLogger::new(true, file);

    logger.log(
        &Record::builder()
            .args(format_args!("{PROBE_RECORD}"))
            .level(Level::Info)
            .build(),
    );
    logger.flush();

    let _ = std::fs::remove_file(&log_path);
}

#[test]
fn console_records_reach_stderr_and_never_stdout() {
    let child = Command::new(std::env::current_exe().expect("test binary path"))
        .args([
            "emits_one_console_record_as_a_child",
            "--exact",
            "--nocapture",
        ])
        .env(CHILD_MARKER, "1")
        .output()
        .expect("run the child test binary");

    let stdout = String::from_utf8_lossy(&child.stdout);
    let stderr = String::from_utf8_lossy(&child.stderr);

    assert!(
        stderr.contains(PROBE_RECORD),
        "the console sink must write to stderr; stderr was:\n{stderr}"
    );
    assert!(
        !stdout.contains(PROBE_RECORD),
        "a log record on stdout corrupts whatever the run itself prints; \
         stdout was:\n{stdout}"
    );
}
