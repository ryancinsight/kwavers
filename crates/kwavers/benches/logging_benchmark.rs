use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers_core::log::file::CombinedLogger;
use log::Log;
use log::{Level, Record};
use std::fs::File;

fn logging_benchmark(c: &mut Criterion) {
    let run_directory = tempfile::tempdir().expect("benchmark tempdir must be available");

    // Setup for Info logs (buffered)
    let file_info =
        File::create(run_directory.path().join("info.log")).expect("benchmark log must be created");
    let logger_info = CombinedLogger::new(false, file_info);

    let record_info = Record::builder()
        .args(format_args!("This is a benchmark log message"))
        .level(Level::Info)
        .target("benchmark")
        .file(Some("benchmark.rs"))
        .line(Some(1))
        .module_path(Some("benchmark"))
        .build();

    c.bench_function("log_message_info_buffered", |b| {
        b.iter(|| {
            logger_info.log(black_box(&record_info));
        })
    });

    // Use a separate file so buffered and flushed measurements share no state.
    let file_error = File::create(run_directory.path().join("error.log"))
        .expect("benchmark log must be created");
    let logger_error = CombinedLogger::new(false, file_error);

    let record_error = Record::builder()
        .args(format_args!("This is a benchmark error message"))
        .level(Level::Error)
        .target("benchmark")
        .file(Some("benchmark.rs"))
        .line(Some(2))
        .module_path(Some("benchmark"))
        .build();

    c.bench_function("log_message_error_flushed", |b| {
        b.iter(|| {
            logger_error.log(black_box(&record_error));
        })
    });
}

criterion_group!(benches, logging_benchmark);
criterion_main!(benches);
