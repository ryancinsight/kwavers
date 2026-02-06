use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::core::log::file::CombinedLogger;
use log::Log;
use log::{Level, Record};
use std::fs::{self, File};

fn logging_benchmark(c: &mut Criterion) {
    let file_path = "benchmark_log.txt";
    // Ensure cleanup from previous runs
    let _ = fs::remove_file(file_path);

    // Setup for Info logs (buffered)
    let file_info = File::create(file_path).unwrap();
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

    // Setup for Error logs (flushed)
    // Re-create file/logger to reset state, although Mutex protects it.
    let _ = fs::remove_file(file_path);
    let file_error = File::create(file_path).unwrap();
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

    let _ = fs::remove_file(file_path);
}

criterion_group!(benches, logging_benchmark);
criterion_main!(benches);
