use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::core::log::file::CombinedLogger;
use log::{Level, Record};
use std::fs::{self, File};
use log::Log;

fn logging_benchmark(c: &mut Criterion) {
    let file_path = "benchmark_log.txt";
    // Ensure cleanup from previous runs
    let _ = fs::remove_file(file_path);

    let file = File::create(file_path).unwrap();
    let logger = CombinedLogger::new(false, file);

    let record = Record::builder()
        .args(format_args!("This is a benchmark log message"))
        .level(Level::Info)
        .target("benchmark")
        .file(Some("benchmark.rs"))
        .line(Some(1))
        .module_path(Some("benchmark"))
        .build();

    c.bench_function("log_message", |b| {
        b.iter(|| {
            logger.log(black_box(&record));
        })
    });

    let _ = fs::remove_file(file_path);
}

criterion_group!(benches, logging_benchmark);
criterion_main!(benches);
