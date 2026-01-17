#[cfg(feature = "gpu")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(feature = "gpu")]
use kwavers::gpu::pipeline::{RealtimeImagingPipeline, RealtimePipelineConfig};
#[cfg(feature = "gpu")]
use ndarray::Array4;

#[cfg(feature = "gpu")]
fn bench_pipeline_process(c: &mut Criterion) {
    let config = RealtimePipelineConfig {
        target_fps: 30.0,
        max_latency_ms: 33.0,
        buffer_size: 10,
        gpu_accelerated: false,
        adaptive_processing: false,
        streaming_mode: false,
    };

    let mut pipeline = RealtimeImagingPipeline::new(config).unwrap();
    pipeline.start().unwrap();

    // Create a dummy RF data frame
    // Dimensions: (Tx, Rx, Samples, Frames)
    // Smallish dimensions to keep benchmark fast but realistic enough to trigger the issue
    // 4 Tx, 64 Rx, 1024 Samples, 1 Frame
    let rf_data = Array4::<f32>::zeros((4, 64, 1024, 1));

    c.bench_function("process_pipeline_frame", |b| {
        b.iter(|| {
            // We need to submit data first
            pipeline.submit_rf_data(rf_data.clone()).unwrap();

            // Then process
            pipeline.process_pipeline().unwrap();

            // Consume output to prevent buffer full
            pipeline.get_processed_frame();
        })
    });

    pipeline.stop().unwrap();
}

#[cfg(feature = "gpu")]
criterion_group!(benches, bench_pipeline_process);
#[cfg(feature = "gpu")]
criterion_main!(benches);

#[cfg(not(feature = "gpu"))]
fn main() {}
