use super::*;
use leto::Array4 as LetoArray4;

#[test]
fn test_pipeline_creation() {
    let config = RealtimePipelineConfig {
        target_fps: 30.0,
        max_latency_ms: 33.0,
        buffer_size: 10,
        gpu_accelerated: false,
        adaptive_processing: true,
        streaming_mode: false,
    };

    let _pipeline = RealtimeImagingPipeline::new(config).unwrap();
}

#[test]
fn test_pipeline_start_stop() {
    let config = RealtimePipelineConfig {
        target_fps: 10.0,
        max_latency_ms: 100.0,
        buffer_size: 5,
        gpu_accelerated: false,
        adaptive_processing: false,
        streaming_mode: false,
    };

    let mut pipeline = RealtimeImagingPipeline::new(config).unwrap();

    pipeline.start().unwrap();
    assert_eq!(pipeline.state(), PipelineState::Running);

    pipeline.stop().unwrap();
    assert_eq!(pipeline.state(), PipelineState::Stopped);
}

#[test]
fn test_data_submission() {
    let config = RealtimePipelineConfig {
        target_fps: 10.0,
        max_latency_ms: 100.0,
        buffer_size: 5,
        gpu_accelerated: false,
        adaptive_processing: false,
        streaming_mode: false,
    };

    let mut pipeline = RealtimeImagingPipeline::new(config).unwrap();
    pipeline.start().unwrap();

    let rf_data = LetoArray4::from_elem([4, 32, 1024, 1], 1.0);
    pipeline.submit_rf_data(rf_data).unwrap();

    pipeline.stop().unwrap();
}

#[test]
fn test_frame_processing() {
    let config = RealtimePipelineConfig {
        target_fps: 10.0,
        max_latency_ms: 100.0,
        buffer_size: 5,
        gpu_accelerated: false,
        adaptive_processing: false,
        streaming_mode: false,
    };

    let mut pipeline = RealtimeImagingPipeline::new(config).unwrap();
    pipeline.start().unwrap();

    let rf_data = LetoArray4::from_elem([2, 16, 512, 1], 1.0);
    pipeline.submit_rf_data(rf_data).unwrap();
    pipeline.process_pipeline().unwrap();

    let processed = pipeline.get_processed_frame();

    let frame = processed.unwrap();
    assert_eq!(frame.shape(), [16, 512, 1]);
    assert!(frame.iter().all(|&value| (0.0..=1.0).contains(&value)));

    pipeline.stop().unwrap();
}

#[test]
fn test_streaming_data_source() {
    let stream_cfg = StreamingConfig {
        frame_rate: 50.0,
        frame_size: (2, 8, 128, 1),
        noise_level: 0.05,
        signal_amplitude: 1.0,
        source_id: "test".to_string(),
        sample_rate: 40_000_000.0,
    };

    let mut data_source = StreamingDataSource::new(stream_cfg);

    let mut pipeline = RealtimeImagingPipeline::new(RealtimePipelineConfig {
        target_fps: 60.0,
        max_latency_ms: 50.0,
        buffer_size: 8,
        gpu_accelerated: false,
        adaptive_processing: false,
        streaming_mode: true,
    })
    .unwrap();

    pipeline.start().unwrap();
    let pipeline_arc = std::sync::Arc::new(std::sync::Mutex::new(pipeline));
    data_source
        .start_streaming(std::sync::Arc::clone(&pipeline_arc))
        .unwrap();

    std::thread::sleep(std::time::Duration::from_millis(100));

    data_source.stop_streaming().unwrap();
    let mut pipeline_lock = pipeline_arc.lock().unwrap_or_else(|e| e.into_inner());
    let (input_util, _output_util) = pipeline_lock.buffer_utilization();
    assert!(input_util > 0.0);

    pipeline_lock.stop().unwrap();
}
