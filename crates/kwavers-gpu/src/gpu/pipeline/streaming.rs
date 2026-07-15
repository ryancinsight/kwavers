use kwavers_core::error::KwaversResult;
use leto::Array4 as LetoArray4;
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{RealtimeImagingPipeline, StreamingConfig, StreamingDataSource};

impl StreamingDataSource {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            generation_thread: None,
            stop_signal: Arc::new(Mutex::new(false)),
        }
    }
    /// Start streaming.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn start_streaming(
        &mut self,
        pipeline: Arc<Mutex<RealtimeImagingPipeline>>,
    ) -> KwaversResult<()> {
        let config = self.config.clone();
        let stop_signal = Arc::clone(&self.stop_signal);

        let handle = std::thread::spawn(move || {
            let frame_interval = Duration::from_secs_f64(1.0 / config.frame_rate);

            loop {
                let start_time = Instant::now();

                {
                    let stop = stop_signal.lock().unwrap_or_else(|e| e.into_inner());
                    if *stop {
                        break;
                    }
                }

                let rf_data = Self::generate_rf_frame(&config);

                {
                    let mut pipeline_lock = pipeline.lock().unwrap_or_else(|e| e.into_inner());
                    let _ = pipeline_lock.submit_rf_data(rf_data);
                }

                let elapsed = start_time.elapsed();
                if elapsed < frame_interval {
                    std::thread::sleep(frame_interval - elapsed);
                }
            }
        });

        self.generation_thread = Some(handle);
        Ok(())
    }
    /// Stop streaming.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn stop_streaming(&mut self) -> KwaversResult<()> {
        {
            let mut stop = self.stop_signal.lock().unwrap_or_else(|e| e.into_inner());
            *stop = true;
        }

        if let Some(handle) = self.generation_thread.take() {
            let _ = handle.join();
        }

        Ok(())
    }

    fn generate_rf_frame(config: &StreamingConfig) -> LetoArray4<f32> {
        let (n_tx, n_rx, n_samples, n_frames) = config.frame_size;
        let mut rf_data = LetoArray4::zeros([n_tx, n_rx, n_samples, n_frames]);
        let rx_frame_len = n_samples * n_frames;

        if let Some(values) = rf_data.as_slice_memory_order_mut() {
            for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                values,
                rx_frame_len,
                |tx_rx, rx_slice| {
                    let tx = tx_rx / n_rx;
                    let rx = tx_rx % n_rx;
                    fill_rf_rx_slice(config, tx, rx, n_samples, n_frames, rx_slice);
                },
            );
        }

        rf_data
    }
}

fn fill_rf_rx_slice(
    config: &StreamingConfig,
    tx: usize,
    rx: usize,
    n_samples: usize,
    n_frames: usize,
    rx_slice: &mut [f32],
) {
    let mut rng = rand::thread_rng();
    let delay = (tx + rx) as f64 * 1e-7;

    for sample in 0..n_samples {
        let time = sample as f64 * 1e-8;

        let signal = config.signal_amplitude
            * (-0.5 * ((time - delay) * 5e7).powi(2)).exp()
            * ((time - delay) * 3e7).cos();

        for frame in 0..n_frames {
            let noise = config.noise_level * rng.gen::<f64>();
            rx_slice[sample * n_frames + frame] = (signal + noise) as f32;
        }
    }
}
