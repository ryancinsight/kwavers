use crate::core::error::KwaversResult;
use ndarray::Array4;
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{RealtimeImagingPipeline, StreamingConfig, StreamingDataSource};

impl StreamingDataSource {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            generation_thread: None,
            stop_signal: Arc::new(Mutex::new(false)),
        }
    }

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

    fn generate_rf_frame(config: &StreamingConfig) -> Array4<f32> {
        let (n_tx, n_rx, n_samples, n_frames) = config.frame_size;
        let mut rf_data = Array4::zeros((n_tx, n_rx, n_samples, n_frames));

        use ndarray::Axis;

        rf_data
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(tx, mut tx_slice)| {
                tx_slice
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(rx, mut rx_slice)| {
                        let mut rng = rand::thread_rng();
                        let delay = (tx + rx) as f64 * 1e-7;

                        for sample in 0..n_samples {
                            let time = sample as f64 * 1e-8;

                            let signal = config.signal_amplitude
                                * (-0.5 * ((time - delay) * 5e7).powi(2)).exp()
                                * ((time - delay) * 3e7).cos();

                            for frame in 0..n_frames {
                                let noise = config.noise_level * rng.gen::<f64>();
                                rx_slice[[sample, frame]] = (signal + noise) as f32;
                            }
                        }
                    });
            });

        rf_data
    }
}
