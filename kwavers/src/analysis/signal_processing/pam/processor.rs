use super::config::PAMConfig;
use crate::core::error::KwaversResult;
use crate::math::fft::fft_1d_array;
use ndarray::Array3;

pub struct PAMProcessor {
    pub(super) config: PAMConfig,
}

impl std::fmt::Debug for PAMProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PAMProcessor")
            .field("config", &self.config)
            .finish()
    }
}

impl PAMProcessor {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: PAMConfig) -> KwaversResult<Self> {
        Ok(Self { config })
    }
    /// Process.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn process(&mut self, beamformed_data: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let shape = beamformed_data.shape();
        let (nx, ny, nt) = (shape[0], shape[1], shape[2]);

        let mut cavitation_map = Array3::zeros((nx, ny, self.config.frequency_bands.len()));

        for ix in 0..nx {
            for iy in 0..ny {
                let time_series: Vec<f64> =
                    (0..nt).map(|it| beamformed_data[[ix, iy, it]]).collect();

                let spectrum = self.compute_spectrum(&time_series)?;

                for (band_idx, &(f_min, f_max)) in self.config.frequency_bands.iter().enumerate() {
                    let power = self.integrate_band_power(&spectrum, f_min, f_max);

                    if power > self.config.threshold {
                        cavitation_map[[ix, iy, band_idx]] = power;
                    }
                }

                if self.config.enable_harmonic_analysis {
                    self.analyze_harmonics(&spectrum, ix, iy, &mut cavitation_map)?;
                }
            }
        }

        Ok(cavitation_map)
    }

    fn compute_spectrum(&mut self, time_series: &[f64]) -> KwaversResult<Vec<f64>> {
        let spectrum = fft_1d_array(&ndarray::Array1::from_vec(time_series.to_vec()))
            .iter()
            .map(|c| c.re.hypot(c.im))
            .collect();
        Ok(spectrum)
    }

    fn integrate_band_power(&self, spectrum: &[f64], f_min: f64, f_max: f64) -> f64 {
        let n = spectrum.len();
        if n == 0 {
            return 0.0;
        }

        let f_s = self.config.beamforming.core.sampling_frequency;
        if !f_s.is_finite() || f_s <= 0.0 {
            return 0.0;
        }

        if !f_min.is_finite() || !f_max.is_finite() || f_min < 0.0 || f_max < 0.0 || f_min > f_max {
            return 0.0;
        }

        let idx_min = ((f_min * n as f64) / f_s).floor().max(0.0) as usize;
        let idx_max = ((f_max * n as f64) / f_s).ceil().max(0.0) as usize;

        let lo = idx_min.min(n - 1);
        let hi = idx_max.min(n);
        if lo >= hi {
            return 0.0;
        }

        spectrum[lo..hi].iter().sum()
    }

    fn analyze_harmonics(
        &self,
        spectrum: &[f64],
        ix: usize,
        iy: usize,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let fundamental_idx = self.find_fundamental(spectrum);

        for harmonic in 2..5 {
            let harmonic_idx = fundamental_idx * harmonic;
            if harmonic_idx < spectrum.len() {
                let harmonic_power = spectrum[harmonic_idx];
                if harmonic_power > self.config.threshold * 0.5 && harmonic - 2 < output.shape()[2]
                {
                    output[[ix, iy, harmonic - 2]] += harmonic_power;
                }
            }
        }

        Ok(())
    }

    fn find_fundamental(&self, spectrum: &[f64]) -> usize {
        spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or(0, |(idx, _)| idx)
    }
    /// Config.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn config(&self) -> &PAMConfig {
        &self.config
    }
    /// Set config.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn set_config(&mut self, config: PAMConfig) -> KwaversResult<()> {
        self.config = config;
        Ok(())
    }
}
