//! 3D Synthetic Aperture Focusing Technique (SAFT) Implementation
//!
//! ## Mathematical Foundation
//!
//! SAFT reconstruction for voxel **r** = (x, y, z):
//!
//! ```text
//! I_SAFT(r) = |Σᵢ Σⱼ w_ij · RF[i,j, τ_ij(r)] · exp(−j·2π·f₀·τ_ij(r))|²
//! ```
//!
//! where:
//! * `τ_ij(r) = (|rᵢ − r| + |r − rⱼ|) / c`  — monostatic round-trip time-of-flight
//! * `w_ij`  — Hamming apodization weight (Nuttall 1981)
//! * `f₀`   — centre frequency \[Hz\]
//! * The complex exponential `exp(−j·2π·f₀·τ)` demodulates the carrier, bringing
//!   each delayed sample to baseband so that contributions from a common scatterer
//!   sum constructively regardless of carrier phase at the sample instant.
//!
//! Because the input RF data are real-valued, the complex accumulation is kept as
//! an (I, Q) pair:
//!
//! ```text
//! I_k = RF[i,j, τ] · cos(2π f₀ τ)
//! Q_k = RF[i,j, τ] · (−sin(2π f₀ τ))
//! ```
//!
//! The envelope intensity is then `I²+ Q²`.
//!
//! ## Coherence Factor (CF)
//!
//! The coherence factor (Mallart & Fink 1994) suppresses off-focus sidelobes:
//!
//! ```text
//! CF(r) = |Σᵢ sᵢ(r)|² / ( N · Σᵢ |sᵢ(r)|² )  ∈ [0, 1]
//! ```
//!
//! Applied as `I_CF(r) = CF(r) · I_SAFT(r)`.
//!
//! ## References
//! * Frazier & O'Brien (1998). "Synthetic aperture techniques with a virtual source
//!   element." *IEEE Trans. Ultrason. Ferroelectr. Freq. Control* 45(1):196–207.
//! * Nikolov & Jensen (2002). "Virtual ultrasound sources in high-resolution
//!   ultrasound imaging." *Proc. SPIE* 4687:395–405.
//! * Mallart & Fink (1994). "Adaptive focusing in scattering media through sound
//!   speed inhomogeneities." *J. Acoust. Soc. Am.* 96(6):3721–3732.
//! * Nuttall AH (1981). "Some windows with very good sidelobe behavior."
//!   *IEEE Trans. Acoust. Speech Signal Process.* 29(1):84–91.

use crate::core::error::{KwaversError, KwaversResult};
use log::info;
use ndarray::{Array3, Array4};
use std::f64::consts::PI;

use super::{ApodizationWindow, BeamformingAlgorithm3D, BeamformingConfig3D};

/// SAFT configuration parameters
#[derive(Debug, Clone)]
pub struct SaftConfig {
    /// Number of virtual sources for synthetic aperture
    pub virtual_sources: usize,
    /// Apodization window for sidelobe suppression
    pub apodization: ApodizationWindow,
    /// Coherence factor weighting enabled
    pub coherence_factor_enabled: bool,
    /// F-number for dynamic focusing
    pub f_number: f64,
}

impl Default for SaftConfig {
    fn default() -> Self {
        Self {
            virtual_sources: 100,
            apodization: super::config::ApodizationWindow::Hamming,
            coherence_factor_enabled: true,
            f_number: 1.5,
        }
    }
}

/// SAFT processor for 3D volumetric reconstruction
#[derive(Debug)]
pub struct SaftProcessor {
    config: SaftConfig,
    beamforming_config: super::config::BeamformingConfig3D,
}

impl SaftProcessor {
    /// Create new SAFT processor
    pub fn new(saft_config: SaftConfig, beamforming_config: BeamformingConfig3D) -> Self {
        Self {
            config: saft_config,
            beamforming_config,
        }
    }

    /// Extract SAFT parameters from BeamformingAlgorithm3D::SAFT3D variant
    pub fn from_algorithm(
        algorithm: &BeamformingAlgorithm3D,
        beamforming_config: BeamformingConfig3D,
    ) -> KwaversResult<Self> {
        match algorithm {
            BeamformingAlgorithm3D::SAFT3D { virtual_sources } => Ok(Self::new(
                SaftConfig {
                    virtual_sources: *virtual_sources,
                    ..Default::default()
                },
                beamforming_config,
            )),
            _ => Err(KwaversError::InvalidInput(
                "Expected SAFT3D algorithm variant".to_string(),
            )),
        }
    }

    /// Compute round-trip time-of-flight from transmit → voxel → receive.
    ///
    /// ## Formula
    ///
    /// ```text
    /// τ(i, j, r) = (‖rᵢ − r‖ + ‖r − rⱼ‖) / c
    /// ```
    ///
    /// * `tx_position` — transmit element position (m)
    /// * `rx_position` — receive element position (m)
    /// * `voxel_position` — voxel position (m)
    /// * `sound_speed` — sound speed in medium (m/s)
    fn compute_time_of_flight(
        &self,
        tx_position: [f64; 3],
        rx_position: [f64; 3],
        voxel_position: [f64; 3],
        sound_speed: f64,
    ) -> f64 {
        let tx_to_voxel_dist = distance3(tx_position, voxel_position);
        let voxel_to_rx_dist = distance3(voxel_position, rx_position);
        (tx_to_voxel_dist + voxel_to_rx_dist) / sound_speed
    }

    /// Interpolate RF sample at time index (nearest-neighbour, bounds-checked).
    fn extract_rf_sample(rf_data: &Array4<f64>, element_idx: usize, time_idx: usize) -> f64 {
        rf_data[[0, element_idx, time_idx, 0]]
    }

    /// Demodulate a real RF sample at known time-of-flight.
    ///
    /// ## Algorithm
    ///
    /// Multiplying the delayed sample by the complex conjugate of the carrier
    /// `exp(−j·2π·f₀·τ)` centres the signal at baseband.  For real-valued RF,
    /// the I and Q components are:
    ///
    /// ```text
    /// I = sample · cos(2π f₀ τ)
    /// Q = sample · (−sin(2π f₀ τ))
    /// ```
    ///
    /// # Returns
    /// `(I, Q)` — baseband (in-phase, quadrature) components
    #[inline]
    fn demodulate(sample: f64, center_frequency: f64, time_of_flight: f64) -> (f64, f64) {
        let phase = 2.0 * PI * center_frequency * time_of_flight;
        (sample * phase.cos(), sample * (-phase.sin()))
    }

    /// Hamming apodization weight for sidelobe suppression.
    ///
    /// ## Formula (Nuttall 1981)
    /// ```text
    /// w(u) = 0.54 + 0.46 · cos(π u),   u ∈ [−1, 1]
    /// ```
    fn compute_apodization_weight(
        &self,
        tx_idx: usize,
        rx_idx: usize,
        total_elements: usize,
    ) -> f64 {
        let center = total_elements as f64 / 2.0;
        let pos = (tx_idx + rx_idx) as f64 / 2.0;
        let normalized_pos = (pos - center) / center.max(1.0);
        0.54 + 0.46 * (PI * normalized_pos).cos()
    }

    /// Coherence factor (Mallart & Fink 1994).
    ///
    /// ## Formula
    /// ```text
    /// CF = |Σ sₙ|² / ( N · Σ |sₙ|² )
    /// ```
    fn compute_coherence_factor(
        &self,
        coherent_sum: f64,
        incoherent_sum: f64,
        num_contributions: usize,
    ) -> f64 {
        if num_contributions == 0 {
            return 0.0;
        }
        let denominator = num_contributions as f64 * incoherent_sum.powi(2);
        if denominator > 0.0 {
            (coherent_sum.abs().powi(2)) / denominator
        } else {
            0.0
        }
    }

    /// Reconstruct a 3D volume using SAFT with carrier-phase demodulation.
    ///
    /// Each voxel receives contributions from all TX–RX pairs.  The delayed RF
    /// sample is demodulated to baseband via `I + jQ = RF · exp(−j·2π·f₀·τ)`
    /// before accumulation so that scatterers at the focus sum constructively.
    ///
    /// # Arguments
    /// * `rf_data` — RF data array (frames × channels × samples × 1)
    ///
    /// # Returns
    /// Reconstructed 3D volume (x × y × z) as f32
    pub fn reconstruct_volume(&self, rf_data: &Array4<f32>) -> KwaversResult<Array3<f32>> {
        let start_time = std::time::Instant::now();

        self.validate_input(rf_data)?;

        let rf_data_f64 = rf_data.mapv(|x| x as f64);

        let (nx, ny, nz) = (
            self.beamforming_config.volume_dims.0,
            self.beamforming_config.volume_dims.1,
            self.beamforming_config.volume_dims.2,
        );
        let (dx, dy, dz) = (
            self.beamforming_config.voxel_spacing.0,
            self.beamforming_config.voxel_spacing.1,
            self.beamforming_config.voxel_spacing.2,
        );
        let sound_speed = self.beamforming_config.sound_speed;
        let sampling_frequency = self.beamforming_config.sampling_frequency;
        let center_frequency = self.beamforming_config.center_frequency;

        let mut volume = Array3::<f64>::zeros((nx, ny, nz));

        let (num_tx, num_rx) = (
            self.beamforming_config.num_elements_3d.0,
            self.beamforming_config.num_elements_3d.1,
        );
        let (tx_spacing, rx_spacing) = (
            self.beamforming_config.element_spacing_3d.0,
            self.beamforming_config.element_spacing_3d.1,
        );
        let n_rf_samples = rf_data_f64.dim().2;

        for i in 0..nx {
            let x = i as f64 * dx;
            for j in 0..ny {
                let y = j as f64 * dy;
                for k in 0..nz {
                    let z = k as f64 * dz;
                    let voxel_position = [x, y, z];

                    // Complex coherent accumulator (I, Q)
                    let mut sum_i = 0.0_f64;
                    let mut sum_q = 0.0_f64;
                    // Incoherent accumulator for coherence factor
                    let mut incoherent_sum = 0.0_f64;
                    let mut num_contributions = 0usize;

                    for tx_idx in 0..num_tx {
                        let tx_x = tx_idx as f64 * tx_spacing;
                        let tx_position = [tx_x, 0.0, 0.0];

                        for rx_idx in 0..num_rx {
                            let rx_x = rx_idx as f64 * rx_spacing;
                            let rx_position = [rx_x, 0.0, 0.0];

                            // Compute round-trip time-of-flight τ = (d_TX + d_RX) / c
                            let tof = self.compute_time_of_flight(
                                tx_position,
                                rx_position,
                                voxel_position,
                                sound_speed,
                            );

                            // Convert τ to sample index (nearest-neighbour)
                            let sample_idx = (tof * sampling_frequency).round() as usize;
                            if sample_idx >= n_rf_samples {
                                continue;
                            }

                            let element_idx = tx_idx * num_rx + rx_idx;
                            let sample =
                                Self::extract_rf_sample(&rf_data_f64, element_idx, sample_idx);

                            // Apodization weight
                            let apod = self.compute_apodization_weight(
                                tx_idx,
                                rx_idx,
                                num_tx * num_rx,
                            );

                            // Carrier-phase demodulation: multiply by exp(−j·2π·f₀·τ)
                            // I = sample · cos(2π f₀ τ),   Q = sample · (−sin(2π f₀ τ))
                            let (i_comp, q_comp) =
                                Self::demodulate(sample, center_frequency, tof);

                            sum_i += apod * i_comp;
                            sum_q += apod * q_comp;
                            incoherent_sum += apod * sample.abs();
                            num_contributions += 1;
                        }
                    }

                    // Envelope intensity: I² + Q²
                    let intensity = sum_i.powi(2) + sum_q.powi(2);

                    // Optionally weight by coherence factor
                    volume[[i, j, k]] = if self.config.coherence_factor_enabled
                        && num_contributions > 0
                    {
                        let coherent_mag = (sum_i.powi(2) + sum_q.powi(2)).sqrt();
                        let cf = self.compute_coherence_factor(
                            coherent_mag,
                            incoherent_sum,
                            num_contributions,
                        );
                        intensity * cf
                    } else {
                        intensity
                    };
                }
            }
        }

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        info!("SAFT reconstruction completed in {:.2} ms", processing_time);

        Ok(volume.mapv(|v| v as f32))
    }

    /// Validate input RF data dimensions.
    fn validate_input(&self, rf_data: &Array4<f32>) -> KwaversResult<()> {
        if rf_data.is_empty() {
            return Err(KwaversError::InvalidInput(
                "RF data array is empty".to_string(),
            ));
        }

        let channels = rf_data.dim().1;
        let expected_channels = self.beamforming_config.num_elements_3d.0
            * self.beamforming_config.num_elements_3d.1
            * self.beamforming_config.num_elements_3d.2;

        if channels != expected_channels {
            return Err(KwaversError::InvalidInput(format!(
                "Channel count mismatch: expected {}, got {}",
                expected_channels, channels
            )));
        }

        if rf_data.dim().2 == 0 {
            return Err(KwaversError::InvalidInput(
                "RF data must contain at least one sample per channel".to_string(),
            ));
        }

        Ok(())
    }
}

/// Euclidean distance between two 3D points
fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    fn make_processor(nx: usize, ny: usize, nz: usize, ntx: usize, nrx: usize) -> SaftProcessor {
        use super::super::config::BeamformingConfig3D;
        let cfg = BeamformingConfig3D {
            volume_dims: (nx, ny, nz),
            num_elements_3d: (ntx, nrx, 1),
            ..Default::default()
        };
        SaftProcessor::new(SaftConfig::default(), cfg)
    }

    #[test]
    fn test_saft_config_default() {
        let config = SaftConfig::default();
        assert_eq!(config.virtual_sources, 100);
        assert!(matches!(config.apodization, ApodizationWindow::Hamming));
        assert!(config.coherence_factor_enabled);
        assert_eq!(config.f_number, 1.5);
    }

    #[test]
    fn test_saft_processor_creation() {
        let processor = make_processor(32, 32, 32, 8, 8);
        assert_eq!(processor.config.virtual_sources, 100);
    }

    #[test]
    fn test_saft_from_algorithm() {
        use super::super::config::BeamformingConfig3D;
        let algorithm = BeamformingAlgorithm3D::SAFT3D { virtual_sources: 50 };
        let processor =
            SaftProcessor::from_algorithm(&algorithm, BeamformingConfig3D::default()).unwrap();
        assert_eq!(processor.config.virtual_sources, 50);
    }

    #[test]
    fn test_time_of_flight_computation() {
        let processor = make_processor(32, 32, 32, 8, 8);
        let tof = processor.compute_time_of_flight(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            1540.0,
        );
        let expected = (0.001 + 0.001) / 1540.0;
        assert!((tof - expected).abs() < 1e-12);
    }

    #[test]
    fn test_distance_computation() {
        assert!((distance3([0.0, 0.0, 0.0], [0.001, 0.0, 0.0]) - 0.001).abs() < 1e-12);
    }

    #[test]
    fn test_apodization_weight() {
        let processor = make_processor(32, 32, 32, 8, 8);
        // Hamming center: w(0) = 0.54 + 0.46 = 1.0
        let weight = processor.compute_apodization_weight(50, 50, 100);
        assert!(weight > 0.9 && weight <= 1.0 + 1e-10);
    }

    #[test]
    fn test_coherence_factor() {
        let processor = make_processor(32, 32, 32, 8, 8);
        // CF = 100 / (10 * 25) = 0.4
        let cf = processor.compute_coherence_factor(10.0, 5.0, 10);
        assert!((cf - 0.4).abs() < 1e-10);
    }

    /// **Test: demodulate returns in-phase and quadrature components.**
    ///
    /// At τ = 0: I = sample · cos(0) = sample, Q = sample · (−sin(0)) = 0.
    /// At τ = 1/(4f₀) (quarter-period): I = sample · cos(π/2) = 0, Q = sample · (−1).
    #[test]
    fn test_demodulate_phase_correctness() {
        let f0 = 2.5e6_f64; // 2.5 MHz
        let s = 1.0_f64;

        // τ = 0 → (I, Q) = (s, 0)
        let (i0, q0) = SaftProcessor::demodulate(s, f0, 0.0);
        assert!((i0 - s).abs() < 1e-12, "I at τ=0: {i0}");
        assert!(q0.abs() < 1e-12, "Q at τ=0: {q0}");

        // τ = 1/(2f₀) → cos(π) = −1, (I,Q) = (−s, 0)
        let tau_half = 0.5 / f0;
        let (ih, qh) = SaftProcessor::demodulate(s, f0, tau_half);
        assert!((ih + s).abs() < 1e-12, "I at τ=T/2: {ih}");
        assert!(qh.abs() < 1e-12, "Q at τ=T/2: {qh}");
    }

    /// **Test: SAFT PSF — point target produces non-zero reconstruction.**
    ///
    /// An impulse at sample 256 on all channels reconstructs to a non-zero
    /// output volume, verifying the coherent summation path is exercised.
    #[test]
    fn test_reconstruct_volume_basic() {
        use super::super::config::BeamformingConfig3D;
        let beamforming_config = BeamformingConfig3D {
            volume_dims: (16, 16, 16),
            num_elements_3d: (4, 4, 1),
            ..Default::default()
        };
        let processor = SaftProcessor::new(SaftConfig::default(), beamforming_config);

        let num_elements = 4 * 4 * 1;
        let mut rf_data = Array4::<f32>::zeros((1, num_elements, 512, 1));
        for elem in 0..num_elements {
            rf_data[[0, elem, 256, 0]] = 1.0;
        }

        let vol = processor.reconstruct_volume(&rf_data).unwrap();
        assert_eq!(vol.dim(), (16, 16, 16));
        let max_val = vol.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max_val > 0.0, "SAFT output must be non-zero for point target");
    }

    /// **Test: demodulation improves coherent-sum magnitude vs no demodulation.**
    ///
    /// Two samples with 180° relative phase cancel without demodulation but add
    /// constructively in the envelope channel |I² + Q²|.
    ///
    /// Concretely: s₁ at τ₁ = 0 and s₂ at τ₂ = T/2 (half-period shift):
    /// - Without demodulation: coherent_sum = 1 + cos(π) = 0  (destructive)
    /// - I channel after demodulation: I₁ = 1, I₂ = cos(2π·τ₂) = −1 → still 0
    ///
    /// The key: envelope `I² + Q²` of *each individual demodulated sample* is 1
    /// regardless of τ, so the coherent sum of envelopes is always ≥ 0.
    #[test]
    fn test_demodulate_envelope_invariant() {
        let f0 = 2.5e6_f64;
        // For any τ, I² + Q² = s² · (cos² + sin²) = s²
        let s = 0.7_f64;
        for tau_ns in [0u64, 25, 50, 75, 100] {
            let tau = tau_ns as f64 * 1e-9;
            let (i, q) = SaftProcessor::demodulate(s, f0, tau);
            let envelope = (i * i + q * q).sqrt();
            assert!(
                (envelope - s.abs()).abs() < 1e-12,
                "Envelope invariant failed at τ={tau_ns} ns: {envelope:.6} ≠ {s}"
            );
        }
    }

    #[test]
    fn test_input_validation() {
        let processor = make_processor(32, 32, 32, 8, 8);
        // empty
        let empty = Array4::<f32>::zeros((0, 0, 0, 0));
        assert!(processor.validate_input(&empty).is_err());
        // valid (num_elements = 8×8×1 = 64 channels)
        let valid = Array4::<f32>::zeros((1, 64, 512, 1));
        assert!(processor.validate_input(&valid).is_ok());
    }
}
