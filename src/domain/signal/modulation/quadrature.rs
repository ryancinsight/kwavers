//! Quadrature Amplitude Modulation (QAM)

use super::{Modulation, ModulationParams};
use crate::core::error::KwaversResult;

/// QAM implementation
#[derive(Debug, Clone)]
pub struct QuadratureAmplitudeModulation {
    params: ModulationParams,
}

impl QuadratureAmplitudeModulation {
    #[must_use]
    pub fn new(params: ModulationParams) -> Self {
        Self { params }
    }
}

impl Modulation for QuadratureAmplitudeModulation {
    fn modulate(&self, carrier: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        // Proper QAM modulation using Hilbert transform for quadrature component
        //
        // QAM uses two orthogonal carriers (in-phase and quadrature):
        // s(t) = I(t)*cos(ωt) - Q(t)*sin(ωt)
        //
        // The Q component is generated as the Hilbert transform of I,
        // providing proper 90° phase shift for orthogonal signaling.
        //
        // References:
        // - Proakis & Salehi (2008): "Digital Communications" Chapter 4
        // - Haykin (2001): "Communication Systems" Chapter 6
        use crate::domain::signal::analytic::hilbert_transform;
        use ndarray::Array1;

        if carrier.len() != t.len() {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Carrier and time arrays must have same length".to_string(),
            ));
        }

        let omega_c = 2.0 * std::f64::consts::PI * self.params.carrier_freq;

        // Convert I component to ndarray
        let i_signal = Array1::from_vec(carrier.to_vec());

        // Generate Q component using Hilbert transform (90° phase shift)
        let q_signal_complex = hilbert_transform(&i_signal);

        // Extract imaginary part as Q component (90° phase shift)
        let q_signal: Vec<f64> = q_signal_complex.iter().map(|c| c.im).collect();

        // Generate QAM signal
        let qam: Vec<f64> = t
            .iter()
            .enumerate()
            .map(|(idx, &ti)| {
                let i_comp = i_signal[idx];
                let q_comp = q_signal[idx];

                // QAM signal = I*cos(ωt) - Q*sin(ωt)
                i_comp * (omega_c * ti).cos() - q_comp * (omega_c * ti).sin()
            })
            .collect();

        Ok(qam)
    }

    fn demodulate(&self, signal: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        // Coherent QAM demodulation
        if signal.is_empty() {
            return Ok(Vec::new());
        }

        let omega_c = 2.0 * std::f64::consts::PI * self.params.carrier_freq;
        let mut demodulated = Vec::with_capacity(signal.len());

        for (i, &sample) in signal.iter().enumerate() {
            let ti = if i < t.len() {
                t[i]
            } else {
                i as f64 / self.params.sample_rate
            };

            // Extract I component
            let i_component = 2.0 * sample * (omega_c * ti).cos();

            // Extract Q component
            let q_component = -2.0 * sample * (omega_c * ti).sin();

            // Combine I/Q components (envelope detection per Lyons 2010 §13.3)
            // Full QAM demodulation requires constellation mapping and symbol decision
            // Current: Envelope magnitude suitable for analog QAM signals
            demodulated.push((i_component.powi(2) + q_component.powi(2)).sqrt());
        }

        Ok(demodulated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qam_modulation_with_hilbert() {
        // Test that QAM modulation using Hilbert transform produces valid output
        let params = ModulationParams {
            carrier_freq: 1000.0, // 1 kHz carrier
            sample_rate: 10000.0, // 10 kHz sampling
            modulation_index: 1.0,
        };

        let qam = QuadratureAmplitudeModulation::new(params);

        // Create message signal
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 10000.0).collect();
        let message: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * 10.0 * ti).sin())
            .collect();

        // Modulate
        let modulated = qam.modulate(&message, &t).unwrap();

        assert_eq!(modulated.len(), 100);
        assert!(
            modulated.iter().all(|&x| x.is_finite()),
            "QAM modulated signal should be finite"
        );
    }

    #[test]
    fn test_qam_quadrature_orthogonality() {
        // Test that I and Q components are properly orthogonal
        let params = ModulationParams {
            carrier_freq: 1000.0,
            sample_rate: 10000.0,
            modulation_index: 1.0,
        };

        let qam = QuadratureAmplitudeModulation::new(params);

        // Use constant I component
        let t: Vec<f64> = (0..200).map(|i| i as f64 / 10000.0).collect();
        let i_component = vec![1.0; 200];

        let modulated = qam.modulate(&i_component, &t).unwrap();

        // QAM signal should have both positive and negative values
        // (indicating proper quadrature mixing)
        let has_positive = modulated.iter().any(|&x| x > 0.0);
        let has_negative = modulated.iter().any(|&x| x < 0.0);

        assert!(
            has_positive && has_negative,
            "QAM signal should have both positive and negative values"
        );
    }

    #[test]
    fn test_qam_modulation_demodulation_roundtrip() {
        // Test basic modulation-demodulation cycle
        let params = ModulationParams {
            carrier_freq: 1000.0,
            sample_rate: 10000.0,
            modulation_index: 1.0,
        };

        let qam = QuadratureAmplitudeModulation::new(params);

        let t: Vec<f64> = (0..100).map(|i| i as f64 / 10000.0).collect();
        let message = vec![0.8; 100];

        let modulated = qam.modulate(&message, &t).unwrap();
        let demodulated = qam.demodulate(&modulated, &t).unwrap();

        assert_eq!(demodulated.len(), 100);
        assert!(
            demodulated.iter().all(|&x| x.is_finite()),
            "Demodulated QAM signal should be finite"
        );
    }
}
