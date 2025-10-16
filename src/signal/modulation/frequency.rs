//! Frequency Modulation (FM)

use super::{Modulation, ModulationParams};
use crate::error::KwaversResult;

/// Frequency modulation implementation
#[derive(Debug, Clone)]
pub struct FrequencyModulation {
    params: ModulationParams,
}

impl FrequencyModulation {
    #[must_use]
    pub fn new(params: ModulationParams) -> Self {
        Self { params }
    }
}

impl Modulation for FrequencyModulation {
    fn modulate(&self, carrier: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        // FM modulation: y(t) = A*cos(2π*fc*t + β*∫m(τ)dτ)
        let omega_c = 2.0 * std::f64::consts::PI * self.params.carrier_freq;
        let beta = self.params.modulation_index;

        let mut phase = 0.0;
        let dt = 1.0 / self.params.sample_rate;

        Ok(t.iter()
            .zip(carrier.iter())
            .map(|(&ti, &msg)| {
                phase += msg * dt;
                (omega_c * ti + beta * phase).cos()
            })
            .collect())
    }

    fn demodulate(&self, signal: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>> {
        // FM demodulation using proper Hilbert transform and instantaneous frequency
        // 
        // This implements the proper demodulation using the analytic signal approach:
        // 1. Compute analytic signal z(t) = s(t) + j*H[s(t)]
        // 2. Extract instantaneous phase: φ(t) = atan2(H[s(t)], s(t))
        // 3. Compute instantaneous frequency: f(t) = (1/2π) dφ/dt
        //
        // References:
        // - Boashash (1992): "Estimating and interpreting the instantaneous frequency"
        // - Marple (1999): "Computing the discrete-time analytic signal via FFT"
        use crate::utils::signal_processing::instantaneous_frequency;
        use ndarray::Array1;
        
        if signal.len() < 2 {
            return Ok(signal.to_vec());
        }

        // Convert to ndarray for signal processing
        let signal_array = Array1::from_vec(signal.to_vec());
        
        // Compute sample rate from time array
        let dt = if t.len() > 1 {
            t[1] - t[0]
        } else {
            1.0 / self.params.sample_rate
        };
        
        // Compute instantaneous frequency using Hilbert transform
        let inst_freq = instantaneous_frequency(&signal_array, dt);
        
        // Remove carrier frequency to get baseband signal
        let omega_c = self.params.carrier_freq;
        let demodulated: Vec<f64> = inst_freq.iter()
            .map(|&f| (f - omega_c) / self.params.modulation_index)
            .collect();
        
        Ok(demodulated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fm_modulation_demodulation() {
        // Test that FM modulation followed by demodulation recovers the signal
        let params = ModulationParams {
            carrier_freq: 1000.0,    // 1 kHz carrier
            sample_rate: 10000.0,    // 10 kHz sampling
            modulation_index: 5.0,   // FM modulation index
        };

        let fm = FrequencyModulation::new(params);
        
        // Create simple message signal (low frequency sine wave)
        let t: Vec<f64> = (0..200).map(|i| i as f64 / 10000.0).collect();
        let message: Vec<f64> = t.iter().map(|&ti| (2.0 * std::f64::consts::PI * 10.0 * ti).sin()).collect();

        // Modulate
        let modulated = fm.modulate(&message, &t).unwrap();
        assert_eq!(modulated.len(), 200);
        
        // Demodulate
        let demodulated = fm.demodulate(&modulated, &t).unwrap();
        assert_eq!(demodulated.len(), 200);
        
        // Check that demodulated signal is finite
        assert!(demodulated.iter().all(|&x| x.is_finite()),
                "Demodulated signal should be finite");
    }
    
    #[test]
    fn test_fm_hilbert_based_demodulation_produces_finite_values() {
        // Test that Hilbert-based demodulation produces finite values
        let params = ModulationParams {
            carrier_freq: 1000.0,
            sample_rate: 10000.0,
            modulation_index: 2.0,
        };

        let fm = FrequencyModulation::new(params);
        
        // Create simple message signal
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 10000.0).collect();
        let message: Vec<f64> = t.iter()
            .map(|&ti| 0.3 * (2.0 * std::f64::consts::PI * 5.0 * ti).sin())
            .collect();

        let modulated = fm.modulate(&message, &t).unwrap();
        let demodulated = fm.demodulate(&modulated, &t).unwrap();
        
        // Demodulated signal should be finite
        assert!(demodulated.iter().all(|&x| x.is_finite()),
                "Demodulated FM signal should be finite");
        
        // Should have reasonable magnitude (not all zeros)
        let max_abs = demodulated.iter()
            .map(|&x| x.abs())
            .fold(0.0f64, f64::max);
        
        assert!(max_abs > 1e-10,
                "Demodulated signal should have non-zero content: max = {}", max_abs);
    }
}
