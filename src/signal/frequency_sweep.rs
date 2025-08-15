// signal/frequency_sweep.rs
//! Frequency sweep signal generation module
//! 
//! Implements various frequency sweep techniques:
//! - Linear frequency sweep (chirp)
//! - Logarithmic frequency sweep
//! - Hyperbolic frequency sweep
//! - Exponential frequency sweep
//! - Stepped frequency sweep
//! 
//! Literature references:
//! - Klauder et al. (1960): "The theory and design of chirp radars"
//! - Stankovic et al. (1994): "Time-frequency signal analysis"
//! - Misaridis & Jensen (2005): "Use of modulated excitation signals in ultrasound"

use crate::signal::Signal;
use std::f64::consts::{E, PI};
use std::fmt::Debug;

// Physical constants for frequency sweeps
/// Minimum frequency to avoid numerical issues [Hz]
const MIN_FREQUENCY: f64 = 1.0;

/// Maximum frequency ratio for logarithmic sweeps
const MAX_FREQUENCY_RATIO: f64 = 1e6;

/// Default number of steps for stepped frequency sweep
const DEFAULT_FREQUENCY_STEPS: usize = 10;

/// Minimum sweep duration [seconds]
const MIN_SWEEP_DURATION: f64 = 1e-9;

/// Linear frequency sweep (chirp signal)
/// 
/// Frequency varies linearly with time:
/// f(t) = f₀ + (f₁ - f₀) * t / T
/// 
/// Phase: φ(t) = 2π ∫[0,t] f(τ) dτ = 2π(f₀t + (f₁-f₀)t²/(2T))
#[derive(Debug, Clone)]
pub struct LinearFrequencySweep {
    start_frequency: f64,
    end_frequency: f64,
    duration: f64,
    amplitude: f64,
    initial_phase: f64,
}

impl LinearFrequencySweep {
    pub fn new(
        start_frequency: f64,
        end_frequency: f64,
        duration: f64,
        amplitude: f64,
    ) -> Self {
        assert!(start_frequency >= MIN_FREQUENCY, "Start frequency too low");
        assert!(end_frequency >= MIN_FREQUENCY, "End frequency too low");
        assert!(duration >= MIN_SWEEP_DURATION, "Duration too short");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");
        
        Self {
            start_frequency,
            end_frequency,
            duration,
            amplitude,
            initial_phase: 0.0,
        }
    }
    
    pub fn with_initial_phase(mut self, phase: f64) -> Self {
        self.initial_phase = phase;
        self
    }
    
    fn instantaneous_frequency(&self, t: f64) -> f64 {
        if t <= 0.0 {
            self.start_frequency
        } else if t >= self.duration {
            self.end_frequency
        } else {
            self.start_frequency + (self.end_frequency - self.start_frequency) * t / self.duration
        }
    }
    
    fn integrated_phase(&self, t: f64) -> f64 {
        if t <= 0.0 {
            0.0
        } else if t >= self.duration {
            // Complete integral over duration
            2.0 * PI * (self.start_frequency * self.duration 
                + 0.5 * (self.end_frequency - self.start_frequency) * self.duration)
        } else {
            // Integral from 0 to t
            2.0 * PI * (self.start_frequency * t 
                + 0.5 * (self.end_frequency - self.start_frequency) * t * t / self.duration)
        }
    }
}

impl Signal for LinearFrequencySweep {
    fn amplitude(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            0.0
        } else {
            let phase = self.integrated_phase(t) + self.initial_phase;
            self.amplitude * phase.sin()
        }
    }
    
    fn frequency(&self, t: f64) -> f64 {
        self.instantaneous_frequency(t)
    }
    
    fn phase(&self, t: f64) -> f64 {
        self.integrated_phase(t) + self.initial_phase
    }
    
    fn duration(&self) -> Option<f64> {
        Some(self.duration)
    }
    
    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

/// Logarithmic frequency sweep
/// 
/// Frequency varies logarithmically with time:
/// f(t) = f₀ * (f₁/f₀)^(t/T)
/// 
/// Useful for covering wide frequency ranges with constant Q (quality factor)
#[derive(Debug, Clone)]
pub struct LogarithmicFrequencySweep {
    start_frequency: f64,
    end_frequency: f64,
    duration: f64,
    amplitude: f64,
    initial_phase: f64,
}

impl LogarithmicFrequencySweep {
    pub fn new(
        start_frequency: f64,
        end_frequency: f64,
        duration: f64,
        amplitude: f64,
    ) -> Self {
        assert!(start_frequency >= MIN_FREQUENCY, "Start frequency too low");
        assert!(end_frequency >= MIN_FREQUENCY, "End frequency too low");
        assert!(duration >= MIN_SWEEP_DURATION, "Duration too short");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");
        
        let ratio = (end_frequency / start_frequency).abs();
        assert!(ratio <= MAX_FREQUENCY_RATIO, "Frequency ratio too large");
        
        Self {
            start_frequency,
            end_frequency,
            duration,
            amplitude,
            initial_phase: 0.0,
        }
    }
    
    fn instantaneous_frequency(&self, t: f64) -> f64 {
        if t <= 0.0 {
            self.start_frequency
        } else if t >= self.duration {
            self.end_frequency
        } else {
            let ratio = self.end_frequency / self.start_frequency;
            self.start_frequency * ratio.powf(t / self.duration)
        }
    }
    
    fn integrated_phase(&self, t: f64) -> f64 {
        if t <= 0.0 {
            0.0
        } else if t >= self.duration {
            // Complete integral
            let k = (self.end_frequency / self.start_frequency).ln() / self.duration;
            2.0 * PI * self.start_frequency * (E.powf(k * self.duration) - 1.0) / k
        } else {
            // Integral from 0 to t
            let k = (self.end_frequency / self.start_frequency).ln() / self.duration;
            if k.abs() < 1e-10 {
                // Linear case (avoid division by zero)
                2.0 * PI * self.start_frequency * t
            } else {
                2.0 * PI * self.start_frequency * (E.powf(k * t) - 1.0) / k
            }
        }
    }
}

impl Signal for LogarithmicFrequencySweep {
    fn amplitude(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            0.0
        } else {
            let phase = self.integrated_phase(t) + self.initial_phase;
            self.amplitude * phase.sin()
        }
    }
    
    fn frequency(&self, t: f64) -> f64 {
        self.instantaneous_frequency(t)
    }
    
    fn phase(&self, t: f64) -> f64 {
        self.integrated_phase(t) + self.initial_phase
    }
    
    fn duration(&self) -> Option<f64> {
        Some(self.duration)
    }
    
    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

/// Hyperbolic frequency sweep
/// 
/// Frequency varies hyperbolically with time:
/// f(t) = f₀f₁T / (f₁T - (f₁-f₀)t)
/// 
/// Provides constant group delay across frequency spectrum
#[derive(Debug, Clone)]
pub struct HyperbolicFrequencySweep {
    start_frequency: f64,
    end_frequency: f64,
    duration: f64,
    amplitude: f64,
    initial_phase: f64,
}

impl HyperbolicFrequencySweep {
    pub fn new(
        start_frequency: f64,
        end_frequency: f64,
        duration: f64,
        amplitude: f64,
    ) -> Self {
        assert!(start_frequency >= MIN_FREQUENCY, "Start frequency too low");
        assert!(end_frequency >= MIN_FREQUENCY, "End frequency too low");
        assert!(duration >= MIN_SWEEP_DURATION, "Duration too short");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");
        
        Self {
            start_frequency,
            end_frequency,
            duration,
            amplitude,
            initial_phase: 0.0,
        }
    }
    
    fn instantaneous_frequency(&self, t: f64) -> f64 {
        if t <= 0.0 {
            self.start_frequency
        } else if t >= self.duration {
            self.end_frequency
        } else {
            let f0 = self.start_frequency;
            let f1 = self.end_frequency;
            let T = self.duration;
            
            // Hyperbolic sweep formula
            (f0 * f1 * T) / (f1 * T - (f1 - f0) * t)
        }
    }
    
    fn integrated_phase(&self, t: f64) -> f64 {
        if t <= 0.0 {
            0.0
        } else {
            let f0 = self.start_frequency;
            let f1 = self.end_frequency;
            let T = self.duration;
            
            let t_clamped = t.min(self.duration * 0.999); // Avoid singularity
            
            // Integral of hyperbolic function
            let arg = (f1 * T - (f1 - f0) * t_clamped) / (f1 * T);
            -2.0 * PI * f0 * f1 * T / (f1 - f0) * arg.ln()
        }
    }
}

impl Signal for HyperbolicFrequencySweep {
    fn amplitude(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            0.0
        } else {
            let phase = self.integrated_phase(t) + self.initial_phase;
            self.amplitude * phase.sin()
        }
    }
    
    fn frequency(&self, t: f64) -> f64 {
        self.instantaneous_frequency(t)
    }
    
    fn phase(&self, t: f64) -> f64 {
        self.integrated_phase(t) + self.initial_phase
    }
    
    fn duration(&self) -> Option<f64> {
        Some(self.duration)
    }
    
    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

/// Stepped frequency sweep
/// 
/// Frequency changes in discrete steps
/// Useful for frequency response measurements
#[derive(Debug, Clone)]
pub struct SteppedFrequencySweep {
    frequencies: Vec<f64>,
    step_duration: f64,
    amplitude: f64,
    initial_phase: f64,
    transition_type: TransitionType,
}

#[derive(Debug, Clone, Copy)]
pub enum TransitionType {
    Instantaneous,
    Linear { transition_fraction: f64 },
    Smooth { smoothness: f64 },
}

impl SteppedFrequencySweep {
    pub fn new(
        start_frequency: f64,
        end_frequency: f64,
        num_steps: usize,
        total_duration: f64,
        amplitude: f64,
    ) -> Self {
        assert!(start_frequency >= MIN_FREQUENCY, "Start frequency too low");
        assert!(end_frequency >= MIN_FREQUENCY, "End frequency too low");
        assert!(num_steps > 0, "Must have at least one step");
        assert!(total_duration >= MIN_SWEEP_DURATION, "Duration too short");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");
        
        // Generate frequency steps (linear spacing)
        let frequencies = (0..num_steps)
            .map(|i| {
                let fraction = i as f64 / (num_steps - 1).max(1) as f64;
                start_frequency + (end_frequency - start_frequency) * fraction
            })
            .collect();
        
        Self {
            frequencies,
            step_duration: total_duration / num_steps as f64,
            amplitude,
            initial_phase: 0.0,
            transition_type: TransitionType::Instantaneous,
        }
    }
    
    pub fn with_frequencies(frequencies: Vec<f64>, step_duration: f64, amplitude: f64) -> Self {
        assert!(!frequencies.is_empty(), "Must have at least one frequency");
        assert!(frequencies.iter().all(|&f| f >= MIN_FREQUENCY), "Frequencies too low");
        assert!(step_duration >= MIN_SWEEP_DURATION, "Step duration too short");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");
        
        Self {
            frequencies,
            step_duration,
            amplitude,
            initial_phase: 0.0,
            transition_type: TransitionType::Instantaneous,
        }
    }
    
    pub fn with_transition(mut self, transition_type: TransitionType) -> Self {
        self.transition_type = transition_type;
        self
    }
    
    fn get_step_and_phase(&self, t: f64) -> (usize, f64) {
        if t <= 0.0 {
            (0, 0.0)
        } else {
            let step = (t / self.step_duration).floor() as usize;
            let phase_in_step = (t % self.step_duration) / self.step_duration;
            (step.min(self.frequencies.len() - 1), phase_in_step)
        }
    }
    
    fn instantaneous_frequency(&self, t: f64) -> f64 {
        let (step, phase_in_step) = self.get_step_and_phase(t);
        
        match self.transition_type {
            TransitionType::Instantaneous => {
                self.frequencies[step]
            }
            
            TransitionType::Linear { transition_fraction } => {
                if phase_in_step < transition_fraction && step < self.frequencies.len() - 1 {
                    // Linear transition to next frequency
                    let f_current = self.frequencies[step];
                    let f_next = self.frequencies[step + 1];
                    let transition_phase = phase_in_step / transition_fraction;
                    f_current + (f_next - f_current) * transition_phase
                } else {
                    self.frequencies[step]
                }
            }
            
            TransitionType::Smooth { smoothness } => {
                if step < self.frequencies.len() - 1 {
                    // Smooth transition using sigmoid
                    let f_current = self.frequencies[step];
                    let f_next = self.frequencies[step + 1];
                    let x = (phase_in_step - 0.5) * smoothness;
                    let sigmoid = 1.0 / (1.0 + (-x).exp());
                    f_current + (f_next - f_current) * sigmoid
                } else {
                    self.frequencies[step]
                }
            }
        }
    }
}

impl Signal for SteppedFrequencySweep {
    fn amplitude(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.step_duration * self.frequencies.len() as f64 {
            0.0
        } else {
            // Accumulate phase across steps
            let mut phase = self.initial_phase;
            let (current_step, phase_in_step) = self.get_step_and_phase(t);
            
            // Add phase from completed steps
            for i in 0..current_step {
                phase += 2.0 * PI * self.frequencies[i] * self.step_duration;
            }
            
            // Add phase from current step
            phase += 2.0 * PI * self.instantaneous_frequency(t) * phase_in_step * self.step_duration;
            
            self.amplitude * phase.sin()
        }
    }
    
    fn frequency(&self, t: f64) -> f64 {
        self.instantaneous_frequency(t)
    }
    
    fn phase(&self, t: f64) -> f64 {
        // Simplified phase calculation
        let mut phase = self.initial_phase;
        let (current_step, phase_in_step) = self.get_step_and_phase(t);
        
        for i in 0..current_step {
            phase += 2.0 * PI * self.frequencies[i] * self.step_duration;
        }
        
        phase += 2.0 * PI * self.frequencies[current_step] * phase_in_step * self.step_duration;
        phase
    }
    
    fn duration(&self) -> Option<f64> {
        Some(self.step_duration * self.frequencies.len() as f64)
    }
    
    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

/// Polynomial frequency sweep
/// 
/// Frequency varies according to a polynomial function:
/// f(t) = Σ aₙtⁿ
#[derive(Debug, Clone)]
pub struct PolynomialFrequencySweep {
    coefficients: Vec<f64>, // a₀, a₁, a₂, ...
    duration: f64,
    amplitude: f64,
    initial_phase: f64,
}

impl PolynomialFrequencySweep {
    /// Create a quadratic frequency sweep
    /// f(t) = f₀ + βt + γt²
    pub fn quadratic(
        start_frequency: f64,
        mid_frequency: f64,
        end_frequency: f64,
        duration: f64,
        amplitude: f64,
    ) -> Self {
        assert!(start_frequency >= MIN_FREQUENCY, "Start frequency too low");
        assert!(mid_frequency >= MIN_FREQUENCY, "Mid frequency too low");
        assert!(end_frequency >= MIN_FREQUENCY, "End frequency too low");
        assert!(duration >= MIN_SWEEP_DURATION, "Duration too short");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");
        
        // Solve for coefficients given three points
        let t_mid = duration / 2.0;
        
        // System of equations:
        // f(0) = a₀ = start_frequency
        // f(T/2) = a₀ + a₁(T/2) + a₂(T/2)² = mid_frequency
        // f(T) = a₀ + a₁T + a₂T² = end_frequency
        
        let a0 = start_frequency;
        let a2 = (2.0 * (end_frequency + start_frequency - 2.0 * mid_frequency)) / (duration * duration);
        let a1 = (end_frequency - start_frequency - a2 * duration * duration) / duration;
        
        Self {
            coefficients: vec![a0, a1, a2],
            duration,
            amplitude,
            initial_phase: 0.0,
        }
    }
    
    fn instantaneous_frequency(&self, t: f64) -> f64 {
        if t <= 0.0 {
            self.coefficients[0]
        } else if t >= self.duration {
            // Evaluate polynomial at duration
            self.coefficients.iter()
                .enumerate()
                .map(|(i, &coeff)| coeff * self.duration.powi(i as i32))
                .sum()
        } else {
            // Evaluate polynomial at t
            self.coefficients.iter()
                .enumerate()
                .map(|(i, &coeff)| coeff * t.powi(i as i32))
                .sum()
        }
    }
    
    fn integrated_phase(&self, t: f64) -> f64 {
        if t <= 0.0 {
            0.0
        } else {
            let t_clamped = t.min(self.duration);
            
            // Integrate polynomial: ∫ Σ aₙtⁿ dt = Σ aₙtⁿ⁺¹/(n+1)
            let integral: f64 = self.coefficients.iter()
                .enumerate()
                .map(|(i, &coeff)| coeff * t_clamped.powi(i as i32 + 1) / (i as f64 + 1.0))
                .sum();
            
            2.0 * PI * integral
        }
    }
}

impl Signal for PolynomialFrequencySweep {
    fn amplitude(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            0.0
        } else {
            let phase = self.integrated_phase(t) + self.initial_phase;
            self.amplitude * phase.sin()
        }
    }
    
    fn frequency(&self, t: f64) -> f64 {
        self.instantaneous_frequency(t)
    }
    
    fn phase(&self, t: f64) -> f64 {
        self.integrated_phase(t) + self.initial_phase
    }
    
    fn duration(&self) -> Option<f64> {
        Some(self.duration)
    }
    
    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_sweep() {
        let sweep = LinearFrequencySweep::new(1000.0, 2000.0, 0.001, 1.0);
        
        // Check frequency at start
        assert!((sweep.frequency(0.0) - 1000.0).abs() < 1e-6);
        
        // Check frequency at middle
        assert!((sweep.frequency(0.0005) - 1500.0).abs() < 1e-6);
        
        // Check frequency at end
        assert!((sweep.frequency(0.001) - 2000.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_logarithmic_sweep() {
        let sweep = LogarithmicFrequencySweep::new(100.0, 10000.0, 0.01, 1.0);
        
        // Check frequency at start
        assert!((sweep.frequency(0.0) - 100.0).abs() < 1e-6);
        
        // Check frequency at end
        assert!((sweep.frequency(0.01) - 10000.0).abs() < 1e-6);
        
        // Check logarithmic progression
        let mid_freq = sweep.frequency(0.005);
        let expected = 100.0 * (10000.0 / 100.0).sqrt(); // Geometric mean
        assert!((mid_freq - expected).abs() / expected < 0.01);
    }
    
    #[test]
    fn test_stepped_sweep() {
        let sweep = SteppedFrequencySweep::new(1000.0, 3000.0, 3, 0.003, 1.0);
        
        // Check frequencies at each step
        assert!((sweep.frequency(0.0) - 1000.0).abs() < 1e-6);
        assert!((sweep.frequency(0.001) - 1000.0).abs() < 1e-6); // Still in first step
        assert!((sweep.frequency(0.0015) - 2000.0).abs() < 1e-6); // Second step
        assert!((sweep.frequency(0.0025) - 3000.0).abs() < 1e-6); // Third step
    }
}