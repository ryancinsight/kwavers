//! Demonstrates the FFT planner optimization in time reversal reconstruction

use rustfft::{FftPlanner, num_complex::Complex};
use std::time::Instant;

fn main() {
    println!("FFT Planner Optimization Demo");
    println!("=============================\n");
    
    // Test parameters
    let signal_length = 8192;
    let num_signals = 100;
    
    // Generate test signals
    let signals: Vec<Vec<f64>> = (0..num_signals)
        .map(|i| {
            (0..signal_length)
                .map(|t| (t as f64 * 0.1 * (i + 1) as f64).sin())
                .collect()
        })
        .collect();
    
    // Method 1: Creating a new planner for each signal (inefficient)
    println!("Method 1: Creating new FFT planner for each signal");
    let start = Instant::now();
    
    for signal in &signals {
        let mut planner = FftPlanner::new();
        let mut complex_signal: Vec<Complex<f64>> = signal.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        let fft = planner.plan_fft_forward(signal_length);
        fft.process(&mut complex_signal);
        
        let ifft = planner.plan_fft_inverse(signal_length);
        ifft.process(&mut complex_signal);
    }
    
    let duration1 = start.elapsed();
    println!("Time taken: {:?}", duration1);
    println!("Average per signal: {:?}\n", duration1 / num_signals as u32);
    
    // Method 2: Reusing the same planner (efficient)
    println!("Method 2: Reusing FFT planner for all signals");
    let start = Instant::now();
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal_length);
    let ifft = planner.plan_fft_inverse(signal_length);
    
    for signal in &signals {
        let mut complex_signal: Vec<Complex<f64>> = signal.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        fft.process(&mut complex_signal);
        ifft.process(&mut complex_signal);
    }
    
    let duration2 = start.elapsed();
    println!("Time taken: {:?}", duration2);
    println!("Average per signal: {:?}\n", duration2 / num_signals as u32);
    
    // Show improvement
    let speedup = duration1.as_secs_f64() / duration2.as_secs_f64();
    println!("Performance improvement: {:.2}x faster", speedup);
    println!("Time saved: {:?}", duration1 - duration2);
    
    // Memory usage insight
    println!("\nMemory usage insight:");
    println!("- Method 1: Creates {} FFT planners", num_signals);
    println!("- Method 2: Creates 1 FFT planner");
    println!("- Each planner caches twiddle factors and other precomputed data");
}