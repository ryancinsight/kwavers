//! Signal Generation Demo
//! 
//! Demonstrates the comprehensive signal generation capabilities of Kwavers,
//! including frequency sweeps, pulse signals, and modulation techniques.

use kwavers::signal::{
    Signal, SineWave,
    // Pulse signals
    GaussianPulse, ToneBurst, RickerWavelet, PulseTrain,
    WindowType, PulseShape,
    // Frequency sweeps
    LinearFrequencySweep, LogarithmicFrequencySweep, SteppedFrequencySweep,
    TransitionType,
    // Modulation
    AmplitudeModulation, FrequencyModulation, PhaseModulation,
};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Kwavers Signal Generation Demo ===\n");
    
    // Sampling parameters
    let sample_rate = 100_000.0; // 100 kHz
    let dt = 1.0 / sample_rate;
    
    // 1. Demonstrate pulse signals
    demonstrate_pulse_signals(dt)?;
    
    // 2. Demonstrate frequency sweeps
    demonstrate_frequency_sweeps(dt)?;
    
    // 3. Demonstrate modulation techniques
    demonstrate_modulation(dt)?;
    
    // 4. Demonstrate complex ultrasound signals
    demonstrate_ultrasound_signals(dt)?;
    
    println!("\n✅ All signal demonstrations complete!");
    println!("Check the generated CSV files for signal data.");
    
    Ok(())
}

fn demonstrate_pulse_signals(dt: f64) -> Result<(), Box<dyn std::error::Error>> {
    println!("1. PULSE SIGNALS");
    println!("================");
    
    // Gaussian pulse
    let gaussian = GaussianPulse::new(
        5000.0,  // 5 kHz center frequency
        0.001,   // 1 ms center time
        0.0001,  // 100 μs pulse width
        1.0,     // amplitude
    ).with_q_factor(8.0);
    
    println!("✓ Gaussian Pulse: 5 kHz, Q=8");
    save_signal(&gaussian, dt, 0.002, "gaussian_pulse.csv")?;
    
    // Tone burst with Hann window
    let tone_burst = ToneBurst::new(
        10000.0, // 10 kHz
        5.0,     // 5 cycles
        0.0,     // start at t=0
        1.0,     // amplitude
    ).with_window(WindowType::Hann);
    
    println!("✓ Tone Burst: 10 kHz, 5 cycles, Hann window");
    save_signal(&tone_burst, dt, 0.001, "tone_burst.csv")?;
    
    // Ricker wavelet (commonly used in seismic)
    let ricker = RickerWavelet::new(
        30.0,    // 30 Hz peak frequency
        0.05,    // 50 ms peak time
        1.0,     // amplitude
    );
    
    println!("✓ Ricker Wavelet: 30 Hz peak frequency");
    save_signal(&ricker, dt, 0.2, "ricker_wavelet.csv")?;
    
    // Pulse train with Gaussian pulses
    let pulse_train = PulseTrain::new(
        1000.0,  // 1 kHz pulse repetition frequency
        10000.0, // 10 kHz carrier
        1.0,     // amplitude
    )
    .with_duty_cycle(0.3)
    .with_pulse_shape(PulseShape::Gaussian { q_factor: 5.0 });
    
    println!("✓ Pulse Train: 1 kHz PRF, 10 kHz carrier, 30% duty cycle");
    save_signal(&pulse_train, dt, 0.005, "pulse_train.csv")?;
    
    Ok(())
}

fn demonstrate_frequency_sweeps(dt: f64) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. FREQUENCY SWEEPS");
    println!("===================");
    
    // Linear chirp
    let linear_sweep = LinearFrequencySweep::new(
        1000.0,  // 1 kHz start
        10000.0, // 10 kHz end
        0.01,    // 10 ms duration
        1.0,     // amplitude
    );
    
    println!("✓ Linear Chirp: 1-10 kHz over 10 ms");
    save_signal(&linear_sweep, dt, 0.01, "linear_chirp.csv")?;
    
    // Logarithmic sweep
    let log_sweep = LogarithmicFrequencySweep::new(
        100.0,   // 100 Hz start
        20000.0, // 20 kHz end
        0.1,     // 100 ms duration
        1.0,     // amplitude
    );
    
    println!("✓ Logarithmic Sweep: 100 Hz - 20 kHz over 100 ms");
    save_signal(&log_sweep, dt, 0.1, "log_sweep.csv")?;
    
    // Stepped frequency sweep with smooth transitions
    let stepped_sweep = SteppedFrequencySweep::new(
        1000.0,  // 1 kHz start
        5000.0,  // 5 kHz end
        5,       // 5 steps
        0.005,   // 5 ms total duration
        1.0,     // amplitude
    ).with_transition(TransitionType::Smooth { smoothness: 10.0 });
    
    println!("✓ Stepped Sweep: 1-5 kHz, 5 steps with smooth transitions");
    save_signal(&stepped_sweep, dt, 0.005, "stepped_sweep.csv")?;
    
    Ok(())
}

fn demonstrate_modulation(dt: f64) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. MODULATION TECHNIQUES");
    println!("========================");
    
    // Create modulating signal
    let modulating = Box::new(SineWave::new(100.0, 1.0, 0.0));
    
    // Amplitude modulation
    let am = AmplitudeModulation::new(
        5000.0,  // 5 kHz carrier
        1.0,     // carrier amplitude
        modulating.clone(),
    ).with_modulation_index(0.8);
    
    println!("✓ AM: 5 kHz carrier, 100 Hz modulation, m=0.8");
    save_signal(&am, dt, 0.02, "amplitude_modulation.csv")?;
    
    // Frequency modulation
    let fm = FrequencyModulation::new(
        5000.0,  // 5 kHz carrier
        1.0,     // carrier amplitude
        500.0,   // 500 Hz frequency deviation
        modulating.clone(),
    );
    
    println!("✓ FM: 5 kHz carrier, 500 Hz deviation");
    save_signal(&fm, dt, 0.02, "frequency_modulation.csv")?;
    
    // Phase modulation
    let pm = PhaseModulation::new(
        5000.0,  // 5 kHz carrier
        1.0,     // carrier amplitude
        2.0,     // 2 radian phase deviation
        modulating,
    );
    
    println!("✓ PM: 5 kHz carrier, 2 rad phase deviation");
    save_signal(&pm, dt, 0.02, "phase_modulation.csv")?;
    
    Ok(())
}

fn demonstrate_ultrasound_signals(dt: f64) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n4. ULTRASOUND SIGNALS");
    println!("=====================");
    
    // Typical medical ultrasound pulse
    let ultrasound_pulse = ToneBurst::new(
        2.5e6,   // 2.5 MHz center frequency
        3.0,     // 3 cycles
        0.0,     // start at t=0
        1.0,     // amplitude
    ).with_window(WindowType::Gaussian);
    
    println!("✓ Medical Ultrasound: 2.5 MHz, 3 cycles, Gaussian window");
    let us_dt = 1.0 / 50e6; // 50 MHz sampling for ultrasound
    save_signal(&ultrasound_pulse, us_dt, 3e-6, "ultrasound_pulse.csv")?;
    
    // Coded excitation using linear chirp
    let coded_excitation = LinearFrequencySweep::new(
        2e6,     // 2 MHz start
        3e6,     // 3 MHz end
        10e-6,   // 10 μs duration
        1.0,     // amplitude
    );
    
    println!("✓ Coded Excitation: 2-3 MHz chirp over 10 μs");
    save_signal(&coded_excitation, us_dt, 10e-6, "coded_excitation.csv")?;
    
    // Doppler ultrasound simulation (frequency shift)
    let doppler_shift = 500.0; // 500 Hz Doppler shift
    let doppler = FrequencyModulation::new(
        2.5e6,           // 2.5 MHz carrier
        1.0,             // amplitude
        doppler_shift,   // frequency deviation
        Box::new(SineWave::new(60.0, 1.0, 0.0)), // 60 Hz heart rate
    );
    
    println!("✓ Doppler Ultrasound: 2.5 MHz with 500 Hz shift at 60 BPM");
    save_signal(&doppler, us_dt, 50e-6, "doppler_ultrasound.csv")?;
    
    Ok(())
}

/// Save signal samples to CSV file
fn save_signal(
    signal: &dyn Signal,
    dt: f64,
    duration: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let num_samples = (duration / dt) as usize;
    let mut file = File::create(filename)?;
    
    writeln!(file, "time,amplitude,frequency,phase")?;
    
    for i in 0..num_samples {
        let t = i as f64 * dt;
        let amplitude = signal.amplitude(t);
        let frequency = signal.frequency(t);
        let phase = signal.phase(t);
        
        writeln!(file, "{:.9},{:.6},{:.3},{:.6}", t, amplitude, frequency, phase)?;
    }
    
    Ok(())
}

/// Analyze signal properties
fn analyze_signal(signal: &dyn Signal, name: &str, dt: f64, duration: f64) {
    println!("\nAnalyzing: {}", name);
    println!("------------------------");
    
    let num_samples = (duration / dt) as usize;
    let mut max_amplitude: f64 = 0.0;
    let mut total_energy = 0.0;
    let mut freq_sum = 0.0;
    let mut freq_count = 0;
    
    for i in 0..num_samples {
        let t = i as f64 * dt;
        let amplitude = signal.amplitude(t);
        let frequency = signal.frequency(t);
        
        max_amplitude = max_amplitude.max(amplitude.abs());
        total_energy += amplitude * amplitude * dt;
        
        if amplitude.abs() > 0.1 * max_amplitude {
            freq_sum += frequency;
            freq_count += 1;
        }
    }
    
    let avg_frequency = if freq_count > 0 {
        freq_sum / freq_count as f64
    } else {
        0.0
    };
    
    println!("  Peak amplitude: {:.3}", max_amplitude);
    println!("  Total energy: {:.6}", total_energy);
    println!("  Average frequency: {:.1} Hz", avg_frequency);
    
    if let Some(signal_duration) = signal.duration() {
        println!("  Signal duration: {:.6} s", signal_duration);
    }
}