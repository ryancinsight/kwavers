//! Ultrafast Ultrasound Imaging
//!
//! This module implements ultrafast ultrasound imaging techniques using plane wave
//! transmission and coherent compounding for high-frame-rate imaging applications.
//!
//! # Overview
//!
//! Ultrafast ultrasound achieves frame rates of 500-10,000 Hz by transmitting unfocused
//! plane waves instead of focused beams, enabling real-time capture of transient phenomena
//! such as blood flow, shear waves, and dynamic tissue motion.
//!
//! ## Plane Wave Delay Calculation (Implemented)
//!
//! Tilted plane wave compounding delay geometry is fully implemented in `plane_wave.rs`:
//!
//! Fully implemented in `plane_wave.rs` module with:
//! - Transmission delays: τ_tx(x,θ) = -x·sin(θ)/c
//! - Reception delays: τ_rx(x,y,θ) = (x·sin(θ) + y·cos(θ))/c
//! - Total beamforming delays: τ_total = (2x·sin(θ) + y·cos(θ))/c
//! - F-number dependent Hann apodization
//! - Speed of sound compensation (1540 m/s tissue default)
//! - Delay surface computation for full image grids
//! - Support for 11-angle compounding (-10° to +10°, 2° steps)
//!
//! See `plane_wave.rs` for complete API and tests.
//! REFERENCES: Jensen et al. (2006), Montaldo et al. (2009), Tanter & Fink (2014)
//!
//! ## Diverging Wave / STA (Implemented in `diverging_wave.rs`)
//!
//! Virtual source model (Papadacci et al. 2014): τ_tx = (√((x−xᵢ)²+(z+F)²)−F)/c.
//! Synthetic transmit aperture delay table, transmit-delay surfaces, Hann F-number
//! apodization, and PRF theorem: PRF_max = c/(2·z_max) (Tanter & Fink 2014).
//!
//! ## Transmission Sequence Scheduling (Implemented in `sequencer.rs`)
//!
//! Sequential and interleaved (Montaldo et al. 2009) angle schedules, STA element
//! firing order, flash (0°) schedule, PRF enforcement with range-ambiguity check.
//!
//! # Key Concepts
//!
//! ## Plane Wave Imaging
//!
//! Traditional focused ultrasound:
//! - Transmit focused beam → Receive along scan line → Repeat for each line
//! - Frame rate limited to ~30-100 Hz
//!
//! Plane wave imaging:
//! - Transmit unfocused plane wave → Receive from entire field
//! - Single transmission images entire region
//! - Frame rate = PRF (potentially >10,000 Hz)
//!
//! ## Coherent Compounding
//!
//! Transmit N plane waves at different angles θ₁, θ₂, ..., θₙ:
//! ```text
//! I_compounded = (1/N) Σᵢ |I(θᵢ)|²
//! ```
//!
//! Benefits:
//! - SNR improvement: ~√N
//! - Reduced side lobes and clutter
//! - Improved contrast and resolution
//!
//! Trade-off: Frame rate reduced by factor of N
//! - 11 angles @ 5500 Hz PRF → 500 Hz compounded frame rate
//!
//! ## Delay Calculation
//!
//! For plane wave with tilt angle θ:
//! ```text
//! Transmission delay: τ_tx(x, θ) = -x·sin(θ) / c
//! Reception delay: τ_rx(x, y, θ) = (x·sin(θ) + y·cos(θ)) / c
//! Total delay: τ(x, y, θ) = τ_rx(x, y, θ) - τ_tx(x, θ)
//! ```
//!
//! where:
//! - x: lateral position (perpendicular to beam)
//! - y: axial depth (along beam direction)
//! - θ: plane wave tilt angle
//! - c: speed of sound (~1540 m/s)
//!
//! # Applications Enabled by Ultrafast Imaging
//!
//! - **Functional Ultrasound (fUS)**: High-speed Doppler for brain imaging
//! - **Shear Wave Elastography**: Transient wave propagation tracking
//! - **Ultrafast Doppler**: Blood flow imaging with high temporal resolution
//! - **4D Imaging**: Real-time 3D volume acquisition
//! - **Contrast-Enhanced Imaging**: Microbubble dynamics
//!
//! # Performance Specifications
//!
//! Based on Nouhoum et al. (2021) and typical ultrafast systems:
//!
//! | Parameter | Value | Notes |
//! |-----------|-------|-------|
//! | Frequency | 15-20 MHz | High resolution for small animals |
//! | PRF | 5500 Hz | Pulse repetition frequency |
//! | Compounded Frame Rate | 500 Hz | 11 angles compounded |
//! | Tilt Angles | -10° to +10° | 2° step (11 angles) |
//! | In-plane Resolution | 100 μm | After compounding |
//! | Slice Thickness | 400 μm | Elevation direction |
//! | Elements | 128 | Linear array |
//! | Pitch | 0.11 mm | Element spacing |
//!
//! # Literature References
//!
//! **Foundational Papers:**
//! - Tanter, M., & Fink, M. (2014). "Ultrafast imaging in biomedical ultrasound."
//!   *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*, 61(1), 102-119.
//!   DOI: 10.1109/TUFFC.2014.2882
//!
//! - Montaldo, G., Tanter, M., Bercoff, J., Benech, N., & Fink, M. (2009).
//!   "Coherent plane-wave compounding for very high frame rate ultrasonography and transient elastography."
//!   *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*, 56(3), 489-506.
//!   DOI: 10.1109/TUFFC.2009.1067
//!
//! **Functional Ultrasound:**
//! - Macé, E., et al. (2011). "Functional ultrasound imaging of the brain."
//!   *Nature Methods*, 8(8), 662-664. DOI: 10.1038/nmeth.1641
//!
//! - Deffieux, T., et al. (2018). "Functional ultrasound neuroimaging: a review."
//!   *Current Opinion in Neurobiology*, 50, 128-135. DOI: 10.1016/j.conb.2018.02.001
//!
//! **Beamforming:**
//! - Jensen, J. A., et al. (2006). "Synthetic aperture ultrasound imaging."
//!   *Ultrasonics*, 44, e5-e15. DOI: 10.1016/j.ultras.2006.07.017
//!
//! # Module Organization
//!
//! - `plane_wave`: Plane wave transmission and delay calculation
//! - `diverging_wave`: Virtual source / STA diverging wave delays and apodization
//! - `sequencer`: Transmission sequence scheduling, interleaved angles, STA firing order

pub mod diverging_wave;
pub mod plane_wave;
pub mod sequencer;

pub use diverging_wave::{DivergingWave, DivergingWaveConfig};
pub use plane_wave::{PlaneWave, PlaneWaveConfig};
pub use sequencer::{TransmissionSchedule, TransmissionSequencer};

#[cfg(test)]
mod tests {
    use super::PlaneWaveConfig;

    #[test]
    fn test_ultrafast_module() {
        let config = PlaneWaveConfig::default();
        assert_eq!(config.tilt_angles.len(), 11);
        assert!(config.sound_speed > 0.0);
        assert!(config.f_number.is_some());
    }
}
