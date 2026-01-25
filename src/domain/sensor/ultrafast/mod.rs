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
//! TODO_AUDIT: P1 - Tilted Plane Wave Compounding - Implement multi-angle coherent compounding
//! DEPENDS ON: domain/sensor/ultrafast/plane_wave.rs (to be created)
//! DEPENDS ON: domain/sensor/transducer.rs (enhance existing)
//! DEPENDS ON: solver/forward/acoustic/plane_wave_simulation.rs (to be created)
//! MISSING: Plane wave transmission at multiple tilt angles (-10° to +10°)
//! MISSING: Coherent compounding of 9-11 angles for SNR improvement
//! MISSING: Delay-and-sum beamforming for tilted plane waves
//! MISSING: Frame rate calculation and timing control
//! MISSING: Pulse repetition frequency (PRF) management (5500 Hz)
//! SEVERITY: HIGH (foundation for functional ultrasound imaging)
//! PERFORMANCE: Target 500 Hz compounded frame rate, 5500 Hz PRF
//! THEOREM: SNR improvement: √N for N compounded angles
//! THEOREM: Frame rate = PRF / (N_angles × N_emissions)
//! REFERENCES: Nouhoum et al. (2021) "11 tilted plane waves, angles from -10° to +10° (step 2°)"
//! REFERENCES: Tanter & Fink (2014) "Ultrafast imaging in biomedical ultrasound" IEEE TUFFC
//! REFERENCES: Montaldo et al. (2009) "Coherent plane-wave compounding for very high frame rate ultrasonography"
//!
//! ✅ IMPLEMENTED: Plane Wave Delay Calculation - Complete geometric delay computation
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
//! TODO_AUDIT: P2 - Diverging Wave Transmission - Implement synthetic transmit aperture
//! DEPENDS ON: domain/sensor/ultrafast/diverging_wave.rs (to be created)
//! MISSING: Diverging wave from virtual point sources
//! MISSING: Synthetic transmit focusing (STF/STA)
//! MISSING: Multi-line transmission (MLT) for parallel beamforming
//! SEVERITY: MEDIUM (alternative to plane waves for some applications)
//! REFERENCES: Montaldo et al. (2009) "Diverging wave imaging"
//!
//! TODO_AUDIT: P2 - Ultrafast Frame Sequencing - Implement transmission scheduling
//! DEPENDS ON: domain/sensor/ultrafast/sequencer.rs (to be created)
//! MISSING: Interleaved transmission sequences
//! MISSING: Flash angle scheduling (optimize angular coverage)
//! MISSING: PRF limit calculation based on imaging depth
//! MISSING: Multi-zone imaging with different PRFs
//! SEVERITY: MEDIUM (optimization for specific applications)
//! THEOREM: Maximum PRF = c / (2 × depth) for unambiguous imaging
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
//! - `diverging_wave`: Diverging wave transmission (future)
//! - `sequencer`: Transmission sequence scheduling (future)

// Implemented modules
pub mod plane_wave;

// Future modules
// pub mod diverging_wave;
// pub mod sequencer;

// Re-export main types
pub use plane_wave::{PlaneWave, PlaneWaveConfig};

#[cfg(test)]
mod tests {
    #[test]
    fn test_ultrafast_module() {
        // Placeholder test
        assert!(true);
    }
}
