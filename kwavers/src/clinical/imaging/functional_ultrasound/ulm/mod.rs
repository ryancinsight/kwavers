//! Ultrasound Localization Microscopy (ULM)
//!
//! This module implements super-resolution vascular imaging through microbubble
//! localization and tracking, achieving resolution far beyond the diffraction limit
//! of conventional ultrasound imaging.
//!
//! # Overview
//!
//! ULM breaks the diffraction limit by localizing individual microbubbles with sub-wavelength
//! precision and accumulating their trajectories over time to reconstruct super-resolved
//! vascular networks. Achieves 5-10 μm resolution compared to 100-150 μm for conventional
//! ultrasound.
//!
//! ## Implemented Capabilities
//!
//! - **Microbubble Detection** (`microbubble_detection.rs`): Spatiotemporal SVD clutter
//!   filtering (Demené et al. 2015), 2D Gaussian fitting for sub-pixel localisation
//!   (`σ_loc ≈ λ/(2NA·√SNR)`), local-maximum detection, and PSF-width estimation.
//!
//! - **Microbubble Tracking** (`tracking.rs`): Hungarian algorithm O(n³) optimal
//!   bipartite matching (Kuhn 1955), multi-frame trajectory linking with maximum
//!   displacement constraint, sliding-window track smoothing.
//!
//! - **Super-Resolution Reconstruction** (`super_resolution.rs`): 2D histogram
//!   accumulation with 5 μm pixels, Gaussian kernel splatting (Errico et al. 2015),
//!   sliding average trajectory smoothing (Nouhoum et al. 2021), density normalization.
//!
//! - **Velocity Mapping** (`velocity_mapping.rs`): Instantaneous velocity estimation
//!   from consecutive detections, grid-averaged velocity fields (vx, vz, speed,
//!   direction), wall shear stress proxy via ‖∇speed‖ (Heiles et al. 2022;
//!   Womersley 1955).
//!
//! ## Not yet implemented
//!
//! - Contrast agent simulation: SonoVue backscatter cross-section and
//!   destruction/replenishment dynamics (De Jong et al. 2002).
//!
//! # ULM Imaging Pipeline
//!
//! ## Acquisition Protocol
//!
//! ```text
//! 1. Contrast Agent Injection
//!    ├── SonoVue: 2.5 mg powder in 5 mL NaCl
//!    ├── Injection volume: 100-200 μL IV bolus
//!    └── Wait for circulation (~10-30 seconds)
//!
//! 2. Ultrafast Acquisition
//!    ├── Frame rate: 1000 Hz (9 tilted plane waves)
//!    ├── Angles: -8° to +8° (2° step)
//!    ├── Duration: 600 blocks × 400 frames = 240 seconds
//!    └── Continuous vs triggered acquisition
//!
//! 3. Image Compounding
//!    └── Coherent compounding of 9 angles per frame
//! ```
//!
//! ## Processing Pipeline
//!
//! ```text
//! 1. Clutter Filtering (per 400-frame block)
//!    ├── Spatiotemporal SVD
//!    ├── Remove tissue subspace (K=2-5 components)
//!    └── Isolate microbubble signals
//!
//! 2. Microbubble Detection (per frame)
//!    ├── Find local maxima above threshold
//!    ├── Reject edge artifacts
//!    └── Estimate ~500-2000 bubbles/frame
//!
//! 3. Localization (per detection)
//!    ├── Extract region of interest (ROI) around maximum
//!    ├── 2D Gaussian fitting: I(x,y) = A·exp(-[(x-x₀)²/σ_x² + (y-y₀)²/σ_y²])
//!    ├── Extract center (x₀, y₀) with sub-pixel precision
//!    └── Compute localization uncertainty from fit
//!
//! 4. Tracking (across frames within block)
//!    ├── Frame-to-frame association via Hungarian algorithm
//!    ├── Cost matrix: Euclidean distance between detections
//!    ├── Maximum linking distance: d_max (e.g., 50 μm)
//!    └── Build trajectory chains
//!
//! 5. Track Filtering
//!    ├── Minimum track length: L_min (e.g., 3 frames)
//!    ├── Sliding average smoothing (window=3-5 frames)
//!    └── Interpolate missing detections
//!
//! 6. Super-Resolution Reconstruction
//!    ├── Accumulate all localizations into high-res grid
//!    ├── Pixel size: 5 μm (20× finer than acquisition)
//!    ├── Render as 2D histogram or Gaussian splatting
//!    └── Normalize by acquisition time for density
//! ```
//!
//! # Mathematical Framework
//!
//! ## Microbubble Localization Precision
//!
//! Thompson's formula (adapted from fluorescence microscopy):
//! ```text
//! σ_loc² = (s² + a²/12) / N + (8πs⁴b²) / (a²N²)
//!
//! where:
//! s = PSF standard deviation (≈ λ/(2NA) ≈ 50 μm for 15 MHz)
//! a = pixel size (100 μm)
//! N = signal photons (SNR proxy)
//! b = background photons
//! ```
//!
//! For high SNR (>20 dB) and small pixels:
//! ```text
//! σ_loc ≈ s / √N ≈ 50 μm / √100 ≈ 5 μm
//! ```
//!
//! ## Hungarian Algorithm for Tracking
//!
//! **Assignment Problem:**
//! ```text
//! Minimize: Σᵢⱼ c_ij · x_ij
//!
//! Subject to:
//! Σⱼ x_ij = 1  (each bubble in frame t assigned once)
//! Σᵢ x_ij ≤ 1  (each bubble in frame t+1 assigned at most once)
//! x_ij ∈ {0,1}
//!
//! where:
//! c_ij = distance between bubble i (frame t) and j (frame t+1)
//! x_ij = 1 if i→j assignment, 0 otherwise
//! ```
//!
//! **Cost Matrix:**
//! ```text
//! C[i,j] = ||p_i(t) - p_j(t+1)||  if distance < d_max
//!        = ∞                       otherwise
//! ```
//!
//! ## Super-Resolution Rendering
//!
//! **Histogram Accumulation:**
//! ```text
//! I_SR[x,y] = Σₖ δ(x - x_k, y - y_k)
//!
//! where (x_k, y_k) are localized positions
//! δ = Dirac delta (or Gaussian for rendering)
//! ```
//!
//! **Gaussian Rendering:**
//! ```text
//! I_SR[x,y] = Σₖ (1/(2πσ²)) · exp(-[(x-x_k)² + (y-y_k)²] / (2σ²))
//!
//! where σ ≈ σ_loc (localization precision)
//! ```
//!
//! # Performance Specifications
//!
//! From Nouhoum et al. (2021):
//!
//! | Parameter | Value | Notes |
//! |-----------|-------|-------|
//! | Acquisition frame rate | 1000 Hz | 9 plane wave angles |
//! | Plane wave angles | -8° to +8° | 2° steps |
//! | Blocks acquired | 600 | 400 frames each |
//! | Total frames | 240,000 | 240 seconds |
//! | Bubbles detected | ~310,000 | Accumulated |
//! | Final resolution | 5 μm | Pixel size |
//! | Improvement factor | 20× | vs 100 μm Doppler |
//!
//! ## Resolution Comparison
//!
//! | Modality | In-Plane Resolution | Notes |
//! |----------|-------------------|-------|
//! | Power Doppler | 100 μm | Diffraction-limited |
//! | ULM | 5 μm | Super-resolution |
//! | **Improvement** | **20×** | Below diffraction limit |
//!
//! # Literature References
//!
//! **Foundational ULM:**
//! - Errico, C., et al. (2015). "Ultrafast ultrasound localization microscopy for deep super-resolution vascular imaging."
//!   *Nature*, 527(7579), 499-502. DOI: 10.1038/nature16066
//!
//! - Couture, O., et al. (2018). "Ultrasound localization microscopy and super-resolution: A state of the art."
//!   *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*, 65(8), 1304-1320.
//!   DOI: 10.1109/TUFFC.2018.2850811
//!
//! **Microbubble Tracking:**
//! - Kuhn, H. W. (1955). "The Hungarian method for the assignment problem."
//!   *Naval Research Logistics Quarterly*, 2(1-2), 83-97. DOI: 10.1002/nav.3800020109
//!
//! - Jaqaman, K., et al. (2008). "Robust single-particle tracking in live-cell time-lapse sequences."
//!   *Nature Methods*, 5(8), 695-702. DOI: 10.1038/nmeth.1237
//!
//! **Performance Benchmarking:**
//! - Heiles, B., et al. (2022). "Performance benchmarking of microbubble-localization algorithms for ultrasound localization microscopy."
//!   *Nature Biomedical Engineering*, 6(5), 605-616. DOI: 10.1038/s41551-021-00824-8
//!
//! **Clinical Applications:**
//! - Christensen-Jeffries, K., et al. (2020). "Super-resolution ultrasound imaging."
//!   *Ultrasound in Medicine & Biology*, 46(4), 865-891. DOI: 10.1016/j.ultrasmedbio.2019.11.013
//!
//! **Microbubble Physics:**
//! - De Jong, N., Hoff, L., Skotland, T., & Bom, N. (1992). "Absorption and scatter of encapsulated gas filled microspheres."
//!   *Ultrasonics*, 30(2), 95-103. DOI: 10.1016/0041-624X(92)90041-J
//!
//! # Module Organization
//!
//! - `microbubble_detection`: Detection and localization algorithms
//! - `microbubble_detection`: Detection and localization algorithms
//! - `tracking`: Hungarian assignment and trajectory linking
//! - `super_resolution`: Accumulated rendering and visualization
//! - `velocity_mapping`: Flow quantification from tracks
//!
//! # Implementation Status
//!
//! ## Implemented modules
//! - `microbubble_detection`: SVD clutter filter, Gaussian localiser, ULM detector
//! - `tracking`: Hungarian tracker, trajectory reconstruction
//! - `super_resolution`: 2D histogram + Gaussian splat accumulation, sliding smoothing, density normalization
//! - `velocity_mapping`: velocity field reconstruction, wall shear stress estimation

pub mod microbubble_detection;
pub mod super_resolution;
pub mod tracking;
pub mod velocity_mapping;

pub use microbubble_detection::{
    BubbleDetection, GaussianLocalizationConfig, GaussianLocalizer, SvdClutterConfig,
    UlmSvdClutterFilter, UlmDetector,
};
pub use super_resolution::{RenderMode, SuperResConfig, SuperResReconstructor};
pub use tracking::{BubbleTrack, HungarianTracker, TrackingConfig};
pub use velocity_mapping::{VelocityMapConfig, VelocityMapper};

#[cfg(test)]
mod tests {
    #[test]
    fn test_ulm_module() {
        let _ = super::SvdClutterConfig::default();
        let _ = super::GaussianLocalizationConfig::default();
        let _ = super::TrackingConfig::default();
    }
}
