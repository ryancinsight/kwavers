//! Functional Ultrasound (fUS) Imaging
//!
//! This module implements functional ultrasound imaging capabilities for neurovascular
//! imaging and brain activity mapping, including the vascular-based brain positioning
//! system (BPS) for automatic neuronavigation.
//!
//! # Overview
//!
//! Functional ultrasound imaging uses ultrafast Doppler to detect cerebral blood volume (CBV)
//! changes as a proxy for neural activity, enabling real-time brain functional imaging with
//! high spatiotemporal resolution.
//!
//! TODO_AUDIT: P1 - Functional Ultrasound Brain GPS System - Implement complete vascular-based neuronavigation
//! DEPENDS ON: clinical/imaging/functional_ultrasound/neuronavigation/brain_gps.rs (to be created)
//! DEPENDS ON: clinical/imaging/functional_ultrasound/registration/vascular_registration.rs (to be created)
//! DEPENDS ON: domain/sensor/ultrafast/plane_wave.rs (to be created)
//! DEPENDS ON: clinical/imaging/doppler/power_doppler.rs (enhance existing)
//! MISSING: Automatic affine registration using Mattes mutual information
//! MISSING: Evolutionary optimizer for parameter optimization
//! MISSING: Vascular atlas integration (Allen Mouse Brain CCF compatibility)
//! MISSING: Inverse kinematics solver for probe positioning
//! MISSING: Real-time registration (~1 minute for whole brain)
//! SEVERITY: HIGH (enables precise neuroscience experimentation)
//! ACCURACY: Target 44 μm intra-animal, 96 μm inter-animal positioning error
//! REFERENCES: Nouhoum et al. (2021) "A functional ultrasound brain GPS for automatic vascular-based neuronavigation"
//! REFERENCES: Scientific Reports 11:15197. DOI: 10.1038/s41598-021-94764-7
//! REFERENCES: https://pmc.ncbi.nlm.nih.gov/articles/PMC8313708/
//!
//! TODO_AUDIT: P1 - Ultrafast Power Doppler Imaging - Implement high-sensitivity vascular imaging
//! DEPENDS ON: domain/sensor/ultrafast/plane_wave.rs (to be created)
//! DEPENDS ON: analysis/signal_processing/clutter_filter/svd_filter.rs (to be created)
//! MISSING: 11 tilted plane wave compounding (-10° to +10° in 2° steps)
//! MISSING: 500 Hz compounded frame rate (5500 Hz PRF)
//! MISSING: Spatiotemporal SVD clutter filter for blood/tissue discrimination
//! MISSING: 200 frame block processing for Power Doppler
//! MISSING: Transcranial imaging through intact skull
//! SEVERITY: HIGH (foundation for fUS imaging)
//! PERFORMANCE: 100 μm × 100 μm in-plane resolution, 400 μm slice thickness
//! REFERENCES: Nouhoum et al. (2021) Section "Ultrafast Doppler imaging"
//!
//! TODO_AUDIT: P1 - Ultrasound Localization Microscopy (ULM) - Implement super-resolution vascular imaging
//! DEPENDS ON: clinical/imaging/functional_ultrasound/ulm/microbubble_tracking.rs (to be created)
//! DEPENDS ON: clinical/imaging/functional_ultrasound/ulm/super_resolution.rs (to be created)
//! MISSING: Microbubble detection and tracking (Hungarian algorithm)
//! MISSING: 1000 Hz frame rate acquisition (9 angle plane waves -8° to +8°)
//! MISSING: Sliding average smoothing and interpolation
//! MISSING: 5 μm pixel super-resolution reconstruction
//! MISSING: SonoVue microbubble contrast agent simulation
//! SEVERITY: MEDIUM (validation and super-resolution capability)
//! PERFORMANCE: 5 μm resolution from ~310,000 detected bubbles
//! REFERENCES: Nouhoum et al. (2021) Section "Ultrasound localization microscopy"
//!
//! TODO_AUDIT: P2 - Functional Connectivity Analysis - Implement brain network analysis
//! DEPENDS ON: clinical/imaging/functional_ultrasound/analysis/functional_connectivity.rs (to be created)
//! MISSING: Generalized linear model (GLM) for Z-score computation
//! MISSING: Bonferroni correction for multiple comparisons
//! MISSING: Pearson correlation for connectivity matrices
//! MISSING: Region of interest (ROI) extraction from anatomical templates
//! SEVERITY: MEDIUM (enables neuroscience applications)
//! REFERENCES: Nouhoum et al. (2021) Section "Functional imaging"
//!
//! # Architecture
//!
//! ```text
//! Functional Ultrasound Pipeline:
//!
//! 1. Ultrafast Plane Wave Transmission
//!    ├── 11 tilted angles (-10° to +10°)
//!    ├── 5500 Hz pulse repetition frequency
//!    └── 500 Hz compounded frame rate
//!
//! 2. Power Doppler Processing
//!    ├── SVD clutter filtering
//!    ├── 200 frame blocks
//!    └── Blood flow discrimination
//!
//! 3. Vascular Registration (Brain GPS)
//!    ├── Automatic affine registration
//!    ├── Mattes mutual information metric
//!    ├── Evolutionary optimization
//!    └── Atlas alignment (Allen CCF)
//!
//! 4. Neuronavigation
//!    ├── Virtual plane definition
//!    ├── Inverse kinematics
//!    └── Automatic probe positioning
//!
//! 5. Functional Imaging
//!    ├── CBV change detection
//!    ├── GLM statistical analysis
//!    └── Connectivity matrices
//! ```
//!
//! # Key Features
//!
//! - **Ultrafast Imaging**: 500 Hz Doppler frame rate via plane wave compounding
//! - **Automatic Registration**: Sub-100 μm positioning accuracy
//! - **Transcranial**: Non-invasive imaging through intact skull
//! - **Real-time**: ~1 minute registration for whole brain vasculature
//! - **Super-resolution**: 5 μm ULM reconstruction (vs 100 μm Doppler)
//!
//! # Clinical Applications
//!
//! - **Neuroscience Research**: Precise anatomical targeting for experiments
//! - **Brain Mapping**: Functional connectivity and network analysis
//! - **Preclinical Studies**: Small animal neurovascular imaging
//! - **Surgical Planning**: Vascular anatomy mapping
//!
//! # Performance Specifications
//!
//! Based on Nouhoum et al. (2021):
//!
//! | Metric | Intra-Animal | Inter-Animal | Expert Manual |
//! |--------|--------------|--------------|---------------|
//! | Registration Error | 44 ± 32 μm | 96 ± 69 μm | 215-259 μm |
//! | Correlation (ρ) | 0.9 | 0.64 | N/A |
//! | Processing Time | ~1 minute | ~1 minute | ~5 minutes |
//!
//! # Literature References
//!
//! **Primary Reference:**
//! - Nouhoum, M., Ferrier, J., Osmanski, B.-F., Ialy-Radio, N., Pezet, S., Tanter, M., & Deffieux, T. (2021).
//!   "A functional ultrasound brain GPS for automatic vascular-based neuronavigation."
//!   *Scientific Reports*, 11(1), 15197. DOI: 10.1038/s41598-021-94764-7
//!
//! **Foundational Work:**
//! - Macé, E., et al. (2011). "Functional ultrasound imaging of the brain."
//!   *Nature Methods*, 8(8), 662-664. DOI: 10.1038/nmeth.1641
//! - Deffieux, T., et al. (2018). "Functional ultrasound neuroimaging: a review of the preclinical and clinical state of the art."
//!   *Current Opinion in Neurobiology*, 50, 128-135. DOI: 10.1016/j.conb.2018.02.001
//!
//! **Ultrasound Localization Microscopy:**
//! - Errico, C., et al. (2015). "Ultrafast ultrasound localization microscopy for deep super-resolution vascular imaging."
//!   *Nature*, 527(7579), 499-502. DOI: 10.1038/nature16066
//!
//! **Registration Methods:**
//! - Mattes, D., et al. (2003). "PET-CT image registration in the chest using free-form deformations."
//!   *IEEE Transactions on Medical Imaging*, 22(1), 120-128. DOI: 10.1109/TMI.2003.809072
//!
//! # Module Organization
//!
//! - `neuronavigation`: Brain GPS and automatic probe positioning (planned - see ulm/mod.rs)
//! - `registration`: Vascular atlas registration and alignment (planned - see registration/mod.rs)
//! - `ulm`: Ultrasound localization microscopy for super-resolution imaging (planned - see ulm/mod.rs)
//! - `analysis`: Functional connectivity and statistical analysis (planned)

// Module stub declarations - awaiting implementation
// See individual module files for TODO_AUDIT tracking:
// - registration/mod.rs: TODO_AUDIT P1 items for atlas registration
// - ulm/mod.rs: TODO_AUDIT P1 items for super-resolution imaging
// pub mod neuronavigation; // Not yet created
// pub mod registration;    // Stub with TODO_AUDIT markers
// pub mod ulm;             // Stub with TODO_AUDIT markers
// pub mod analysis;        // Not yet created

