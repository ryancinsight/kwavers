//! Vascular Registration for Brain Positioning System
//!
//! This module implements automatic registration of ultrasound vascular images to
//! anatomical atlases for precise neuronavigation and standardized brain coordinate systems.
//!
//! # Overview
//!
//! The Brain Positioning System (BPS) automatically aligns Power Doppler vascular images
//! to reference atlases (e.g., Allen Mouse Brain CCF) with sub-100 μm accuracy, enabling
//! reproducible targeting of brain structures across animals and experiments.
//!
//! ## Core Registration (Implemented — requires `ritk` feature)
//!
//! When compiled with `features = ["ritk"]` the following are available via
//! the `ritk` submodule:
//! - **Mattes MI metric** (`MattesMutualInformation`): B-spline Parzen windows,
//!   random spatial sampling, MI(A,B) = H(A)+H(B)−H(A,B)  (Mattes et al. 2003)
//! - **CMA-ES optimizer** (`CmaEsOptimizer`): 12-parameter affine search,
//!   covariance matrix adaptation (Hansen & Ostermeier 2001)
//! - **Affine transform** (`AffineTransform`): 12 DOF (rotation, translation,
//!   scaling, shearing)
//! - **Registration framework** (`Registration`): multi-resolution pyramid,
//!   full brain registration in ~1 minute, 44 µm intra-animal accuracy
//!
//! ## Not yet implemented
//!
//! - **B-spline deformable registration** (Rueckert et al. 1999): free-form deformation
//!   control point grid, regularized non-rigid warping for inter-subject variability.
//!   Referenced in Nouhoum et al. (2021) via Slicer3D/elastix.
//!
//! - **Atlas integration**: Allen Mouse Brain CCF (Wang et al. 2020), Dorr vascular
//!   atlas intermediate reference, 70+ ROI ontology, NIfTI I/O.
//!
//! - **Inverse kinematics for probe positioning**: 6-DOF IK solver, virtual imaging
//!   plane targeting, collision avoidance, motion planning (Nouhoum et al. 2021;
//!   Craig 2005).
//!
//! # Registration Pipeline
//!
//! ## Offline Reference Creation (Once per template animal)
//!
//! ```text
//! 1. Acquire high-resolution 3D Doppler volume
//!    └── 19 rotations for isotropic reconstruction
//!
//! 2. Semi-automatic alignment to structural atlas
//!    ├── Manual landmark identification
//!    ├── B-spline deformable registration
//!    └── Create reference Doppler atlas
//!
//! 3. Link to Allen CCF anatomical template
//!    └── Store transformation matrices
//! ```
//!
//! ## Online Registration (Per experiment, <1 minute)
//!
//! ```text
//! 1. Acquire quick 3D Power Doppler scan
//!    └── Single sweep (~10 seconds)
//!
//! 2. Automatic affine registration
//!    ├── Initialize near identity transform
//!    ├── Mattes MI metric evaluation
//!    ├── CMA-ES optimization (12 parameters)
//!    └── Multi-resolution refinement
//!
//! 3. Transform to Allen CCF coordinates
//!    ├── Propagate reference → atlas mapping
//!    └── Obtain standardized brain coordinates
//!
//! 4. (Optional) Inverse kinematics
//!    ├── User selects target plane in atlas
//!    ├── Compute required probe position
//!    └── Move motorized stage automatically
//! ```
//!
//! # Mathematical Framework
//!
//! ## Affine Transformation
//!
//! Transform point **p** to **p'**:
//! ```text
//! p' = A * p + t
//!
//! where A is 3×3 matrix encoding:
//! - Rotation: R_x(θ_x) * R_y(θ_y) * R_z(θ_z)
//! - Scaling: diag(s_x, s_y, s_z)
//! - Shearing: off-diagonal terms
//!
//! and t is 3×1 translation vector
//!
//! Total: 12 parameters to optimize
//! ```
//!
//! ## Mutual Information Metric
//!
//! **Shannon Entropy:**
//! ```text
//! H(X) = -Σ p(x) log p(x)
//! ```
//!
//! **Joint Entropy:**
//! ```text
//! H(X,Y) = -Σ p(x,y) log p(x,y)
//! ```
//!
//! **Mutual Information:**
//! ```text
//! MI(X,Y) = H(X) + H(Y) - H(X,Y)
//!         = Σ p(x,y) log[p(x,y) / (p(x)p(y))]
//! ```
//!
//! **Mattes Approximation:**
//! - Use spatial sampling: Only N_samples pixels (e.g., 10% of image)
//! - Parzen window density estimation: Smooth histogram with Gaussian kernel
//! - B-spline interpolation for gradient computation
//!
//! ## Optimization Objective
//!
//! ```text
//! θ* = argmax MI(I_fixed, T(I_moving; θ))
//!
//! where:
//! - I_fixed: Reference vascular atlas
//! - I_moving: Current acquisition
//! - T(·; θ): Transformation with parameters θ
//! ```
//!
//! # Performance Specifications
//!
//! From Nouhoum et al. (2021):
//!
//! ## Registration Accuracy (using ULM validation)
//!
//! **Intra-Animal (same mouse, n=5):**
//! - Lateral (X): 44 ± 32 μm
//! - Elevation (Y): 31 ± 23 μm
//! - Axial (Z): 21 ± 10 μm
//! - Overall: 44 μm mean error
//!
//! **Inter-Animal (different mice, n=6):**
//! - Lateral (X): 74 ± 38 μm
//! - Elevation (Y): 96 ± 69 μm
//! - Axial (Z): 50 ± 29 μm
//! - Overall: 96 μm mean error
//!
//! **Comparison to Expert Manual:**
//! - Manual inter-annotator error: 215-259 μm
//! - **Automatic is 2-3× more accurate than manual**
//!
//! ## Computational Performance
//!
//! - Registration time: ~1 minute (full brain vasculature)
//! - Manual annotation: ~5 minutes (4 landmarks only)
//! - **5× faster with better accuracy**
//!
//! # Literature References
//!
//! **Primary Method:**
//! - Nouhoum, M., et al. (2021). "A functional ultrasound brain GPS for automatic vascular-based neuronavigation."
//!   *Scientific Reports*, 11, 15197. DOI: 10.1038/s41598-021-94764-7
//!
//! **Mutual Information Registration:**
//! - Mattes, D., et al. (2003). "PET-CT image registration in the chest using free-form deformations."
//!   *IEEE Transactions on Medical Imaging*, 22(1), 120-128. DOI: 10.1109/TMI.2003.809072
//!
//! - Pluim, J. P. W., Maintz, J. B. A., & Viergever, M. A. (2003). "Mutual-information-based registration of medical images: a survey."
//!   *IEEE Transactions on Medical Imaging*, 22(8), 986-1004. DOI: 10.1109/TMI.2003.815867
//!
//! **Evolutionary Optimization:**
//! - Hansen, N., & Ostermeier, A. (2001). "Completely derandomized self-adaptation in evolution strategies."
//!   *Evolutionary Computation*, 9(2), 159-195. DOI: 10.1162/106365601750190398
//!
//! **Deformable Registration:**
//! - Rueckert, D., et al. (1999). "Nonrigid registration using free-form deformations: application to breast MR images."
//!   *IEEE Transactions on Medical Imaging*, 18(8), 712-721. DOI: 10.1109/42.796284
//!
//! **Brain Atlases:**
//! - Wang, Q., et al. (2020). "The Allen Mouse Brain Common Coordinate Framework: A 3D reference atlas."
//!   *Cell*, 181(4), 936-953. DOI: 10.1016/j.cell.2020.04.007
//!
//! - Dorr, A. E., Lerch, J. P., Spring, S., Kabani, N., & Henkelman, R. M. (2008). "High resolution three-dimensional brain atlas using an average magnetic resonance image of 40 adult C57Bl/6J mice."
//!   *NeuroImage*, 42(1), 60-69. DOI: 10.1016/j.neuroimage.2008.03.037
//!
//! # Module Organization
//!
//! - `mattes_mi`: Mattes mutual information metric
//! - `affine_transform`: Affine transformation and composition
//! - `evolutionary_optimizer`: CMA-ES parameter optimization
//! - `bspline_deform`: B-spline deformable registration (future)
//! - `atlas`: Anatomical atlas integration (future)
//! - `inverse_kinematics`: Probe positioning solver (future)

// Future modules (not yet implemented):
// pub mod mattes_mi;           // Mattes mutual information metric
// pub mod affine_transform;    // Affine transformation and composition
// pub mod evolutionary_optimizer; // CMA-ES parameter optimization
// pub mod bspline_deform;      // B-spline deformable registration
// pub mod atlas;               // Anatomical atlas integration
// pub mod inverse_kinematics;  // Probe positioning solver

/// Re-exports from the ritk medical image registration toolkit.
///
/// Enabled by the `ritk` Cargo feature. Provides:
/// - **Metrics**: Mattes mutual information (B-spline Parzen windows, Mattes et al. 2003),
///   normalized cross-correlation, standard mutual information
/// - **Optimizers**: CMA-ES evolutionary strategy (Hansen & Ostermeier 2001), Adam
/// - **Transforms**: Affine (12 DOF) and B-spline free-form deformation
/// - **Registration**: Full registration framework with multi-resolution support
///
/// # Feature Gate
///
/// Add `features = ["ritk"]` to the kwavers dependency to enable these re-exports.
///
/// # References
///
/// - Mattes, D., et al. (2003). *IEEE Trans. Med. Imaging* 22(1):120-128.
///   DOI: 10.1109/TMI.2003.809072
/// - Hansen, N., & Ostermeier, A. (2001). *Evol. Comput.* 9(2):159-195.
///   DOI: 10.1162/106365601750190398
// ritk is now a mandatory dep (see Cargo.toml); the feature gate is dropped.
pub mod ritk {
    pub use ritk_registration::metric::{
        MutualInformation, NormalizedCrossCorrelation, MattesMutualInformation,
    };
    pub use ritk_registration::optimizer::{
        AdamOptimizer, CmaEsConfig, CmaEsOptimizer, CmaEsResult, StopReason,
    };
    pub use ritk_registration::registration::Registration;
    pub use ritk_core::transform::{AffineTransform, BSplineTransform};
}
