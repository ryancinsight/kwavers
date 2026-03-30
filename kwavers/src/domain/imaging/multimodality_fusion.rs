//! Multi-Modality Medical Image Fusion
//!
//! This module provides comprehensive support for registering and fusing multiple
//! medical imaging modalities (CT, MR, ultrasound) for improved diagnostics and therapy.
//!
//! ## Image Fusion Workflow
//!
//! ```text
//! Reference Image (CT/MR)
//!        ↓
//! Feature Extraction → Registration → Transform → Fusion → Output
//!        ↓                ↓             ↓         ↓
//!    Landmarks      Affine/Deformable Warp    Blending
//!    Edges          B-spline                   Overlay
//!    Intensity      Thin-plate spline          False color
//! ```
//!
//! ## Supported Modalities
//!
//! - **CT**: Dense 3D anatomical reference
//! - **MR**: High soft-tissue contrast
//! - **Ultrasound**: Real-time functional imaging
//! - **PET/SPECT**: Metabolic/functional data
//!
//! ## Registration Methods
//!
//! 1. **Rigid (6 DOF)**: Translation + rotation
//! 2. **Affine (12 DOF)**: Includes scaling + shear
//! 3. **Non-rigid**: Deformable registration for anatomical variations
//!
//! ## Fusion Output Modes
//!
//! - **Overlay**: Transparent overlay of floating image
//! - **Checkerboard**: Alternating tiles (alignment verification)
//! - **Difference**: Subtraction (change detection)
//! - **False Color**: Color-coded fusion
//!
//! ## References
//!
//! - Maintz & Viergever (1998): "A survey of medical image registration"
//! - Hill et al. (2001): "Medical image registration"
//! - Rueckert et al. (1999): "Non-rigid registration using free-form deformations"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array2, Array3};
use std::collections::HashMap;

/// Multi-modality imaging session
#[derive(Debug, Clone)]
pub struct MultimodalitySession {
    /// Session identifier
    pub session_id: String,
    /// Reference image (CT or MR)
    pub reference_image: Option<ImageData>,
    /// Floating image (ultrasound, PET, or other)
    pub floating_image: Option<ImageData>,
    /// Registration transformation
    pub transformation: Option<RegistrationTransform>,
    /// Fusion parameters
    pub fusion_params: FusionParameters,
}

/// Medical image data with metadata
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Image modality type
    pub modality: ImageModality,
    /// 3D image array (nx, ny, nz)
    pub data: Array3<f64>,
    /// Voxel spacing (dx, dy, dz) in mm
    pub voxel_spacing_mm: (f64, f64, f64),
    /// Image origin (x, y, z) in mm
    pub origin_mm: (f64, f64, f64),
    /// Intensity range (min, max)
    pub intensity_range: (f64, f64),
    /// Image dimensions
    pub dimensions: (usize, usize, usize),
}

/// Image modality type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageModality {
    /// Computed Tomography
    CT,
    /// Magnetic Resonance
    MR,
    /// Ultrasound
    Ultrasound,
    /// Positron Emission Tomography
    PET,
    /// Single Photon Emission Computed Tomography
    SPECT,
    /// Other modality
    Other,
}

impl std::fmt::Display for ImageModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CT => write!(f, "CT"),
            Self::MR => write!(f, "MR"),
            Self::Ultrasound => write!(f, "Ultrasound"),
            Self::PET => write!(f, "PET"),
            Self::SPECT => write!(f, "SPECT"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// Registration transformation (rigid or affine)
#[derive(Debug, Clone)]
pub struct RegistrationTransform {
    /// Transformation type
    pub transform_type: TransformationType,
    /// 4×4 transformation matrix (homogeneous coordinates)
    pub matrix: Array2<f64>,
    /// Registration error (RMS distance after alignment)
    pub registration_error_mm: f64,
    /// Number of iterations until convergence
    pub iterations: usize,
}

/// Transformation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformationType {
    /// Rigid (6 DOF): translation + rotation
    Rigid,
    /// Affine (12 DOF): rigid + scaling + shear
    Affine,
    /// Non-rigid (many DOF): deformable registration
    NonRigid,
}

impl RegistrationTransform {
    /// Create identity transformation
    pub fn identity() -> Self {
        let mut matrix = Array2::zeros((4, 4));
        for i in 0..4 {
            matrix[[i, i]] = 1.0;
        }

        Self {
            transform_type: TransformationType::Rigid,
            matrix,
            registration_error_mm: 0.0,
            iterations: 0,
        }
    }

    /// Apply transformation to 3D point
    pub fn apply_to_point(&self, point: (f64, f64, f64)) -> (f64, f64, f64) {
        // Convert to homogeneous coordinates
        let p = [point.0, point.1, point.2, 1.0];

        // Apply transformation
        let mut result = [0.0; 4];
        for (i, item) in result.iter_mut().enumerate() {
            *item = (0..4).map(|j| self.matrix[[i, j]] * p[j]).sum();
        }

        // Convert back to 3D
        (
            result[0] / result[3],
            result[1] / result[3],
            result[2] / result[3],
        )
    }

    /// Invert a homogeneous rigid-body transformation.
    ///
    /// ## Theorem — Inverse of a 4×4 Homogeneous Rigid Transform
    ///
    /// A rigid-body transform in homogeneous coordinates is:
    /// ```text
    ///   T = [ R  t ]    where R ∈ SO(3), t ∈ ℝ³
    ///       [ 0  1 ]
    /// ```
    ///
    /// Its exact inverse is:
    /// ```text
    ///   T⁻¹ = [ Rᵀ   −Rᵀ·t ]
    ///         [ 0       1   ]
    /// ```
    ///
    /// **Proof**: T · T⁻¹ =
    ///   [ R  t ] · [ Rᵀ  −Rᵀt ] = [ R·Rᵀ  −R·Rᵀt + t ] = [ I  0 ] = I₄  ✓
    ///   [ 0  1 ]   [ 0     1  ]   [ 0         1       ]   [ 0  1 ]
    ///
    /// Since R ∈ SO(3), R·Rᵀ = I₃ (orthogonality) and det(R) = 1.
    ///
    /// This formula is exact (no numerical LU decomposition needed) and produces
    /// a result that is guaranteed to lie in SE(3) — the naive LU inverse of a
    /// rotation matrix can drift out of SO(3) due to floating-point rounding.
    ///
    /// ## Reference
    ///
    /// Craig, J.J. (2005). *Introduction to Robotics: Mechanics and Control*.
    /// 3rd ed. Pearson. §2.3, Eq. 2.45.
    pub fn invert(&self) -> KwaversResult<Self> {
        // Extract R (3×3 upper-left) and t (column 3, rows 0..3) from the 4×4 matrix.
        // Matrix is stored row-major: self.matrix[[row, col]].
        let m = &self.matrix;

        // Rᵀ (3×3): element [i,j] of Rᵀ = element [j,i] of R
        let mut rt = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rt[i][j] = m[[j, i]];
            }
        }

        // −Rᵀ·t (3×1)
        let t = [m[[0, 3]], m[[1, 3]], m[[2, 3]]];
        let mut neg_rt_t = [0.0_f64; 3];
        for i in 0..3 {
            neg_rt_t[i] = -(rt[i][0] * t[0] + rt[i][1] * t[1] + rt[i][2] * t[2]);
        }

        // Assemble 4×4 inverse matrix
        let mut inv = Array2::<f64>::zeros((4, 4));
        for i in 0..3 {
            for j in 0..3 {
                inv[[i, j]] = rt[i][j];
            }
            inv[[i, 3]] = neg_rt_t[i];
        }
        inv[[3, 3]] = 1.0;

        Ok(Self {
            transform_type: self.transform_type,
            matrix: inv,
            registration_error_mm: self.registration_error_mm,
            iterations: self.iterations,
        })
    }
}

/// Image fusion parameters
#[derive(Debug, Clone)]
pub struct FusionParameters {
    /// Fusion method
    pub method: FusionMethod,
    /// Blend weight for floating image (0-1)
    pub blend_weight: f64,
    /// Enable automatic contrast adjustment
    pub auto_contrast: bool,
    /// Output intensity scaling
    pub output_range: (f64, f64),
}

impl Default for FusionParameters {
    fn default() -> Self {
        Self {
            method: FusionMethod::Overlay,
            blend_weight: 0.5,
            auto_contrast: true,
            output_range: (0.0, 255.0),
        }
    }
}

/// Image fusion method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMethod {
    /// Simple overlay with transparency
    Overlay,
    /// Checkerboard pattern (alternating tiles)
    Checkerboard,
    /// Difference (subtraction)
    Difference,
    /// False color (color-coded)
    FalseColor,
    /// Multi-channel (R=ref, G=float, B=diff)
    MultiChannel,
}

impl std::fmt::Display for FusionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overlay => write!(f, "Overlay"),
            Self::Checkerboard => write!(f, "Checkerboard"),
            Self::Difference => write!(f, "Difference"),
            Self::FalseColor => write!(f, "False Color"),
            Self::MultiChannel => write!(f, "Multi-Channel"),
        }
    }
}

/// Multi-modality image registration engine
#[derive(Debug)]
pub struct RegistrationEngine {
    /// Registration parameters
    #[allow(dead_code)] // Used when registration optimization is implemented
    params: RegistrationParams,
}

/// Registration parameters
#[derive(Debug, Clone)]
pub struct RegistrationParams {
    /// Transformation type
    pub transform_type: TransformationType,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance (mm)
    pub tolerance_mm: f64,
    /// Use multi-resolution (coarse-to-fine)
    pub multi_resolution: bool,
    /// Number of resolution levels
    pub num_resolution_levels: usize,
}

impl Default for RegistrationParams {
    fn default() -> Self {
        Self {
            transform_type: TransformationType::Rigid,
            max_iterations: 100,
            tolerance_mm: 0.1,
            multi_resolution: true,
            num_resolution_levels: 3,
        }
    }
}

impl RegistrationEngine {
    /// Create new registration engine
    pub fn new(params: RegistrationParams) -> Self {
        Self { params }
    }

    /// Register floating image to reference image
    ///
    /// # Errors
    /// Returns `KwaversError::NotImplemented` — mutual information registration pending.
    pub fn register(
        &self,
        reference: &ImageData,
        floating: &ImageData,
    ) -> KwaversResult<RegistrationTransform> {
        // Validate inputs
        if reference.dimensions != floating.dimensions {
            return Err(KwaversError::InvalidInput(
                "Reference and floating images must have same dimensions".to_string(),
            ));
        }

        Err(KwaversError::NotImplemented(
            "Multimodality image registration not yet implemented. \
             Requires mutual information metric and gradient descent \
             optimization for rigid/affine transform estimation."
                .into(),
        ))
    }

    /// Register with landmark-based initialization
    ///
    /// # Errors
    /// Returns `KwaversError::NotImplemented` — Procrustes analysis pending.
    pub fn register_with_landmarks(
        &self,
        reference: &ImageData,
        floating: &ImageData,
        reference_landmarks: &[(f64, f64, f64)],
        floating_landmarks: &[(f64, f64, f64)],
    ) -> KwaversResult<RegistrationTransform> {
        if reference_landmarks.len() != floating_landmarks.len() {
            return Err(KwaversError::InvalidInput(
                "Landmark count mismatch".to_string(),
            ));
        }

        if reference.dimensions != floating.dimensions {
            return Err(KwaversError::InvalidInput(
                "Reference and floating images must have same dimensions".to_string(),
            ));
        }

        Err(KwaversError::NotImplemented(
            "Landmark-based registration not yet implemented. \
             Requires Procrustes analysis (least-squares rigid transform \
             from corresponding point sets)."
                .into(),
        ))
    }
}

/// Image fusion engine
#[derive(Debug)]
pub struct FusionEngine {
    /// Fusion parameters
    params: FusionParameters,
}

impl FusionEngine {
    /// Create new fusion engine
    pub fn new(params: FusionParameters) -> Self {
        Self { params }
    }

    /// Perform image fusion
    pub fn fuse(
        &self,
        reference: &ImageData,
        floating: &ImageData,
        transform: &RegistrationTransform,
    ) -> KwaversResult<Array3<f64>> {
        // Apply transformation to floating image
        let transformed = self.apply_transform(floating, transform)?;

        // Perform fusion based on method
        let fused = match self.params.method {
            FusionMethod::Overlay => self.fusion_overlay(&reference.data, &transformed),
            FusionMethod::Checkerboard => self.fusion_checkerboard(&reference.data, &transformed),
            FusionMethod::Difference => self.fusion_difference(&reference.data, &transformed),
            FusionMethod::FalseColor => self.fusion_false_color(&reference.data, &transformed),
            FusionMethod::MultiChannel => self.fusion_multi_channel(&reference.data, &transformed),
        };

        Ok(fused)
    }

    /// Apply transformation to floating image (nearest neighbor interpolation)
    fn apply_transform(
        &self,
        floating: &ImageData,
        transform: &RegistrationTransform,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = floating.dimensions;
        let mut transformed = Array3::zeros((nx, ny, nz));

        // Apply inverse transform to avoid holes
        let inv_transform = transform.invert()?;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Physical coordinates
                    let phys_x = i as f64 * floating.voxel_spacing_mm.0;
                    let phys_y = j as f64 * floating.voxel_spacing_mm.1;
                    let phys_z = k as f64 * floating.voxel_spacing_mm.2;

                    // Transform
                    let (tx, ty, tz) = inv_transform.apply_to_point((phys_x, phys_y, phys_z));

                    // Convert back to indices
                    let vi = (tx / floating.voxel_spacing_mm.0).round() as usize;
                    let vj = (ty / floating.voxel_spacing_mm.1).round() as usize;
                    let vk = (tz / floating.voxel_spacing_mm.2).round() as usize;

                    // Nearest neighbor interpolation with bounds check
                    if vi < nx && vj < ny && vk < nz {
                        transformed[[i, j, k]] = floating.data[[vi, vj, vk]];
                    }
                }
            }
        }

        Ok(transformed)
    }

    /// Overlay fusion: weighted blend
    fn fusion_overlay(&self, reference: &Array3<f64>, floating: &Array3<f64>) -> Array3<f64> {
        reference * (1.0 - self.params.blend_weight) + floating * self.params.blend_weight
    }

    /// Checkerboard fusion: alternating tiles
    fn fusion_checkerboard(&self, reference: &Array3<f64>, floating: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = reference.dim();
        let tile_size = 32; // 32×32 voxel tiles

        let mut result = reference.clone();
        for i in (0..nx).step_by(tile_size) {
            for j in (0..ny).step_by(tile_size) {
                // Alternate tiles
                if ((i / tile_size) + (j / tile_size)) % 2 == 0 {
                    let i_end = (i + tile_size).min(nx);
                    let j_end = (j + tile_size).min(ny);
                    for ii in i..i_end {
                        for jj in j..j_end {
                            for kk in 0..nz {
                                result[[ii, jj, kk]] = floating[[ii, jj, kk]];
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Difference fusion: subtraction (change detection)
    fn fusion_difference(&self, reference: &Array3<f64>, floating: &Array3<f64>) -> Array3<f64> {
        (floating - reference).mapv(f64::abs)
    }

    /// False color fusion: color-coded composite
    ///
    /// Proper false-color fusion requires mapping each modality to a separate
    /// color channel (e.g., R=CT, G=PET, B=MRI) and returning a 4D RGBA array.
    /// Current 3D output format cannot represent color information.
    fn fusion_false_color(&self, reference: &Array3<f64>, floating: &Array3<f64>) -> Array3<f64> {
        // Approximate: intensity-weighted blend (true color requires 4D output)
        let ref_norm = reference
            / (reference
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .max(1e-10));
        let flt_norm = floating
            / (floating
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .max(1e-10));
        &ref_norm * (1.0 - self.params.blend_weight) + &flt_norm * self.params.blend_weight
    }

    /// Multi-channel fusion: R=ref, G=float, B=diff
    ///
    /// True multi-channel output requires a 4D array [nx, ny, nz, 3].
    /// This returns a 3D weighted combination as an approximation.
    fn fusion_multi_channel(&self, reference: &Array3<f64>, floating: &Array3<f64>) -> Array3<f64> {
        // Weighted combination (true RGB requires 4D output format)
        reference * 0.4 + floating * 0.4 + (floating - reference).mapv(f64::abs) * 0.2
    }
}

/// Multimodality fusion session manager
#[derive(Debug)]
pub struct MultimodalityFusionManager {
    /// Active sessions
    sessions: HashMap<String, MultimodalitySession>,
    /// Registration engine
    registration_engine: RegistrationEngine,
    /// Fusion engine
    fusion_engine: FusionEngine,
}

impl MultimodalityFusionManager {
    /// Create new fusion manager
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            registration_engine: RegistrationEngine::new(RegistrationParams::default()),
            fusion_engine: FusionEngine::new(FusionParameters::default()),
        }
    }

    /// Create new fusion session
    pub fn create_session(&mut self, session_id: String) -> KwaversResult<()> {
        let session = MultimodalitySession {
            session_id: session_id.clone(),
            reference_image: None,
            floating_image: None,
            transformation: None,
            fusion_params: FusionParameters::default(),
        };
        self.sessions.insert(session_id, session);

        Ok(())
    }

    /// Load reference image into session
    pub fn load_reference(&mut self, session_id: &str, image_data: ImageData) -> KwaversResult<()> {
        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Session {} not found", session_id))
        })?;

        session.reference_image = Some(image_data);
        Ok(())
    }

    /// Load floating image into session
    pub fn load_floating(&mut self, session_id: &str, image_data: ImageData) -> KwaversResult<()> {
        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Session {} not found", session_id))
        })?;

        session.floating_image = Some(image_data);
        Ok(())
    }

    /// Register images in session
    pub fn register(&mut self, session_id: &str) -> KwaversResult<()> {
        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Session {} not found", session_id))
        })?;

        let reference = session
            .reference_image
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Reference image not loaded".to_string()))?;

        let floating = session
            .floating_image
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Floating image not loaded".to_string()))?;

        let transform = self.registration_engine.register(reference, floating)?;
        session.transformation = Some(transform);

        Ok(())
    }

    /// Perform fusion in session
    pub fn fuse(&self, session_id: &str) -> KwaversResult<Array3<f64>> {
        let session = self.sessions.get(session_id).ok_or_else(|| {
            KwaversError::InvalidInput(format!("Session {} not found", session_id))
        })?;

        let reference = session
            .reference_image
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Reference image not loaded".to_string()))?;

        let floating = session
            .floating_image
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Floating image not loaded".to_string()))?;

        let transform = session
            .transformation
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Images not yet registered".to_string()))?;

        self.fusion_engine.fuse(reference, floating, transform)
    }

    /// Get session
    pub fn get_session(&self, session_id: &str) -> Option<&MultimodalitySession> {
        self.sessions.get(session_id)
    }
}

impl Default for MultimodalityFusionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_modality_display() {
        assert_eq!(ImageModality::CT.to_string(), "CT");
        assert_eq!(ImageModality::MR.to_string(), "MR");
        assert_eq!(ImageModality::Ultrasound.to_string(), "Ultrasound");
    }

    #[test]
    fn test_registration_transform_identity() {
        let transform = RegistrationTransform::identity();
        assert_eq!(transform.transform_type, TransformationType::Rigid);

        // Identity transform should preserve coordinates
        let point = (10.0, 20.0, 30.0);
        let result = transform.apply_to_point(point);
        assert!((result.0 - point.0).abs() < 0.01);
        assert!((result.1 - point.1).abs() < 0.01);
        assert!((result.2 - point.2).abs() < 0.01);
    }

    #[test]
    fn test_fusion_parameters_default() {
        let params = FusionParameters::default();
        assert_eq!(params.blend_weight, 0.5);
        assert!(params.auto_contrast);
    }

    #[test]
    fn test_registration_engine_creation() {
        let params = RegistrationParams::default();
        let engine = RegistrationEngine::new(params);
        assert_eq!(engine.params.max_iterations, 100);
    }

    #[test]
    fn test_fusion_engine_creation() {
        let params = FusionParameters::default();
        let _engine = FusionEngine::new(params);
        // Engine created successfully
    }

    #[test]
    fn test_multimodality_fusion_manager() {
        let mut manager = MultimodalityFusionManager::new();
        assert!(manager.create_session("test_session".to_string()).is_ok());
        assert!(manager.get_session("test_session").is_some());
    }

    #[test]
    fn test_fusion_method_display() {
        assert_eq!(FusionMethod::Overlay.to_string(), "Overlay");
        assert_eq!(FusionMethod::Checkerboard.to_string(), "Checkerboard");
    }

    #[test]
    fn test_registration_params_default() {
        let params = RegistrationParams::default();
        assert_eq!(params.max_iterations, 100);
        assert!(params.multi_resolution);
    }

    /// Identity transform inverted must still be identity.
    ///
    /// T = I₄  ⟹  T⁻¹ = I₄  (Craig 2005, §2.3)
    #[test]
    fn test_invert_identity() {
        let id = RegistrationTransform::identity();
        let inv = id.invert().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv.matrix[[i, j]] - expected).abs() < 1e-14,
                    "identity.invert()[[{i},{j}]] = {}, expected {expected}",
                    inv.matrix[[i, j]]
                );
            }
        }
    }

    /// A pure translation T(d) inverted must give T(-d).
    ///
    /// T = [ I  d ]  ⟹  T⁻¹ = [ I  -d ]
    ///     [ 0  1 ]             [ 0   1 ]
    #[test]
    fn test_invert_pure_translation() {
        let dx = 5.0_f64;
        let mut t = RegistrationTransform::identity();
        t.matrix[[0, 3]] = dx;

        let inv = t.invert().unwrap();
        assert!(
            (inv.matrix[[0, 3]] - (-dx)).abs() < 1e-14,
            "translation inverse x = {}, expected {}",
            inv.matrix[[0, 3]], -dx
        );
        assert!((inv.matrix[[1, 3]]).abs() < 1e-14);
        assert!((inv.matrix[[2, 3]]).abs() < 1e-14);
    }

    /// Round-trip T · T⁻¹ ≈ I₄ for a combined rotation + translation.
    ///
    /// Uses a 90° rotation about the z-axis with a non-trivial translation.
    /// The round-trip error must be below f64 machine epsilon × grid_size (12).
    #[test]
    fn test_invert_round_trip() {
        use std::f64::consts::FRAC_PI_2;

        // 90° rotation about z: R_z(π/2) = [[0,-1,0],[1,0,0],[0,0,1]]
        let mut t = RegistrationTransform::identity();
        t.matrix[[0, 0]] = FRAC_PI_2.cos();
        t.matrix[[0, 1]] = -FRAC_PI_2.sin();
        t.matrix[[1, 0]] = FRAC_PI_2.sin();
        t.matrix[[1, 1]] = FRAC_PI_2.cos();
        // Translation
        t.matrix[[0, 3]] = 3.0;
        t.matrix[[1, 3]] = -7.0;
        t.matrix[[2, 3]] = 1.5;

        let inv = t.invert().unwrap();

        // Compute T · T⁻¹
        let mut product = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    product[[i, j]] += t.matrix[[i, k]] * inv.matrix[[k, j]];
                }
            }
        }

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (product[[i, j]] - expected).abs();
                assert!(
                    err < 1e-12,
                    "T·T⁻¹[[{i},{j}]] = {:.2e} error, expected ~0",
                    err
                );
            }
        }
    }
}
