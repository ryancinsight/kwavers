/// Validation results for aberration correction.
#[derive(Debug)]
pub struct CorrectionValidation {
    /// Focal intensity (W/m²) via trilinear interpolation at target.
    pub focal_intensity: f64,
    /// Peak sidelobe level (dB below main lobe, negative = below main lobe).
    pub sidelobe_level_db: f64,
    /// Geometric-mean FWHM focal spot size: `(FWHM_x · FWHM_y · FWHM_z)^(1/3)` (m).
    pub focal_spot_size: f64,
}
