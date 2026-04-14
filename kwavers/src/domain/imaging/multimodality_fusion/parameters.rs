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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_parameters_default() {
        let params = FusionParameters::default();
        assert_eq!(params.blend_weight, 0.5);
        assert!(params.auto_contrast);
    }

    #[test]
    fn test_fusion_method_display() {
        assert_eq!(FusionMethod::Overlay.to_string(), "Overlay");
        assert_eq!(FusionMethod::Checkerboard.to_string(), "Checkerboard");
    }
}
