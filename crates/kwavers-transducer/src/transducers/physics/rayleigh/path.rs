//! Straight-ray propagation paths: homogeneous or ordered layers.

use super::input::invalid;
use kwavers_core::error::KwaversResult;
use kwavers_core::validation::validate_positive_finite;

/// One straight-ray propagation segment, ordered outward from the source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayleighLayer {
    wavenumber_rad_m: f64,
    attenuation_np_m: f64,
    thickness_m: Option<f64>,
}

/// Validated straight-ray acoustic propagation path.
///
/// Finite layers consume their configured thickness in order; the final
/// semi-infinite layer receives the remaining distance. The contract models
/// phase and attenuation only: interface reflection and refraction remain
/// outside the straight-ray Rayleigh approximation.
#[derive(Debug, Clone, PartialEq)]
pub struct RayleighPropagationPath {
    layers: Vec<RayleighLayer>,
}

impl RayleighLayer {
    /// Construct a propagation segment; `None` thickness denotes the final
    /// semi-infinite layer.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` for non-finite coefficients, a
    /// non-positive wavenumber, negative attenuation, or non-positive finite
    /// thickness.
    pub fn new(
        wavenumber_rad_m: f64,
        attenuation_np_m: f64,
        thickness_m: Option<f64>,
    ) -> KwaversResult<Self> {
        validate_positive_finite(
            "wavenumber_rad_m",
            wavenumber_rad_m,
            "must be finite and positive",
        )?;
        if !attenuation_np_m.is_finite() || attenuation_np_m < 0.0 {
            return Err(invalid(
                "attenuation_np_m",
                attenuation_np_m.to_string(),
                "finite and >= 0",
            ));
        }
        if let Some(thickness) = thickness_m {
            validate_positive_finite(
                "layer_thickness_m",
                thickness,
                "must be finite and positive",
            )?;
        }
        Ok(Self {
            wavenumber_rad_m,
            attenuation_np_m,
            thickness_m,
        })
    }
}

impl RayleighPropagationPath {
    /// Construct a homogeneous semi-infinite propagation path.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` unless the wavenumber is finite and
    /// positive and attenuation is finite and non-negative.
    pub fn homogeneous(wavenumber_rad_m: f64, attenuation_np_m: f64) -> KwaversResult<Self> {
        Self::layered(vec![RayleighLayer::new(
            wavenumber_rad_m,
            attenuation_np_m,
            None,
        )?])
    }

    /// Construct an ordered straight-ray layered propagation path.
    ///
    /// The final layer must be semi-infinite and every preceding layer must
    /// carry a finite thickness.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` for an empty or structurally invalid
    /// layer sequence.
    pub fn layered(layers: Vec<RayleighLayer>) -> KwaversResult<Self> {
        if layers.is_empty() {
            return Err(invalid("layers", "0".to_owned(), "at least one layer"));
        }
        if layers
            .last()
            .is_some_and(|layer| layer.thickness_m.is_some())
        {
            return Err(invalid(
                "layers",
                "finite final layer".to_owned(),
                "final layer is semi-infinite",
            ));
        }
        if layers[..layers.len() - 1]
            .iter()
            .any(|layer| layer.thickness_m.is_none())
        {
            return Err(invalid(
                "layers",
                "non-final semi-infinite layer".to_owned(),
                "only final layer is semi-infinite",
            ));
        }
        Ok(Self { layers })
    }

    /// Integrate phase and amplitude attenuation along one straight-ray path.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` when `range_m` is not finite or is
    /// negative.
    pub fn propagation_terms(&self, range_m: f64) -> KwaversResult<(f64, f64)> {
        if !range_m.is_finite() || range_m < 0.0 {
            return Err(invalid("range_m", range_m.to_string(), "finite and >= 0"));
        }
        let mut remaining = range_m;
        let mut phase = 0.0;
        let mut attenuation = 0.0;
        for layer in &self.layers {
            let segment = layer
                .thickness_m
                .map_or(remaining, |thickness| remaining.min(thickness));
            phase = layer.wavenumber_rad_m.mul_add(segment, phase);
            attenuation = layer.attenuation_np_m.mul_add(segment, attenuation);
            remaining -= segment;
            if remaining <= 0.0 {
                break;
            }
        }
        Ok((phase, attenuation))
    }

    pub(super) fn wavenumber_rad_m(&self) -> f64 {
        self.layers[0].wavenumber_rad_m
    }

    pub(super) fn attenuation_np_m(&self) -> f64 {
        self.layers[0].attenuation_np_m
    }
}
