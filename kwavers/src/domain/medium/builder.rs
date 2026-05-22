//! Medium builder - Complex medium construction logic
//!
//! Follows Builder pattern for complex medium instantiation

use super::{DomainMediumParameters, InterfaceTypeParameters, LayerParameters, MediumType};
use crate::core::constants::SOUND_SPEED_WATER_SIM;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::medium::heterogeneous::HeterogeneousFactory;
use crate::domain::medium::{homogeneous::HomogeneousMedium, Medium};

/// Specialized medium builder following Builder pattern from GRASP
#[derive(Debug)]
pub struct MediumBuilder;

impl MediumBuilder {
    /// Build medium instance from validated configuration
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn build(config: &DomainMediumParameters, grid: &Grid) -> KwaversResult<Box<dyn Medium>> {
        match config.medium_type {
            MediumType::Homogeneous => Self::build_homogeneous(
                config.density,
                config.sound_speed.unwrap_or(SOUND_SPEED_WATER_SIM),
                config,
                grid,
            ),
            MediumType::Heterogeneous => Self::build_heterogeneous(config, grid),
            MediumType::Layered => Self::build_layered(&config.layers, grid),
            MediumType::Anisotropic => Self::build_anisotropic(config, grid),
            ref variant => Err(KwaversError::FeatureNotAvailable(format!(
                "MediumType::{variant:?} requires a domain-specific medium loader; \
                 scalar fallback is prohibited"
            ))),
        }
    }

    /// Build homogeneous medium
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn build_homogeneous(
        density: f64,
        sound_speed: f64,
        config: &DomainMediumParameters,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        // HomogeneousMedium::new(density, sound_speed, mu_a, mu_s_prime, grid)
        // Extract optical properties from map or defaults
        let mu_a = config.properties.get("mu_a").copied().unwrap_or(0.0);
        let mu_s_prime = config.properties.get("mu_s_prime").copied().unwrap_or(0.0);

        let mut medium = HomogeneousMedium::new(density, sound_speed, mu_a, mu_s_prime, grid);

        medium.set_acoustic_properties(
            config.absorption,
            config.absorption_power,
            config.nonlinearity,
        )?;

        Ok(Box::new(medium))
    }

    /// Build heterogeneous medium from configuration.
    ///
    /// # Boundary contract
    ///
    /// Scalar heterogeneous configurations are materialized directly from the
    /// validated scalar parameters. File-backed tissue maps and property maps
    /// require a real parser/loader boundary before construction; they must not
    /// degrade to scalar fields because that erases heterogeneity requested by
    /// the caller.
    /// # Errors
    /// - Returns [`KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn build_heterogeneous(
        config: &DomainMediumParameters,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        let (nx, ny, nz) = grid.dimensions();

        if let Some(file) = &config.tissue_file {
            return Err(KwaversError::FeatureNotAvailable(format!(
                "heterogeneous tissue_file '{file}' requires an explicit medium-volume loader; \
                 scalar fallback for {nx}x{ny}x{nz} grid is prohibited"
            )));
        }

        if !config.property_maps.is_empty() {
            let mut keys = config.property_maps.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            return Err(KwaversError::FeatureNotAvailable(format!(
                "heterogeneous property_maps {keys:?} require explicit property-volume loaders; \
                 scalar fallback is prohibited"
            )));
        }

        let c0 = config.sound_speed.unwrap_or(SOUND_SPEED_WATER_SIM);
        let rho0 = config.density;
        let absorption = config.absorption;
        let nonlinearity = config.nonlinearity;
        let reference_frequency = 1.0e6; // 1 MHz default

        let medium = HeterogeneousFactory::from_functions(
            grid,
            move |_x, _y, _z| c0,
            move |_x, _y, _z| rho0,
            Some(Box::new(move |_x, _y, _z| absorption)),
            None, // alpha_power: use default 1.0
            Some(Box::new(move |_x, _y, _z| nonlinearity)),
            reference_frequency,
        );

        log::debug!(
            "Built HeterogeneousMedium (uniform c0={:.1} m/s, rho0={:.1} kg/m³) for {}x{}x{} grid",
            c0,
            rho0,
            nx,
            ny,
            nz
        );

        Ok(Box::new(medium))
    }

    /// Build a heterogeneous layered medium with per-layer material properties.
    ///
    /// ## Theorem — step-function medium with interface blending
    ///
    /// For N layers with thicknesses `{h_i}` stacked along the x-axis, the
    /// depth-to-layer mapping is:
    /// ```text
    /// z_i = Σ_{j≤i} h_j        (cumulative boundary depth of layer i)
    /// idx(x) = min{ i : z_i > x }   (first boundary strictly above x)
    /// ```
    /// The property function `p(x,y,z)` returns the value of the current
    /// layer, with optional blending at the lower boundary:
    ///
    /// - **Sharp**: step function — `p(x) = p_idx(x)`.
    /// - **Smooth(σ)**: sigmoid blend of width σ [m] centred at `z_idx(x)`:
    ///   ```text
    ///   p(x) = p_cur·(1−t) + p_next·t,   t = ½(1 + tanh((x−z)/σ))
    ///   ```
    /// - **Gradient(d)**: linear blend over d [m] starting at `z_idx(x)`:
    ///   ```text
    ///   p(x) = p_cur·(1−t) + p_next·t,   t = ((x−z)/d).clamp(0,1)
    ///   ```
    ///
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if `layers` is empty or any
    ///   layer has non-positive thickness.
    fn build_layered(layers: &[LayerParameters], grid: &Grid) -> KwaversResult<Box<dyn Medium>> {
        if layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "layered medium requires at least one layer".to_owned(),
            ));
        }
        for (i, l) in layers.iter().enumerate() {
            if l.thickness <= 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "layer {i} thickness {:.4e} m is not positive",
                    l.thickness
                )));
            }
        }

        // Cumulative depth boundaries [m]: boundaries[i] = depth of lower face of layer i.
        let boundaries: Vec<f64> = layers
            .iter()
            .scan(0.0_f64, |acc, l| {
                *acc += l.thickness;
                Some(*acc)
            })
            .collect();

        // Evaluate one material property at depth x with interface blending.
        //
        // Two boundary contributions are considered:
        //  (A) LOWER boundary of current layer → blend towards next layer
        //      using ls[idx].interface_type at z = bs[idx].
        //  (B) UPPER boundary of current layer → blend back towards previous
        //      layer using ls[idx-1].interface_type at z = bs[idx-1].
        //
        // For Smooth(σ): t = ½(1 + tanh((x−z)/σ)), blending from prev to next.
        // For Gradient(d): t = ((x−z)/d).clamp(0,1), linear over d metres.
        //
        // Repeated per property to allow independent `move` captures.
        macro_rules! blended_prop {
            ($prop:ident) => {{
                let ls = layers.to_vec();
                let bs = boundaries.clone();
                move |x: f64, _y: f64, _z: f64| -> f64 {
                    let n = ls.len();
                    let x = x.max(0.0);
                    let idx = bs.partition_point(|&b| b <= x).min(n - 1);
                    let v = ls[idx].$prop;

                    // (A) blend current layer → next at ls[idx]'s lower boundary
                    let after_lower = if idx + 1 < n {
                        let z_lo = bs[idx];
                        let v_next = ls[idx + 1].$prop;
                        match ls[idx].interface_type {
                            InterfaceTypeParameters::Sharp => v,
                            InterfaceTypeParameters::Smooth(sigma) if sigma > 0.0 => {
                                let t = 0.5 * (1.0 + f64::tanh((x - z_lo) / sigma));
                                v.mul_add(1.0 - t, v_next * t)
                            }
                            InterfaceTypeParameters::Gradient(d) if d > 0.0 => {
                                let t = ((x - z_lo) / d).clamp(0.0, 1.0);
                                v.mul_add(1.0 - t, v_next * t)
                            }
                            _ => v,
                        }
                    } else {
                        v
                    };

                    // (B) blend previous layer → current at ls[idx-1]'s lower boundary
                    if idx > 0 {
                        let z_up = bs[idx - 1];
                        let v_prev = ls[idx - 1].$prop;
                        match ls[idx - 1].interface_type {
                            InterfaceTypeParameters::Sharp => after_lower,
                            InterfaceTypeParameters::Smooth(sigma) if sigma > 0.0 => {
                                // t → 0 as x → z_up from below (prev layer dominant),
                                // t → 1 as x ≫ z_up (current layer dominant).
                                let t = 0.5 * (1.0 + f64::tanh((x - z_up) / sigma));
                                v_prev.mul_add(1.0 - t, after_lower * t)
                            }
                            InterfaceTypeParameters::Gradient(d) if d > 0.0 => {
                                let t = ((x - z_up) / d).clamp(0.0, 1.0);
                                v_prev.mul_add(1.0 - t, after_lower * t)
                            }
                            _ => after_lower,
                        }
                    } else {
                        after_lower
                    }
                }
            }};
        }

        let medium = HeterogeneousFactory::from_functions(
            grid,
            blended_prop!(sound_speed),
            blended_prop!(density),
            Some(Box::new(blended_prop!(absorption))),
            None, // alpha_power: uniform 1.0 (LayerParameters has no power-law exponent field)
            None, // nonlinearity: LayerParameters has no nonlinearity field
            1.0e6,
        );

        log::debug!(
            "Built layered HeterogeneousMedium: {} layers, total depth {:.3} m",
            layers.len(),
            boundaries.last().copied().unwrap_or(0.0),
        );

        Ok(Box::new(medium))
    }

    /// Build anisotropic medium.
    ///
    /// Full anisotropy requires an explicit tensor field supplied via
    /// `tensor_file` or `property_maps`. Scalar fallback is prohibited because
    /// it silently erases the requested anisotropy.
    ///
    /// # Errors
    /// - Always returns [`KwaversError::FeatureNotAvailable`]: callers must
    ///   supply tensor data through a domain-specific medium loader.
    fn build_anisotropic(
        config: &DomainMediumParameters,
        _grid: &Grid,
    ) -> KwaversResult<Box<dyn Medium>> {
        let source = config
            .tensor_file
            .as_deref()
            .unwrap_or("<no tensor_file supplied>");
        Err(KwaversError::FeatureNotAvailable(format!(
            "anisotropic medium '{source}' requires an explicit tensor-field loader; \
             scalar-property fallback is prohibited because it silently \
             discards the requested directional heterogeneity"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::{DENSITY_BRAIN, SOUND_SPEED_TISSUE};
    use std::collections::HashMap;

    fn test_grid() -> Grid {
        Grid::new(3, 3, 3, 1.0e-3, 1.0e-3, 1.0e-3).unwrap()
    }

    #[test]
    fn heterogeneous_scalar_config_builds_input_sensitive_uniform_medium() {
        let grid = test_grid();
        let config = DomainMediumParameters {
            medium_type: MediumType::Heterogeneous,
            density: DENSITY_BRAIN,
            sound_speed: Some(SOUND_SPEED_TISSUE),
            absorption: 0.45,
            nonlinearity: 6.0,
            ..DomainMediumParameters::default()
        };

        let medium = MediumBuilder::build(&config, &grid).unwrap();

        assert_eq!(medium.sound_speed(0, 0, 0), SOUND_SPEED_TISSUE);
        assert_eq!(medium.sound_speed(2, 2, 2), SOUND_SPEED_TISSUE);
        assert_eq!(medium.density(1, 1, 1), DENSITY_BRAIN);
        assert_eq!(medium.absorption(2, 1, 0), 0.45);
        assert_eq!(medium.nonlinearity(0, 2, 1), 6.0);
    }

    #[test]
    fn heterogeneous_tissue_file_rejects_scalar_fallback() {
        let grid = test_grid();
        let config = DomainMediumParameters {
            medium_type: MediumType::Heterogeneous,
            tissue_file: Some("phantom.nii.gz".to_string()),
            ..DomainMediumParameters::default()
        };

        let error = MediumBuilder::build(&config, &grid).unwrap_err();

        assert!(matches!(error, KwaversError::FeatureNotAvailable(_)));
        assert!(format!("{error}").contains("scalar fallback"));
    }

    /// **Invariant**: Two-layer medium with Sharp interface returns each layer's
    /// sound speed at the correct depth without averaging.
    #[test]
    fn layered_sharp_two_layer_step_function_is_exact() {
        // 4-cell grid, dx = 1 mm → x ∈ {0.0, 1.0, 2.0, 3.0} mm
        let grid = Grid::new(4, 1, 1, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let config = DomainMediumParameters {
            medium_type: MediumType::Layered,
            layers: vec![
                LayerParameters {
                    thickness: 2.0e-3, // 0–2 mm: water-like
                    density: 1000.0,
                    sound_speed: 1500.0,
                    absorption: 0.002,
                    interface_type: InterfaceTypeParameters::Sharp,
                },
                LayerParameters {
                    thickness: 2.0e-3, // 2–4 mm: tissue-like
                    density: 1050.0,
                    sound_speed: 1540.0,
                    absorption: 0.5,
                    interface_type: InterfaceTypeParameters::Sharp,
                },
            ],
            ..DomainMediumParameters::default()
        };

        let medium = MediumBuilder::build(&config, &grid).unwrap();

        // x=0 (i=0) and x=1mm (i=1): layer 0 — water-like
        assert_eq!(medium.sound_speed(0, 0, 0), 1500.0, "x=0 must be layer-0 speed");
        assert_eq!(medium.sound_speed(1, 0, 0), 1500.0, "x=1mm must be layer-0 speed");
        // x=2mm (i=2) and x=3mm (i=3): layer 1 — tissue-like
        assert_eq!(medium.sound_speed(2, 0, 0), 1540.0, "x=2mm must be layer-1 speed");
        assert_eq!(medium.sound_speed(3, 0, 0), 1540.0, "x=3mm must be layer-1 speed");
        // Density follows the same step
        assert_eq!(medium.density(0, 0, 0), 1000.0);
        assert_eq!(medium.density(3, 0, 0), 1050.0);
    }

    /// **Invariant**: Sigmoid blend at the Sharp→Smooth boundary: at x = z_boundary
    /// the blended value must be the exact half-way point between both layers.
    #[test]
    fn layered_smooth_interface_midpoint_is_exact_average() {
        // 6-cell grid, dx = 1 mm → boundaries at 3 mm (between layers 0 and 1)
        let grid = Grid::new(6, 1, 1, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let config = DomainMediumParameters {
            medium_type: MediumType::Layered,
            layers: vec![
                LayerParameters {
                    thickness: 3.0e-3,
                    density: 1000.0,
                    sound_speed: 1500.0,
                    absorption: 0.1,
                    interface_type: InterfaceTypeParameters::Smooth(0.5e-3), // σ = 0.5 mm
                },
                LayerParameters {
                    thickness: 3.0e-3,
                    density: 1200.0,
                    sound_speed: 2800.0, // skull-like
                    absorption: 13.0,
                    interface_type: InterfaceTypeParameters::Sharp,
                },
            ],
            ..DomainMediumParameters::default()
        };

        let medium = MediumBuilder::build(&config, &grid).unwrap();

        // At x = 3 mm (the boundary), tanh(0/σ) = 0, so t = 0.5
        // blended = 1500 * 0.5 + 2800 * 0.5 = 2150
        let at_boundary = medium.sound_speed(3, 0, 0);
        assert!(
            (at_boundary - 2150.0).abs() < 1.0,
            "at boundary, smooth blend must be mid-point 2150 m/s; got {at_boundary:.1}"
        );
    }

    /// **Invariant**: Empty layer list is rejected.
    #[test]
    fn layered_empty_layers_rejected() {
        let grid = test_grid();
        let config = DomainMediumParameters {
            medium_type: MediumType::Layered,
            layers: vec![],
            ..DomainMediumParameters::default()
        };
        let err = MediumBuilder::build(&config, &grid).unwrap_err();
        assert!(matches!(err, KwaversError::InvalidInput(_)));
    }

    /// **Invariant**: Anisotropic always returns FeatureNotAvailable.
    #[test]
    fn anisotropic_always_rejects_without_tensor_field() {
        let grid = test_grid();
        let config = DomainMediumParameters {
            medium_type: MediumType::Anisotropic,
            ..DomainMediumParameters::default()
        };
        let err = MediumBuilder::build(&config, &grid).unwrap_err();
        assert!(matches!(err, KwaversError::FeatureNotAvailable(_)));
        assert!(format!("{err}").contains("tensor-field loader"));
    }

    #[test]
    fn heterogeneous_property_maps_reject_scalar_fallback() {
        let grid = test_grid();
        let config = DomainMediumParameters {
            medium_type: MediumType::Heterogeneous,
            property_maps: HashMap::from([
                ("density".to_string(), "rho.h5".to_string()),
                ("sound_speed".to_string(), "c.h5".to_string()),
            ]),
            ..DomainMediumParameters::default()
        };

        let error = MediumBuilder::build(&config, &grid).unwrap_err();

        assert!(matches!(error, KwaversError::FeatureNotAvailable(_)));
        let message = format!("{error}");
        assert!(message.contains("density"));
        assert!(message.contains("sound_speed"));
        assert!(message.contains("scalar fallback"));
    }
}
