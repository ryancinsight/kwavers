//! Arc source implementation for 2D focused transducers
//!
//! Provides arc-shaped transducer geometry for 2D simulations.

use aequitas::systems::si::quantities::{Angle, Frequency, Length, Pressure, Time};
use aequitas::systems::si::units::{Hertz, Meter, Pascal, Radian, Second};
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use kwavers_core::{constants::SOUND_SPEED_WATER, error::KwaversResult};
use kwavers_grid::Grid;
use leto::{Array2, Array3};
use moirai_parallel::{enumerate_mut_with, Adaptive};
use std::f64::consts::PI;

use super::validation::{
    field_validation_error, validate_finite_field, validate_finite_vector,
    validate_positive_finite_field,
};
use kwavers_core::constants::numerical::TWO_PI;

/// Configuration for an arc source (2D focused transducer)
#[derive(Debug, Clone)]
pub struct ArcConfig {
    /// Radius of curvature (m)
    pub radius: Length<f64>,

    /// Arc angle (radians)
    pub arc_angle: Angle<f64>,

    /// Center position [x, y] (m)
    pub center: [Length<f64>; 2],

    /// Orientation angle (radians)
    pub orientation: Angle<f64>,

    /// Operating frequency (Hz)
    pub frequency: Frequency<f64>,

    /// Source amplitude (Pa)
    pub amplitude: Pressure<f64>,

    /// Element spacing (m)
    pub element_spacing: Option<Length<f64>>,
}

impl Default for ArcConfig {
    fn default() -> Self {
        Self {
            radius: Length::from_unit::<Meter>(0.05),
            arc_angle: Angle::from_unit::<Radian>(PI / 3.0),
            center: [Length::from_unit::<Meter>(0.0); 2],
            orientation: Angle::from_unit::<Radian>(0.0),
            frequency: Frequency::from_unit::<Hertz>(MHZ_TO_HZ),
            amplitude: Pressure::from_unit::<Pascal>(MPA_TO_PA),
            element_spacing: None,
        }
    }
}

/// Arc source for 2D simulations (makeArc equivalent)
#[derive(Debug)]
pub struct ArcSource {
    config: ArcConfig,
    /// Discretized element positions
    element_positions: Vec<[Length<f64>; 2]>,
    /// Element weights
    element_weights: Vec<f64>,
}

impl ArcSource {
    /// Create a new arc source
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: ArcConfig) -> KwaversResult<Self> {
        validate_arc_config(&config)?;

        // Calculate element spacing if not provided
        let element_spacing = config.element_spacing.unwrap_or_else(|| {
            let speed_of_sound = SOUND_SPEED_WATER;
            let wavelength = speed_of_sound / config.frequency.in_unit::<Hertz>();
            Length::from_unit::<Meter>(wavelength / 4.0)
        });
        validate_positive_finite_field("element_spacing", element_spacing.in_unit::<Meter>())?;

        // Number of elements
        let radius_m = config.radius.in_unit::<Meter>();
        let arc_angle_rad = config.arc_angle.in_unit::<Radian>();
        let arc_length = radius_m * arc_angle_rad;
        let n_elements =
            element_count_from_arc_length(arc_length, element_spacing.in_unit::<Meter>())?;

        // Generate element positions
        let mut positions = Vec::with_capacity(n_elements);
        let mut weights = Vec::with_capacity(n_elements);

        for i in 0..n_elements {
            // Angle for this element
            let theta = -arc_angle_rad / 2.0 + (i as f64 + 0.5) * arc_angle_rad / n_elements as f64;
            let rotated_theta = theta + config.orientation.in_unit::<Radian>();

            // Position
            let x = radius_m.mul_add(rotated_theta.cos(), config.center[0].in_unit::<Meter>());
            let y = radius_m.mul_add(rotated_theta.sin(), config.center[1].in_unit::<Meter>());

            positions.push([Length::from_unit::<Meter>(x), Length::from_unit::<Meter>(y)]);
            weights.push(1.0 / n_elements as f64);
        }

        Ok(Self {
            config,
            element_positions: positions,
            element_weights: weights,
        })
    }

    /// Generate 2D source distribution
    #[must_use]
    pub fn generate_source_2d(
        &self,
        nx: usize,
        ny: usize,
        spacing: Length<f64>,
        time: Time<f64>,
    ) -> Array2<f64> {
        self.generate_source_2d_on_grid(
            nx,
            ny,
            [0.0, 0.0],
            [spacing.in_unit::<Meter>(); 2],
            time.in_unit::<Second>(),
        )
    }

    /// Generate a 2-D source distribution on an anisotropic Cartesian grid.
    ///
    /// ## Theorem
    ///
    /// For an arc element at `x_i`, the retained 2-D line-source contribution is
    /// proportional to `sin(omega (t - |x_focus - x_i| / c)) / sqrt(|x - x_i|)`.
    /// Grid coordinates are therefore part of the physical phase/amplitude
    /// contract and must use axis-specific spacing plus origin offsets.
    #[must_use]
    fn generate_source_2d_on_grid(
        &self,
        nx: usize,
        ny: usize,
        origin: [f64; 2],
        spacing: [f64; 2],
        time: f64,
    ) -> Array2<f64> {
        let mut source = Array2::zeros([nx, ny]);
        let omega = TWO_PI * self.config.frequency.in_unit::<Hertz>();

        // Focus is at the center of curvature
        let focus = self.config.center.map(|value| value.in_unit::<Meter>());

        // Calculate delays for focusing
        let speed_of_sound = SOUND_SPEED_WATER;
        let delays: Vec<f64> = self
            .element_positions
            .iter()
            .map(|&pos| {
                let pos = pos.map(|value| value.in_unit::<Meter>());
                let distance = (focus[0] - pos[0]).hypot(focus[1] - pos[1]);
                distance / speed_of_sound
            })
            .collect();

        let source_data = source
            .as_slice_mut()
            .expect("invariant: freshly allocated Array2 is contiguous");

        // Generate source field.
        enumerate_mut_with::<Adaptive, _, _>(source_data, |idx, val| {
            let ix = idx / ny;
            let iy = idx % ny;
            let x = (ix as f64).mul_add(spacing[0], origin[0]);
            let y = (iy as f64).mul_add(spacing[1], origin[1]);

            let mut pressure = 0.0;
            for (i, &pos) in self.element_positions.iter().enumerate() {
                let pos = pos.map(|value| value.in_unit::<Meter>());
                let r = (x - pos[0]).hypot(y - pos[1]);

                if r > 0.0 {
                    // Phase with focusing delay
                    let phase = omega * (time - delays[i]);

                    // 2D Green's function (Hankel function calculation)
                    let element_pressure = self.config.amplitude.in_unit::<Pascal>()
                        * self.element_weights[i]
                        * phase.sin()
                        / r.sqrt();

                    pressure += element_pressure;
                }
            }

            *val = pressure;
        });

        source
    }

    /// Extend 2D source to 3D (uniform in z-direction)
    pub fn generate_source_3d(&self, grid: &Grid, time: Time<f64>) -> Array3<f64> {
        let source_2d = self.generate_source_2d_on_grid(
            grid.nx,
            grid.ny,
            [grid.origin[0], grid.origin[1]],
            [grid.dx, grid.dy],
            time.in_unit::<Second>(),
        );
        let mut source_3d = Array3::zeros([grid.nx, grid.ny, grid.nz]);

        // Copy 2D pattern to all z-slices
        for iz in 0..grid.nz {
            for ix in 0..grid.nx {
                for iy in 0..grid.ny {
                    source_3d[[ix, iy, iz]] = source_2d[[ix, iy]];
                }
            }
        }

        source_3d
    }
}

fn validate_arc_config(config: &ArcConfig) -> KwaversResult<()> {
    validate_positive_finite_field("radius", config.radius.in_unit::<Meter>())?;
    validate_positive_finite_field("frequency", config.frequency.in_unit::<Hertz>())?;
    validate_finite_field("amplitude", config.amplitude.in_unit::<Pascal>())?;
    validate_finite_field("orientation", config.orientation.in_unit::<Radian>())?;
    let center = config.center.map(|value| value.in_unit::<Meter>());
    validate_finite_vector("center", center)?;
    let arc_angle = config.arc_angle.in_unit::<Radian>();
    if !(arc_angle.is_finite() && arc_angle > 0.0 && arc_angle <= TWO_PI) {
        return Err(field_validation_error(
            "arc_angle",
            arc_angle.to_string(),
            "must satisfy 0 < arc_angle <= 2*pi",
        ));
    }
    if let Some(element_spacing) = config.element_spacing {
        validate_positive_finite_field("element_spacing", element_spacing.in_unit::<Meter>())?;
    }
    Ok(())
}

fn element_count_from_arc_length(
    arc_length_m: f64,
    element_spacing_m: f64,
) -> KwaversResult<usize> {
    let count = (arc_length_m / element_spacing_m).ceil();
    if !count.is_finite() || count < 1.0 || count > usize::MAX as f64 {
        return Err(field_validation_error(
            "element_spacing",
            element_spacing_m.to_string(),
            "must produce a finite representable arc element count",
        ));
    }
    Ok(count as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::quantities::{Angle, Frequency, Length, Pressure, Time};
    use aequitas::systems::si::units::{Hertz, Meter, Pascal, Radian};
    use kwavers_core::error::KwaversError;

    #[test]
    fn arc_rejects_invalid_source_domains() {
        let zero_radius = ArcConfig {
            radius: Length::from_unit::<Meter>(0.0),
            ..Default::default()
        };
        assert_validation_error(zero_radius);

        let zero_angle = ArcConfig {
            arc_angle: Angle::from_unit::<Radian>(0.0),
            ..Default::default()
        };
        assert_validation_error(zero_angle);

        let excessive_angle = ArcConfig {
            arc_angle: Angle::from_unit::<Radian>(2.0 * PI + 1.0e-12),
            ..Default::default()
        };
        assert_validation_error(excessive_angle);

        let zero_frequency = ArcConfig {
            frequency: Frequency::from_unit::<Hertz>(0.0),
            ..Default::default()
        };
        assert_validation_error(zero_frequency);

        let zero_spacing = ArcConfig {
            element_spacing: Some(Length::from_unit::<Meter>(0.0)),
            ..Default::default()
        };
        assert_validation_error(zero_spacing);

        let nonfinite_center = ArcConfig {
            center: [
                Length::from_unit::<Meter>(f64::NAN),
                Length::from_unit::<Meter>(0.0),
            ],
            ..Default::default()
        };
        assert_validation_error(nonfinite_center);
    }

    #[test]
    fn arc_source_3d_uses_grid_origin_and_axis_spacing() {
        let config = ArcConfig {
            radius: Length::from_unit::<Meter>(0.04),
            arc_angle: Angle::from_unit::<Radian>(0.5),
            center: [Length::from_unit::<Meter>(0.0); 2],
            orientation: Angle::from_unit::<Radian>(0.25),
            frequency: Frequency::from_unit::<Hertz>(1.2 * MHZ_TO_HZ),
            amplitude: Pressure::from_unit::<Pascal>(2.0e5),
            element_spacing: Some(Length::from_unit::<Meter>(0.1)),
        };
        let arc = ArcSource::new(config.clone()).unwrap();
        assert_eq!(arc.element_positions.len(), 1);

        let mut grid = Grid::new(3, 4, 2, 0.001, 0.002, 0.003).unwrap();
        grid.origin = [0.011, -0.017, 0.023];
        let time = Time::from_base(0.41e-6);
        let source = arc.generate_source_3d(&grid, time);

        let element_position = arc.element_positions[0].map(|value| value.in_unit::<Meter>());
        let element_weight = arc.element_weights[0];
        let delay = config.radius.in_unit::<Meter>() / SOUND_SPEED_WATER;
        let phase = 2.0 * PI * config.frequency.in_unit::<Hertz>() * (time.into_base() - delay);

        for ix in 0..grid.nx {
            for iy in 0..grid.ny {
                let point = [
                    (ix as f64).mul_add(grid.dx, grid.origin[0]),
                    (iy as f64).mul_add(grid.dy, grid.origin[1]),
                ];
                let distance =
                    (point[0] - element_position[0]).hypot(point[1] - element_position[1]);
                let expected = if distance > 0.0 {
                    config.amplitude.in_unit::<Pascal>() * element_weight * phase.sin()
                        / distance.sqrt()
                } else {
                    0.0
                };

                for iz in 0..grid.nz {
                    assert!(
                        (source[[ix, iy, iz]] - expected).abs() < expected.abs().max(1.0) * 1.0e-12,
                        "source[{ix},{iy},{iz}] = {}, expected {expected}",
                        source[[ix, iy, iz]]
                    );
                }
            }
        }
    }

    fn assert_validation_error(config: ArcConfig) {
        let error = ArcSource::new(config).unwrap_err();
        assert!(matches!(error, KwaversError::Validation(_)));
    }
}
