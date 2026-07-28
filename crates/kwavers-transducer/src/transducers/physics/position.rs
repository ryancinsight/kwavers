//! Cartesian 3-D position — typed spatial coordinate for transducer/therapy geometry.
//!
//! `CartesianPosition` wraps three SI-unit `Length<f64>` coordinates so that
//! focal-spot and ablation-target locations carry dimensional safety without
//! the overhead of a full branded-type hierarchy.
//!
//! # Example
//! ```
//! use kwavers_transducer::transducers::physics::CartesianPosition;
//!
//! let pos = CartesianPosition::from_base([0.0, 0.0, 0.08])
//!     .expect("valid position");
//! assert!((pos.z_m() - 0.08).abs() < 1e-12);
//! ```

use aequitas::systems::si::quantities::Length;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};

/// 3-D Cartesian position in SI units (metres).
///
/// Stores `(x, y, z)` as typed `Length<f64>` values.  All coordinates must be
/// finite; distances outside the range `±10 m` are rejected as implausible for
/// medical-ultrasound geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartesianPosition {
    x: Length<f64>,
    y: Length<f64>,
    z: Length<f64>,
}

impl CartesianPosition {
    /// Construct from three typed `Length<f64>` values.
    ///
    /// # Errors
    /// Returns [`KwaversError::Config`] if any coordinate is non-finite or
    /// outside the plausible range `[-10 m, 10 m]`.
    pub fn new(coords: [Length<f64>; 3]) -> KwaversResult<Self> {
        let [x, y, z] = coords;
        Self::validate_coord("x", x.into_base())?;
        Self::validate_coord("y", y.into_base())?;
        Self::validate_coord("z", z.into_base())?;
        Ok(Self { x, y, z })
    }

    /// Construct from raw SI base-unit values `[x_m, y_m, z_m]`.
    ///
    /// # Errors
    /// Returns [`KwaversError::Config`] if any value is non-finite or outside
    /// `[-10 m, 10 m]`.
    pub fn from_base(coords: [f64; 3]) -> KwaversResult<Self> {
        let [x_m, y_m, z_m] = coords;
        Self::validate_coord("x", x_m)?;
        Self::validate_coord("y", y_m)?;
        Self::validate_coord("z", z_m)?;
        Ok(Self {
            x: Length::from_base(x_m),
            y: Length::from_base(y_m),
            z: Length::from_base(z_m),
        })
    }

    /// Returns the `x` coordinate.
    #[must_use]
    #[inline]
    pub fn x(&self) -> Length<f64> {
        self.x
    }

    /// Returns the `y` coordinate.
    #[must_use]
    #[inline]
    pub fn y(&self) -> Length<f64> {
        self.y
    }

    /// Returns the `z` coordinate.
    #[must_use]
    #[inline]
    pub fn z(&self) -> Length<f64> {
        self.z
    }

    /// Returns the `x` coordinate in SI base units (metres).
    #[must_use]
    #[inline]
    pub fn x_m(&self) -> f64 {
        self.x.into_base()
    }

    /// Returns the `y` coordinate in SI base units (metres).
    #[must_use]
    #[inline]
    pub fn y_m(&self) -> f64 {
        self.y.into_base()
    }

    /// Returns the `z` coordinate in SI base units (metres).
    #[must_use]
    #[inline]
    pub fn z_m(&self) -> f64 {
        self.z.into_base()
    }

    /// Returns the coordinates as a `[f64; 3]` in SI base units.
    ///
    /// Follows the `aequitas` `into_base` convention.
    #[must_use]
    #[inline]
    pub fn into_base(self) -> [f64; 3] {
        [self.x_m(), self.y_m(), self.z_m()]
    }

    /// Returns the coordinates as a `[f64; 3]` in SI base units.
    #[must_use]
    #[inline]
    pub fn to_base_array(&self) -> [f64; 3] {
        [self.x_m(), self.y_m(), self.z_m()]
    }

    /// Returns the coordinates as `[Length<f64>; 3]`.
    #[must_use]
    #[inline]
    pub fn to_length_array(&self) -> [Length<f64>; 3] {
        [self.x, self.y, self.z]
    }

    /// Euclidean distance to another position (metres).
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> Length<f64> {
        let dx = self.x_m() - other.x_m();
        let dy = self.y_m() - other.y_m();
        let dz = self.z_m() - other.z_m();
        Length::from_base((dx * dx + dy * dy + dz * dz).sqrt())
    }

    fn validate_coord(name: &str, value: f64) -> KwaversResult<()> {
        if !value.is_finite() {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: format!("CartesianPosition::{name}"),
                value: value.to_string(),
                constraint: "coordinate must be finite".to_owned(),
            }));
        }
        const MAX_COORD_M: f64 = 10.0;
        if value.abs() > MAX_COORD_M {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: format!("CartesianPosition::{name}"),
                value: value.to_string(),
                constraint: format!(
                    "coordinate must be within ±{MAX_COORD_M} m for medical-ultrasound geometry"
                ),
            }));
        }
        Ok(())
    }
}
