/// Spatial discretization order for numerical schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticSpatialOrder {
    /// Second-order accurate spatial discretization
    Second,
    /// Fourth-order accurate spatial discretization
    Fourth,
    /// Sixth-order accurate spatial discretization
    Sixth,
}

impl AcousticSpatialOrder {
    /// Get the CFL stability limit for this spatial order
    ///
    /// For 3D finite difference schemes with central differences:
    /// - 2nd order: CFL ≤ 1/√(3) ≈ 0.577
    /// - 4th order: CFL ≤ 1/√(15) ≈ 0.258
    /// - 6th order: CFL ≤ 1/√(27) ≈ 0.192
    ///
    /// Reference: Gustafsson et al. (1995) "Time compact difference schemes"
    #[must_use]
    pub fn cfl_limit(&self) -> f64 {
        match self {
            Self::Second => 1.0 / (3.0_f64).sqrt(),  // 1/√3 ≈ 0.577
            Self::Fourth => 1.0 / (15.0_f64).sqrt(), // 1/√15 ≈ 0.258
            Self::Sixth => 1.0 / (27.0_f64).sqrt(),  // 1/√27 ≈ 0.192
        }
    }

    /// Get the minimum number of grid points required for this spatial order
    #[must_use]
    pub fn minimum_grid_points(&self) -> usize {
        match self {
            Self::Second => 3,
            Self::Fourth => 5,
            Self::Sixth => 7,
        }
    }

    /// Convert from usize, returning an error for invalid orders
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` for any value other than 2, 4, or 6.
    pub fn from_usize(order: usize) -> Result<Self, kwavers_core::error::KwaversError> {
        match order {
            2 => Ok(Self::Second),
            4 => Ok(Self::Fourth),
            6 => Ok(Self::Sixth),
            _ => Err(kwavers_core::error::ConfigError::InvalidValue {
                parameter: "spatial_order".to_owned(),
                value: order.to_string(),
                constraint: "must be 2, 4, or 6".to_owned(),
            }
            .into()),
        }
    }
}
