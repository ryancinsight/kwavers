//! Boundary enumeration types and direction flags.

/// Field type enumeration for multi-physics support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryFieldType {
    /// Acoustic pressure field (Pa).
    Pressure,
    /// Velocity field components (m/s).
    Velocity,
    /// Particle displacement (m).
    Displacement,
    /// Stress tensor components (Pa).
    Stress,
    /// Electric field (V/m).
    Electric,
    /// Magnetic field (A/m).
    Magnetic,
    /// Temperature field (K).
    Temperature,
    /// Optical fluence (W/m²).
    Fluence,
}

/// Domain in which a boundary condition is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryDomain {
    /// Spatial domain (real-space).
    Spatial,
    /// Frequency domain (k-space, spectral).
    Frequency,
    /// Time domain (discrete time steps).
    Temporal,
}

/// Direction flags for selective boundary application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryDirections {
    pub x_min: bool,
    pub x_max: bool,
    pub y_min: bool,
    pub y_max: bool,
    pub z_min: bool,
    pub z_max: bool,
}

impl Default for BoundaryDirections {
    fn default() -> Self {
        Self::all()
    }
}

impl BoundaryDirections {
    /// Enable all six faces.
    #[must_use]
    pub const fn all() -> Self {
        Self {
            x_min: true,
            x_max: true,
            y_min: true,
            y_max: true,
            z_min: true,
            z_max: true,
        }
    }

    /// Disable all faces.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            x_min: false,
            x_max: false,
            y_min: false,
            y_max: false,
            z_min: false,
            z_max: false,
        }
    }

    /// Enable only XY-plane faces (for 2D simulations).
    #[must_use]
    pub const fn xy_plane() -> Self {
        Self {
            x_min: true,
            x_max: true,
            y_min: true,
            y_max: true,
            z_min: false,
            z_max: false,
        }
    }
}
