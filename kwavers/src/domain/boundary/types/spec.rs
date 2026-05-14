//! `BoundarySpec` — complete boundary condition specification.

use super::boundary_type::BoundaryType;
use super::face_component::{BoundaryComponent, BoundaryFace};
use serde::{Deserialize, Serialize};

/// Boundary specification combining type, face, and component.
///
/// Complete specification of a boundary condition including:
/// - Which boundary face it applies to
/// - What type of condition (Dirichlet, Neumann, etc.)
/// - Which components (for vector fields)
/// - Time-dependent or spatial boundary data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundarySpec {
    /// Boundary face
    pub face: BoundaryFace,
    /// Boundary condition type
    pub boundary_type: BoundaryType,
    /// Component specification (for vector fields)
    pub component: BoundaryComponent,
    /// Time-dependent flag
    pub time_dependent: bool,
}

impl BoundarySpec {
    /// Create a new boundary specification.
    #[must_use]
    pub fn new(
        face: BoundaryFace,
        boundary_type: BoundaryType,
        component: BoundaryComponent,
    ) -> Self {
        Self {
            face,
            boundary_type,
            component,
            time_dependent: false,
        }
    }

    /// Create a time-dependent boundary specification.
    #[must_use]
    pub fn time_dependent(
        face: BoundaryFace,
        boundary_type: BoundaryType,
        component: BoundaryComponent,
    ) -> Self {
        Self {
            face,
            boundary_type,
            component,
            time_dependent: true,
        }
    }
}
