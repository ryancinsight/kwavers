//! Array geometry definitions for PAM

/// Array geometry types for different sensor configurations
#[derive(Debug, Clone)]
pub enum ArrayGeometry {
    /// Linear array (1D)
    Linear {
        elements: usize,
        pitch: f64, // Element spacing in meters
        center: [f64; 3],
        orientation: [f64; 3], // Direction vector
    },
    /// Planar array (2D)
    Planar {
        elements_x: usize,
        elements_y: usize,
        pitch_x: f64,
        pitch_y: f64,
        center: [f64; 3],
        normal: [f64; 3], // Normal to plane
    },
    /// Circular/Ring array
    Circular {
        elements: usize,
        radius: f64,
        center: [f64; 3],
        normal: [f64; 3],
    },
    /// Hemispherical bowl array
    Hemispherical {
        elements_theta: usize, // Elevation elements
        elements_phi: usize,   // Azimuthal elements
        radius: f64,
        center: [f64; 3],
    },
    /// Arbitrary 3D positions
    Arbitrary { positions: Vec<[f64; 3]> },
}

/// Single array element
#[derive(Debug, Clone)]
pub struct ArrayElement {
    pub position: [f64; 3],
    pub sensitivity: f64,
    pub directivity: Option<DirectivityPattern>,
}

/// Directivity pattern for array elements
#[derive(Debug, Clone)]
pub struct DirectivityPattern {
    pub beam_width: f64,
    pub orientation: [f64; 3],
}

impl ArrayGeometry {
    /// Get element positions for the array geometry
    #[must_use]
    pub fn element_positions(&self) -> Vec<[f64; 3]> {
        match self {
            ArrayGeometry::Linear {
                elements,
                pitch,
                center,
                orientation,
            } => {
                let mut positions = Vec::with_capacity(*elements);
                let norm = crate::domain::math::geometry::normalize3(*orientation);

                for i in 0..*elements {
                    let offset = (i as f64 - (*elements as f64 - 1.0) / 2.0) * pitch;
                    positions.push([
                        center[0] + offset * norm[0],
                        center[1] + offset * norm[1],
                        center[2] + offset * norm[2],
                    ]);
                }
                positions
            }

            ArrayGeometry::Planar {
                elements_x,
                elements_y,
                pitch_x,
                pitch_y,
                center,
                normal,
            } => {
                let mut positions = Vec::with_capacity(elements_x * elements_y);
                let (u, v) = crate::domain::math::geometry::orthogonal_basis_from_normal3(*normal);

                for i in 0..*elements_x {
                    for j in 0..*elements_y {
                        let offset_x = (i as f64 - (*elements_x as f64 - 1.0) / 2.0) * pitch_x;
                        let offset_y = (j as f64 - (*elements_y as f64 - 1.0) / 2.0) * pitch_y;

                        positions.push([
                            center[0] + offset_x * u[0] + offset_y * v[0],
                            center[1] + offset_x * u[1] + offset_y * v[1],
                            center[2] + offset_x * u[2] + offset_y * v[2],
                        ]);
                    }
                }
                positions
            }

            ArrayGeometry::Circular {
                elements,
                radius,
                center,
                normal,
            } => {
                let mut positions = Vec::with_capacity(*elements);
                let (u, v) = crate::domain::math::geometry::orthogonal_basis_from_normal3(*normal);

                for i in 0..*elements {
                    let angle = 2.0 * std::f64::consts::PI * i as f64 / *elements as f64;
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    positions.push([
                        center[0] + radius * (cos_a * u[0] + sin_a * v[0]),
                        center[1] + radius * (cos_a * u[1] + sin_a * v[1]),
                        center[2] + radius * (cos_a * u[2] + sin_a * v[2]),
                    ]);
                }
                positions
            }

            ArrayGeometry::Hemispherical {
                elements_theta,
                elements_phi,
                radius,
                center,
            } => {
                let mut positions = Vec::with_capacity(elements_theta * elements_phi);

                for i in 0..*elements_theta {
                    let theta =
                        std::f64::consts::PI * 0.5 * i as f64 / (*elements_theta as f64 - 1.0);
                    for j in 0..*elements_phi {
                        let phi = 2.0 * std::f64::consts::PI * j as f64 / *elements_phi as f64;

                        positions.push([
                            center[0] + radius * theta.sin() * phi.cos(),
                            center[1] + radius * theta.sin() * phi.sin(),
                            center[2] + radius * theta.cos(),
                        ]);
                    }
                }
                positions
            }

            ArrayGeometry::Arbitrary { positions } => positions.clone(),
        }
    }

    /// Get number of elements
    #[must_use]
    pub fn num_elements(&self) -> usize {
        match self {
            ArrayGeometry::Linear { elements, .. } => *elements,
            ArrayGeometry::Planar {
                elements_x,
                elements_y,
                ..
            } => elements_x * elements_y,
            ArrayGeometry::Circular { elements, .. } => *elements,
            ArrayGeometry::Hemispherical {
                elements_theta,
                elements_phi,
                ..
            } => elements_theta * elements_phi,
            ArrayGeometry::Arbitrary { positions } => positions.len(),
        }
    }
}
