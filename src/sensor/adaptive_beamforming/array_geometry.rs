// adaptive_beamforming/array_geometry.rs - Array geometry definitions

/// Element position in 3D space
#[derive(Debug, Clone, Copy]
pub struct ElementPosition {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Array geometry configuration
#[derive(Debug)]
pub struct ArrayGeometry {
    pub elements: Vec<ElementPosition>,
    pub reference_element: usize,
}

impl ArrayGeometry {
    pub fn linear(num_elements: usize, spacing: f64) -> Self {
        let elements = (0..num_elements)
            .map(|i| ElementPosition {
                x: i as f64 * spacing,
                y: 0.0,
                z: 0.0,
            })
            .collect();

        Self {
            elements,
            reference_element: 0,
        }
    }
}
