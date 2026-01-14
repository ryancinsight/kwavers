//! Isosurface extraction using marching cubes

use crate::core::error::KwaversResult;
use ndarray::Array3;

mod marching_cubes_tables {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/_tmp_marching_tables_snippet.rs"
    ));
}

/// Isosurface extractor using marching cubes algorithm
#[derive(Debug)]
pub struct IsosurfaceExtractor {
    edge_table: &'static [i32; 256],
    tri_table: &'static [i8; 256 * 16],
}

impl IsosurfaceExtractor {
    /// Create a new isosurface extractor
    pub fn new(_config: &crate::visualization::VisualizationConfig) -> KwaversResult<Self> {
        Ok(Self {
            edge_table: &marching_cubes_tables::EDGE_TABLE,
            tri_table: &marching_cubes_tables::TRI_TABLE,
        })
    }

    /// Extract isosurface at given threshold
    pub fn extract(&self, field: &Array3<f64>, threshold: f64) -> KwaversResult<Vec<[f32; 3]>> {
        let mut vertices = Vec::new();
        let (nx, ny, nz) = field.dim();

        if nx < 2 || ny < 2 || nz < 2 {
            return Ok(vertices);
        }

        // Marching cubes algorithm
        for i in 0..nx - 1 {
            for j in 0..ny - 1 {
                for k in 0..nz - 1 {
                    // Sample cube corners
                    let cube = [
                        field[[i, j, k]],
                        field[[i + 1, j, k]],
                        field[[i + 1, j + 1, k]],
                        field[[i, j + 1, k]],
                        field[[i, j, k + 1]],
                        field[[i + 1, j, k + 1]],
                        field[[i + 1, j + 1, k + 1]],
                        field[[i, j + 1, k + 1]],
                    ];

                    // Determine cube configuration
                    let mut cube_index = 0;
                    for (idx, &val) in cube.iter().enumerate() {
                        if val > threshold {
                            cube_index |= 1 << idx;
                        }
                    }

                    // Skip if cube is entirely inside or outside
                    if cube_index == 0 || cube_index == 255 {
                        continue;
                    }

                    // Generate triangles for this cube
                    self.generate_triangles(
                        &mut vertices,
                        &cube,
                        cube_index,
                        threshold,
                        [i as f32, j as f32, k as f32],
                    );
                }
            }
        }

        Ok(vertices)
    }

    /// Generate triangles for a cube using marching cubes algorithm
    ///
    /// **Implementation**: Uses lookup tables for edge and triangle generation
    ///
    /// **Reference**: Lorensen & Cline (1987) "Marching Cubes: High Resolution 3D Surface"
    fn generate_triangles(
        &self,
        vertices: &mut Vec<[f32; 3]>,
        cube: &[f64; 8],
        cube_index: usize,
        threshold: f64,
        offset: [f32; 3],
    ) {
        // Skip if cube is entirely inside or outside
        if cube_index == 0 || cube_index == 255 {
            return;
        }

        let edge_flags = self.edge_table[cube_index];
        if edge_flags == 0 {
            return;
        }

        // Get triangle list for this configuration
        let tri_list_start = cube_index * 16;
        let tri_list = &self.tri_table[tri_list_start..(tri_list_start + 16)];

        // Edge vertices (12 edges on a cube)
        let mut edge_vertices = [[0.0_f32; 3]; 12];

        // Compute edge intersections using linear interpolation
        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0), // Bottom face edges
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4), // Top face edges
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7), // Vertical edges
        ];

        for (edge_idx, &(v1, v2)) in edges.iter().enumerate() {
            if (edge_flags & (1 << edge_idx)) == 0 {
                continue;
            }

            let val1 = cube[v1];
            let val2 = cube[v2];

            // Linear interpolation to find intersection point
            let t = if (val2 - val1).abs() > 1e-10 {
                ((threshold - val1) / (val2 - val1)) as f32
            } else {
                0.5
            };

            // Compute 3D position of intersection
            let pos1 = Self::vertex_offset(v1);
            let pos2 = Self::vertex_offset(v2);

            edge_vertices[edge_idx] = [
                offset[0] + pos1[0] + t * (pos2[0] - pos1[0]),
                offset[1] + pos1[1] + t * (pos2[1] - pos1[1]),
                offset[2] + pos1[2] + t * (pos2[2] - pos1[2]),
            ];
        }

        // Generate triangles from edge vertices
        let mut i = 0;
        while i < tri_list.len() && tri_list[i] != -1 {
            if i + 2 < tri_list.len() {
                vertices.push(edge_vertices[tri_list[i] as usize]);
                vertices.push(edge_vertices[tri_list[i + 1] as usize]);
                vertices.push(edge_vertices[tri_list[i + 2] as usize]);
                i += 3;
            } else {
                break;
            }
        }
    }

    /// Get vertex offset in unit cube (0-7 cube corners)
    fn vertex_offset(vertex: usize) -> [f32; 3] {
        match vertex {
            0 => [0.0, 0.0, 0.0],
            1 => [1.0, 0.0, 0.0],
            2 => [1.0, 1.0, 0.0],
            3 => [0.0, 1.0, 0.0],
            4 => [0.0, 0.0, 1.0],
            5 => [1.0, 0.0, 1.0],
            6 => [1.0, 1.0, 1.0],
            7 => [0.0, 1.0, 1.0],
            _ => [0.0, 0.0, 0.0],
        }
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of_val(self.edge_table)
            + std::mem::size_of_val(self.tri_table)
    }
}

#[cfg(test)]
mod tests {
    use super::IsosurfaceExtractor;
    use crate::visualization::VisualizationConfig;
    use ndarray::Array3;

    #[test]
    fn extracts_triangles_for_cube_index_48() {
        let mut field = Array3::zeros((2, 2, 2));
        field[[0, 0, 1]] = 1.0;
        field[[1, 0, 1]] = 1.0;

        let extractor =
            IsosurfaceExtractor::new(&VisualizationConfig::default()).expect("extractor creates");
        let vertices = extractor
            .extract(&field, 0.5)
            .expect("isosurface extraction succeeds");

        assert_eq!(vertices.len(), 6);
        for v in vertices {
            assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
            assert!((0.0..=1.0).contains(&v[0]));
            assert!((0.0..=1.0).contains(&v[1]));
            assert!((0.0..=1.0).contains(&v[2]));
        }
    }

    #[test]
    fn extracts_triangles_for_cube_index_1() {
        let mut field = Array3::zeros((2, 2, 2));
        field[[0, 0, 0]] = 1.0;

        let extractor =
            IsosurfaceExtractor::new(&VisualizationConfig::default()).expect("extractor creates");
        let vertices = extractor
            .extract(&field, 0.5)
            .expect("isosurface extraction succeeds");

        assert_eq!(vertices.len(), 3);
        for v in vertices {
            assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
            assert!((0.0..=1.0).contains(&v[0]));
            assert!((0.0..=1.0).contains(&v[1]));
            assert!((0.0..=1.0).contains(&v[2]));
        }
    }
}
