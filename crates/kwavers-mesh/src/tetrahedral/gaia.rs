//! Gaia mesh conversion for tetrahedral finite-element domains.
//!
//! # Contract
//!
//! `gaia::IndexedMesh<f64>` is the authoritative meshing artifact. Kwavers
//! imports only tetrahedral volume cells into [`TetrahedralMesh`], preserving
//! vertex coordinates, cell connectivity, boundary-condition labels, adjacency,
//! total volume, and element quality metrics.
//!
//! # Theorem
//!
//! For every Gaia tetrahedral cell with vertex set `{a,b,c,d}`, kwavers stores
//! one [`super::types::Tetrahedron`] over the same four coordinates. Since both
//! Gaia and kwavers use the scalar triple product volume
//! `|det(b-a,c-a,d-a)|/6`, conversion preserves per-cell and total volume.

use super::mesh::TetrahedralMesh;
use super::types::MeshBoundaryType;
use kwavers_core::error::{KwaversError, KwaversResult};
use gaia::domain::core::index::{FaceId, VertexId};
use gaia::domain::topology::{Cell, ElementType};
use gaia::IndexedMesh;
use std::collections::BTreeSet;

impl TetrahedralMesh {
    /// Convert a Gaia tetrahedral volume mesh into kwavers' FEM mesh.
    ///
    /// Boundary-condition labels are recognized by exact semantic names:
    /// `dirichlet`, `neumann`, `robin`, and `radiation`/`sommerfeld`.
    /// Geometric labels such as `inlet`, `outlet`, and `wall` remain
    /// [`MeshBoundaryType::Interior`] because they do not define a mathematical
    /// boundary condition by themselves.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn from_gaia_indexed_mesh(mesh: &IndexedMesh<f64>) -> KwaversResult<Self> {
        validate_gaia_mesh(mesh)?;

        let mut boundary_types = vec![MeshBoundaryType::Interior; mesh.vertex_count()];
        apply_gaia_boundary_labels(mesh, &mut boundary_types)?;

        let mut converted = Self::new();
        converted.nodes.reserve(mesh.vertex_count());
        converted.elements.reserve(mesh.cell_count());

        for (node_idx, &boundary_type) in boundary_types.iter().enumerate() {
            let position = mesh.vertices.position(VertexId::from_usize(node_idx));
            if !position.x.is_finite() || !position.y.is_finite() || !position.z.is_finite() {
                return Err(KwaversError::InvalidInput(format!(
                    "Gaia vertex {node_idx} has non-finite coordinates"
                )));
            }
            converted.add_node([position.x, position.y, position.z], boundary_type);
        }

        for (cell_idx, cell) in mesh.cells().iter().enumerate() {
            let nodes = gaia_cell_vertices(mesh, cell, cell_idx)?;
            converted.add_element(nodes, 0)?;
        }

        Ok(converted)
    }
}

fn validate_gaia_mesh(mesh: &IndexedMesh<f64>) -> KwaversResult<()> {
    if mesh.vertex_count() == 0 {
        return Err(KwaversError::InvalidInput(
            "Gaia mesh conversion requires at least one vertex".to_owned(),
        ));
    }
    if mesh.cell_count() == 0 {
        return Err(KwaversError::InvalidInput(
            "Gaia mesh conversion requires tetrahedral volume cells".to_owned(),
        ));
    }
    Ok(())
}

fn apply_gaia_boundary_labels(
    mesh: &IndexedMesh<f64>,
    boundary_types: &mut [MeshBoundaryType],
) -> KwaversResult<()> {
    for (&face_id, label) in &mesh.boundary_labels {
        let Some(boundary_type) = boundary_type_from_label(label) else {
            continue;
        };
        validate_face_index(mesh, face_id.as_usize())?;
        let face = mesh.faces.get(face_id);
        for vertex_id in face.vertices {
            let node_idx = vertex_id.as_usize();
            merge_boundary_type(boundary_types, node_idx, boundary_type)?;
        }
    }
    Ok(())
}

fn boundary_type_from_label(label: &str) -> Option<MeshBoundaryType> {
    match label.trim().to_ascii_lowercase().as_str() {
        "dirichlet" => Some(MeshBoundaryType::Dirichlet),
        "neumann" => Some(MeshBoundaryType::Neumann),
        "robin" => Some(MeshBoundaryType::Robin),
        "radiation" | "sommerfeld" => Some(MeshBoundaryType::Radiation),
        _ => None,
    }
}

fn merge_boundary_type(
    boundary_types: &mut [MeshBoundaryType],
    node_idx: usize,
    candidate: MeshBoundaryType,
) -> KwaversResult<()> {
    let Some(current) = boundary_types.get_mut(node_idx) else {
        return Err(KwaversError::InvalidInput(format!(
            "Gaia boundary references vertex {node_idx}, but mesh has {} vertices",
            boundary_types.len()
        )));
    };

    match (*current, candidate) {
        (MeshBoundaryType::Interior, next) => {
            *current = next;
            Ok(())
        }
        (existing, next) if existing == next => Ok(()),
        (existing, next) => Err(KwaversError::InvalidInput(format!(
            "Gaia boundary labels assign conflicting boundary types {existing:?} and {next:?} to vertex {node_idx}"
        ))),
    }
}

fn gaia_cell_vertices(
    mesh: &IndexedMesh<f64>,
    cell: &Cell,
    cell_idx: usize,
) -> KwaversResult<[usize; 4]> {
    if cell.element_type != ElementType::Tetrahedron {
        return Err(KwaversError::InvalidInput(format!(
            "Gaia cell {cell_idx} is {:?}; only tetrahedral cells are supported",
            cell.element_type
        )));
    }

    if !cell.vertex_ids.is_empty() {
        return vertex_ids_to_array(mesh, &cell.vertex_ids, cell_idx);
    }

    let mut vertices = BTreeSet::new();
    for &face_idx in &cell.faces {
        validate_face_index(mesh, face_idx)?;
        let face = mesh.faces.get(FaceId::from_usize(face_idx));
        for vertex_id in face.vertices {
            vertices.insert(vertex_id.as_usize());
        }
    }
    vertex_ids_to_array(mesh, &vertices.into_iter().collect::<Vec<_>>(), cell_idx)
}

fn vertex_ids_to_array(
    mesh: &IndexedMesh<f64>,
    vertex_ids: &[usize],
    cell_idx: usize,
) -> KwaversResult<[usize; 4]> {
    if vertex_ids.len() != 4 {
        return Err(KwaversError::InvalidInput(format!(
            "Gaia tetrahedral cell {cell_idx} references {} unique vertices; expected 4",
            vertex_ids.len()
        )));
    }

    let mut nodes = [0; 4];
    for (target, &vertex_idx) in nodes.iter_mut().zip(vertex_ids) {
        if vertex_idx >= mesh.vertex_count() {
            return Err(KwaversError::InvalidInput(format!(
                "Gaia cell {cell_idx} references vertex {vertex_idx}, but mesh has {} vertices",
                mesh.vertex_count()
            )));
        }
        *target = vertex_idx;
    }
    Ok(nodes)
}

fn validate_face_index(mesh: &IndexedMesh<f64>, face_idx: usize) -> KwaversResult<()> {
    if face_idx >= mesh.face_count() {
        return Err(KwaversError::InvalidInput(format!(
            "Gaia cell references face {face_idx}, but mesh has {} faces",
            mesh.face_count()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gaia::domain::core::scalar::Point3r;
    use gaia::domain::grid::StructuredGridBuilder;
    use gaia::domain::topology::Cell;

    #[test]
    fn converts_gaia_structured_volume_mesh_and_preserves_unit_cube_volume() {
        let gaia_mesh = StructuredGridBuilder::new(1, 1, 1).build().unwrap();

        let kwavers_mesh = TetrahedralMesh::from_gaia_indexed_mesh(&gaia_mesh).unwrap();
        let stats = kwavers_mesh.statistics();

        assert_eq!(stats.num_nodes, 8);
        assert_eq!(stats.num_elements, 5);
        assert!((stats.total_volume - 1.0).abs() < 1e-12);
        assert!(stats.minimum_quality > 0.0);
        assert_eq!(stats.num_boundary_faces, 12);
    }

    #[test]
    fn maps_explicit_gaia_boundary_condition_labels_to_nodes() {
        let mut gaia_mesh = IndexedMesh::<f64>::new();
        let v0 = gaia_mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
        let v1 = gaia_mesh.add_vertex_pos(Point3r::new(1.0, 0.0, 0.0));
        let v2 = gaia_mesh.add_vertex_pos(Point3r::new(0.0, 1.0, 0.0));
        let v3 = gaia_mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 1.0));

        let f0 = gaia_mesh.add_face(v0, v1, v2);
        let f1 = gaia_mesh.add_face(v0, v1, v3);
        let f2 = gaia_mesh.add_face(v0, v2, v3);
        let f3 = gaia_mesh.add_face(v1, v2, v3);
        let mut cell =
            Cell::tetrahedron(f0.as_usize(), f1.as_usize(), f2.as_usize(), f3.as_usize());
        cell.vertex_ids = vec![0, 1, 2, 3];
        gaia_mesh.add_cell(cell);
        gaia_mesh.mark_boundary(f0, "dirichlet");

        let kwavers_mesh = TetrahedralMesh::from_gaia_indexed_mesh(&gaia_mesh).unwrap();

        assert_eq!(
            kwavers_mesh.nodes[0].boundary_type,
            MeshBoundaryType::Dirichlet
        );
        assert_eq!(
            kwavers_mesh.nodes[1].boundary_type,
            MeshBoundaryType::Dirichlet
        );
        assert_eq!(
            kwavers_mesh.nodes[2].boundary_type,
            MeshBoundaryType::Dirichlet
        );
        assert_eq!(
            kwavers_mesh.nodes[3].boundary_type,
            MeshBoundaryType::Interior
        );
        assert!((kwavers_mesh.statistics().total_volume - (1.0 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn rejects_gaia_cells_with_conflicting_boundary_condition_labels() {
        let mut gaia_mesh = IndexedMesh::<f64>::new();
        let v0 = gaia_mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 0.0));
        let v1 = gaia_mesh.add_vertex_pos(Point3r::new(1.0, 0.0, 0.0));
        let v2 = gaia_mesh.add_vertex_pos(Point3r::new(0.0, 1.0, 0.0));
        let v3 = gaia_mesh.add_vertex_pos(Point3r::new(0.0, 0.0, 1.0));

        let f0 = gaia_mesh.add_face(v0, v1, v2);
        let f1 = gaia_mesh.add_face(v0, v1, v3);
        let f2 = gaia_mesh.add_face(v0, v2, v3);
        let f3 = gaia_mesh.add_face(v1, v2, v3);
        let mut cell =
            Cell::tetrahedron(f0.as_usize(), f1.as_usize(), f2.as_usize(), f3.as_usize());
        cell.vertex_ids = vec![0, 1, 2, 3];
        gaia_mesh.add_cell(cell);
        gaia_mesh.mark_boundary(f0, "dirichlet");
        gaia_mesh.mark_boundary(f1, "neumann");

        let error = TetrahedralMesh::from_gaia_indexed_mesh(&gaia_mesh).unwrap_err();
        assert!(format!("{error}").contains("conflicting boundary types"));
    }
}
