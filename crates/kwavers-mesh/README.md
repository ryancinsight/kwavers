# kwavers-mesh

Tetrahedral mesh infrastructure for [kwavers](https://github.com/ryancinsight/kwavers)
finite-element solvers: nodes, connectivity, statistics, quality metrics, and the gaia
bridge.

Where the rest of kwavers discretizes onto a structured `Grid`, the FEM and BEM paths need
an unstructured tetrahedral representation. This crate owns that representation and
nothing else.

## Boundary with gaia

`gaia` is the authoritative external mesh/STL boundary for the Atlas stack. This crate
keeps only the solver-facing `TetrahedralMesh` and converts inbound geometry through
`TetrahedralMesh::from_gaia_indexed_mesh`; it maintains no independent STL reader or
writer.

## What it provides

| Item | Responsibility |
|---|---|
| `TetrahedralMesh` | Nodes, elements, connectivity, point location |
| `Tetrahedron` / `MeshNode` | Element and node records |
| `MeshBoundaryType` | Boundary classification carried on nodes |
| `MeshStatistics` | Element count, volume, and quality metrics |
| `BoundingBox` | Mesh extent queries |

A mesh can also be generated directly from a structured grid's vertices via
`TetrahedralMesh::from_grid_vertices`, which is the usual route for FEM validation cases.

## Example

```rust
use kwavers_mesh::{MeshBoundaryType, TetrahedralMesh};

let mut mesh = TetrahedralMesh::new();

let a = mesh.add_node([0.0, 0.0, 0.0], MeshBoundaryType::Interior);
let b = mesh.add_node([1.0, 0.0, 0.0], MeshBoundaryType::Interior);
let c = mesh.add_node([0.0, 1.0, 0.0], MeshBoundaryType::Interior);
let d = mesh.add_node([0.0, 0.0, 1.0], MeshBoundaryType::Interior);

mesh.add_element([a, b, c, d], 0).unwrap();

let stats = mesh.statistics();
assert_eq!(stats.num_elements, 1);
```

## Documentation

- API reference: <https://docs.rs/kwavers-mesh>
- Workspace overview and crate map: [kwavers README](https://github.com/ryancinsight/kwavers#readme)

## License

MIT
