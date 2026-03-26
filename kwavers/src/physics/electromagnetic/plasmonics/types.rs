//! Types for plasmonic nanoparticle models and coupling

/// Inter-particle coupling models for dense nanoparticle dispersions
#[derive(Debug, Clone, PartialEq)]
pub enum CouplingModel {
    /// No coupling (dilute limit)
    None,
    /// Dipole-dipole coupling approximation
    DipoleDipole,
    /// Quasi-static approximation for dense media (Bruggeman effective medium)
    QuasiStatic,
}

/// Geometries for coherent nanoparticle arrays
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayGeometry {
    /// Linear chain
    Linear { spacing: f64 },
    /// 2D square lattice
    Square { spacing_x: f64, spacing_y: f64 },
    /// 3D cubic lattice
    Cubic {
        spacing_x: f64,
        spacing_y: f64,
        spacing_z: f64,
    },
    /// Random homogeneous distribution
    Random,
}
