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
pub enum PlasmonicArrayGeometry {
    /// Linear chain
    Linear {
        /// Center-to-center particle spacing [m].
        spacing: f64,
    },
    /// 2D square lattice
    Square {
        /// Lattice spacing along x [m].
        spacing_x: f64,
        /// Lattice spacing along y [m].
        spacing_y: f64,
    },
    /// 3D cubic lattice
    Cubic {
        /// Lattice spacing along x [m].
        spacing_x: f64,
        /// Lattice spacing along y [m].
        spacing_y: f64,
        /// Lattice spacing along z [m].
        spacing_z: f64,
    },
    /// Random homogeneous distribution
    Random,
}
