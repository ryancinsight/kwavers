//! Types for plasmonic nanoparticle models and coupling

use aequitas::systems::si::quantities::Length;

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
        /// Center-to-center particle spacing.
        spacing: Length,
    },
    /// 2D square lattice
    Square {
        /// Lattice spacing along x.
        spacing_x: Length,
        /// Lattice spacing along y.
        spacing_y: Length,
    },
    /// 3D cubic lattice
    Cubic {
        /// Lattice spacing along x.
        spacing_x: Length,
        /// Lattice spacing along y.
        spacing_y: Length,
        /// Lattice spacing along z.
        spacing_z: Length,
    },
    /// Random homogeneous distribution
    Random,
}
