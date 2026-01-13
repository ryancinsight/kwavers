//! Core electromagnetic types and enumerations
//!
//! This module defines the fundamental types used throughout the electromagnetic
//! physics implementation.

/// Spatial dimension for electromagnetic problems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EMDimension {
    /// 1D problems (z-direction only, TEM waves)
    One = 1,
    /// 2D problems (transverse magnetic/electric)
    Two = 2,
    /// 3D problems (full vector field)
    Three = 3,
}

/// Electromagnetic polarization state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Polarization {
    /// Linear polarization along x-axis
    LinearX,
    /// Linear polarization along y-axis
    LinearY,
    /// Right circular polarization
    RightCircular,
    /// Left circular polarization
    LeftCircular,
    /// Elliptical polarization (ratio, phase difference)
    Elliptical { ratio: f64, phase_diff: f64 },
}

/// Electromagnetic wave type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EMWaveType {
    /// Transverse electromagnetic (no longitudinal components)
    TEM,
    /// Transverse electric (E_z = 0)
    TE,
    /// Transverse magnetic (H_z = 0)
    TM,
    /// Hybrid mode (both E_z, H_z nonzero)
    Hybrid,
}

/// Nanoparticle geometry for plasmonics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NanoparticleGeometry {
    /// Spherical nanoparticle
    Sphere { radius: f64 },
    /// Ellipsoidal nanoparticle
    Ellipsoid { a: f64, b: f64, c: f64 },
    /// Nanorod (cylindrical)
    Nanorod { radius: f64, length: f64 },
    /// Nanoshell (core-shell)
    Nanoshell {
        core_radius: f64,
        shell_thickness: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_em_dimension_values() {
        assert_eq!(EMDimension::One as u32, 1);
        assert_eq!(EMDimension::Two as u32, 2);
        assert_eq!(EMDimension::Three as u32, 3);
    }

    #[test]
    fn test_polarization_equality() {
        assert_eq!(Polarization::LinearX, Polarization::LinearX);
        assert_ne!(Polarization::LinearX, Polarization::LinearY);
    }

    #[test]
    fn test_em_wave_type() {
        assert_eq!(EMWaveType::TEM, EMWaveType::TEM);
        assert_ne!(EMWaveType::TE, EMWaveType::TM);
    }

    #[test]
    fn test_nanoparticle_geometry() {
        let sphere = NanoparticleGeometry::Sphere { radius: 15e-9 };
        let ellipsoid = NanoparticleGeometry::Ellipsoid {
            a: 10e-9,
            b: 15e-9,
            c: 20e-9,
        };

        match sphere {
            NanoparticleGeometry::Sphere { radius } => assert_eq!(radius, 15e-9),
            _ => panic!("Expected sphere"),
        }

        match ellipsoid {
            NanoparticleGeometry::Ellipsoid { a, b, c } => {
                assert_eq!(a, 10e-9);
                assert_eq!(b, 15e-9);
                assert_eq!(c, 20e-9);
            }
            _ => panic!("Expected ellipsoid"),
        }
    }
}
