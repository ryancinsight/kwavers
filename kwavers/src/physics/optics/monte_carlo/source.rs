use crate::physics::optics::monte_carlo::utils::{get_perpendicular, normalize, sample_isotropic_direction};
use crate::physics::optics::monte_carlo::photon::Photon;
use rand::Rng;

/// Photon source specification
#[derive(Clone, Debug)]
pub enum PhotonSource {
    /// Pencil beam (collimated)
    PencilBeam {
        origin: [f64; 3],
        direction: [f64; 3],
    },

    /// Gaussian beam profile
    Gaussian {
        origin: [f64; 3],
        direction: [f64; 3],
        beam_waist: f64,
    },

    /// Isotropic point source
    Isotropic { origin: [f64; 3] },
}

impl PhotonSource {
    /// Create pencil beam source
    pub fn pencil_beam(origin: [f64; 3], direction: [f64; 3]) -> Self {
        Self::PencilBeam { origin, direction }
    }

    /// Create Gaussian beam source
    pub fn gaussian(origin: [f64; 3], direction: [f64; 3], beam_waist: f64) -> Self {
        Self::Gaussian {
            origin,
            direction,
            beam_waist,
        }
    }

    /// Create isotropic point source
    pub fn isotropic(origin: [f64; 3]) -> Self {
        Self::Isotropic { origin }
    }

    /// Launch a photon based on current source type
    pub(crate) fn launch_photon<R: Rng>(&self, rng: &mut R) -> Photon {
        match self {
            Self::PencilBeam { origin, direction } => Photon {
                position: *origin,
                direction: normalize(*direction),
                weight: 1.0,
                alive: true,
            },

            Self::Gaussian {
                origin,
                direction,
                beam_waist,
            } => {
                // Sample from 2D Gaussian profile
                let r = beam_waist * (-2.0 * rng.gen::<f64>().ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * rng.gen::<f64>();

                // Perpendicular directions to beam
                let dir_norm = normalize(*direction);
                let perp1 = get_perpendicular(dir_norm);
                let perp2 = crate::physics::optics::monte_carlo::utils::cross(dir_norm, perp1);

                let offset = [
                    r * theta.cos() * perp1[0] + r * theta.sin() * perp2[0],
                    r * theta.cos() * perp1[1] + r * theta.sin() * perp2[1],
                    r * theta.cos() * perp1[2] + r * theta.sin() * perp2[2],
                ];

                Photon {
                    position: [
                        origin[0] + offset[0],
                        origin[1] + offset[1],
                        origin[2] + offset[2],
                    ],
                    direction: dir_norm,
                    weight: 1.0,
                    alive: true,
                }
            }

            Self::Isotropic { origin } => {
                // Sample uniformly on unit sphere
                let dir = sample_isotropic_direction(rng);
                Photon {
                    position: *origin,
                    direction: dir,
                    weight: 1.0,
                    alive: true,
                }
            }
        }
    }
}
