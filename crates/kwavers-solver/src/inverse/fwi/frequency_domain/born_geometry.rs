//! Model-independent geometry for one Born frequency observation.

use self::field::{incident_field, outgoing_green, voxel_centers};
use super::acquisition::TransmissionAcquisition;
use super::types::Config;
use aequitas::systems::si::units::Meter;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::Complex64;
use std::collections::BTreeMap;

pub(super) mod field;
#[cfg(test)]
mod product_tests;
pub(super) mod products;
#[cfg(test)]
mod tests;

pub(super) struct BornTransmit {
    pub(super) incident: Vec<Complex64>,
    pub(super) direct: Vec<Complex64>,
    pub(super) receivers: Vec<usize>,
}

/// Immutable coefficients of the single-scatter observation operator.
///
/// `G(x,r; k_ref)`, incident fields and direct source-to-receiver pressures
/// depend only on geometry, frequency and the reference medium. A solve holds
/// those inputs fixed while changing the slowness/scattering potential. Each
/// coefficient is computed by the original equation once, with the original
/// argument and summation order. Reusing it is bitwise equivalent to computing
/// it again; neither model-dependent predictions nor gradients are retained.
///
/// Receiver fields are interned by all three SI-coordinate bit patterns.
/// This preserves signed zero and never merges approximately equal positions.
/// Transmit-dependent receivers retain their own row mapping. For `V` voxels,
/// `T` transmits, `U` unique receiver positions and `R` receivers per transmit,
/// storage is `(T + U)V + TR` complex values plus `TR` receiver indices per
/// frequency. Model-dependent scratch remains O(V); no Jacobian is assembled.
pub(super) struct BornGeometry {
    pub(super) transmits: Vec<BornTransmit>,
    pub(super) fields: Vec<Vec<Complex64>>,
    pub(super) products: Option<Vec<products::ProductRow>>,
}

impl BornGeometry {
    /// Consumers validate shape, frequency, acquisition and row count first.
    pub(super) fn new(
        shape: [usize; 3],
        frequency_hz: f64,
        acquisition: &dyn TransmissionAcquisition,
        config: &Config,
        transmissions: usize,
    ) -> Self {
        let centers = voxel_centers((shape[0], shape[1], shape[2]), config.spacing_m);
        let wavenumber = TWO_PI * frequency_hz * (1.0 / config.reference_sound_speed_m_s);
        let min_distance = 0.5 * config.spacing_m;
        let mut fields = Vec::new();
        let mut positions = BTreeMap::new();
        let mut transmits = Vec::with_capacity(transmissions);
        for transmit in 0..transmissions {
            let sources = acquisition.sources(transmit);
            let mut row = BornTransmit {
                incident: incident_field(sources, &centers, wavenumber, min_distance),
                direct: Vec::with_capacity(acquisition.receiver_count()),
                receivers: Vec::with_capacity(acquisition.receiver_count()),
            };
            for &receiver in acquisition.receivers(transmit) {
                let key = [
                    receiver.x.in_unit::<Meter>().to_bits(),
                    receiver.y.in_unit::<Meter>().to_bits(),
                    receiver.z.in_unit::<Meter>().to_bits(),
                ];
                let field = *positions.entry(key).or_insert_with(|| {
                    let index = fields.len();
                    fields.push(
                        centers
                            .iter()
                            .map(|(_, point)| {
                                outgoing_green(*point, receiver, wavenumber, min_distance)
                            })
                            .collect(),
                    );
                    index
                });
                row.receivers.push(field);
                row.direct.push(
                    sources
                        .iter()
                        .map(|&source| outgoing_green(source, receiver, wavenumber, min_distance))
                        .sum(),
                );
            }
            transmits.push(row);
        }
        Self {
            transmits,
            fields,
            products: None,
        }
    }
}
