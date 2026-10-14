//! Bounded preparation of the immutable spatial part of the Born operator.

use super::super::born_kernel::{
    arithmetic::Arithmetic,
    fields::{FactoredProducts, FieldProducts},
    VOXELS_PER_BLOCK,
};
use super::BornGeometry;
use kwavers_math::fft::Complex64;

/// Per-inversion bound on requested expanded vector allocations, including row
/// metadata. Preparation uses three fixed blocks (12 KiB), without voxel-sized
/// temporary allocations. Allocator bookkeeping is outside this byte count.
pub(in super::super) const WORKSPACE_BYTES: usize = 8 * 1024 * 1024;

pub(in super::super) struct ProductRow {
    voxels: usize,
    unscaled: Vec<Complex64>,
    unit_scaled: Vec<Complex64>,
}

impl ProductRow {
    /// Receiver order is the acquisition's original order; both arrays have
    /// R*V entries. Validate the index before multiplying it by the stride.
    #[inline]
    pub(in super::super) fn fields(&self, receiver: usize) -> (&[Complex64], &[Complex64]) {
        assert!(receiver < self.unscaled.len() / self.voxels);
        let start = receiver * self.voxels;
        let end = start + self.voxels;
        (&self.unscaled[start..end], &self.unit_scaled[start..end])
    }
}

impl BornGeometry {
    /// `I*G` and `(1*I)*G` depend only on the already immutable geometry.
    /// Preserve both expression trees, including signed-zero behavior. Every
    /// model-dependent weight, prediction, source fit and gradient remains fresh.
    /// If the representation does not fit or contains nonfinite products, keep
    /// the complete factored computation. No selected operation is omitted.
    pub(in super::super) fn prepare_products(&mut self, budget: usize) -> usize {
        self.products = None;
        let required = self.transmits.iter().try_fold(0usize, |total, row| {
            let payload = row
                .receivers
                .len()
                .checked_mul(row.incident.len())?
                .checked_mul(2 * std::mem::size_of::<Complex64>())?;
            total
                .checked_add(payload)?
                .checked_add(std::mem::size_of::<ProductRow>())
        });
        let Some(required) = required.filter(|&bytes| bytes <= budget) else {
            return 0;
        };
        let mut products = Vec::with_capacity(self.transmits.len());
        let mut raw = [Complex64::new(0.0, 0.0); VOXELS_PER_BLOCK];
        let mut scaled_incident = raw;
        let mut unit = raw;
        for row in &self.transmits {
            let voxels = row.incident.len();
            let capacity = row.receivers.len() * voxels;
            let mut prepared = ProductRow {
                voxels,
                unscaled: Vec::with_capacity(capacity),
                unit_scaled: Vec::with_capacity(capacity),
            };
            for &field in &row.receivers {
                let green = &self.fields[field];
                let fields = FactoredProducts::new(&row.incident, green);
                let mut start = 0;
                while start < voxels {
                    let len = (voxels - start).min(VOXELS_PER_BLOCK);
                    let end = start + len;
                    fields.write_products(start..end, &mut raw);
                    let mut lane = 0;
                    while lane < len {
                        scaled_incident[lane] = Arithmetic::product(
                            Complex64::new(1.0, 0.0),
                            row.incident[start + lane],
                        );
                        lane += 1;
                    }
                    FactoredProducts::new(&scaled_incident[..len], &green[start..end])
                        .write_products(0..len, &mut unit);
                    // Each prefix receives exactly the corresponding Copy
                    // values. The preallocated capacity is R*V; appending all
                    // clipped blocks reaches it without growth or Clone calls.
                    let mut lane = 0;
                    while lane < len {
                        let (raw, unit) = (raw[lane], unit[lane]);
                        if !raw.re.is_finite()
                            || !raw.im.is_finite()
                            || !unit.re.is_finite()
                            || !unit.im.is_finite()
                        {
                            return 0;
                        }
                        prepared.unscaled.push(raw);
                        prepared.unit_scaled.push(unit);
                        lane += 1;
                    }
                    start = end;
                }
            }
            products.push(prepared);
        }
        self.products = Some(products);
        required
    }
}
