use super::super::source_injection;
use super::PSTDSolver;
use kwavers_core::error::KwaversResult;
use kwavers_source::{Source, SourceField};
use std::sync::Arc;

impl PSTDSolver {
    /// Add source arc.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub(crate) fn add_source_arc(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        let mask_leto = source.create_mask(&self.grid);
        let mask = ndarray_mask(&mask_leto);
        let mode = source_injection::determine_injection_mode(&mask);

        let grad_mask: Option<leto::Array3<f64>> = match source.source_type() {
            SourceField::VelocityX => {
                if let Some(ops) = &self.kspace_operators {
                    Some(ops.spectral_grad_x(&mask_leto)?)
                } else {
                    None
                }
            }
            SourceField::VelocityY => {
                if let Some(ops) = &self.kspace_operators {
                    Some(ops.spectral_grad_y(&mask_leto)?)
                } else {
                    None
                }
            }
            SourceField::VelocityZ => {
                if let Some(ops) = &self.kspace_operators {
                    Some(ops.spectral_grad_z(&mask_leto)?)
                } else {
                    None
                }
            }
            SourceField::Pressure => None,
        };

        self.dynamic_sources.push((source, mask));
        self.source_injection_modes.push(mode);
        self.velocity_source_grad_masks.push(grad_mask);
        Ok(())
    }
}

fn ndarray_mask(mask: &leto::Array3<f64>) -> leto::Array3<f64> {
    let [nx, ny, nz] = mask.shape();
    leto::Array3::from_shape_vec([nx, ny, nz], mask.iter().copied().collect())
        .expect("PSTD source mask shape must match contiguous ndarray storage")
}
