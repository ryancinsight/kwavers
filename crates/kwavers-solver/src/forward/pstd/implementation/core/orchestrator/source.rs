use super::super::source_injection;
use super::PSTDSolver;
use kwavers_core::error::KwaversResult;
use kwavers_source::{Source, SourceField};
use leto::Array3 as LetoArray3;
use std::sync::Arc;

impl PSTDSolver {
    /// Add source arc.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn add_source_arc(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        let mask = source.create_mask(&self.grid);
        let mode = source_injection::determine_injection_mode(&mask);

        let grad_mask: Option<ndarray::Array3<f64>> = match source.source_type() {
            SourceField::VelocityX => {
                if let Some(ops) = &self.kspace_operators {
                    let mask = leto_mask(&mask);
                    Some(ops.spectral_grad_x(&mask)?)
                } else {
                    None
                }
            }
            SourceField::VelocityY => {
                if let Some(ops) = &self.kspace_operators {
                    let mask = leto_mask(&mask);
                    Some(ops.spectral_grad_y(&mask)?)
                } else {
                    None
                }
            }
            SourceField::VelocityZ => {
                if let Some(ops) = &self.kspace_operators {
                    let mask = leto_mask(&mask);
                    Some(ops.spectral_grad_z(&mask)?)
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

fn leto_mask(mask: &ndarray::Array3<f64>) -> LetoArray3<f64> {
    let (nx, ny, nz) = mask.dim();
    LetoArray3::from_shape_vec([nx, ny, nz], mask.iter().copied().collect())
        .expect("PSTD source mask shape must match its Leto gradient shape")
}
