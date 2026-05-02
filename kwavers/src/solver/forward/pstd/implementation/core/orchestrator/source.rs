use super::super::source_injection;
use super::PSTDSolver;
use crate::core::error::KwaversResult;
use crate::domain::source::{Source, SourceField};
use std::sync::Arc;

impl PSTDSolver {
    pub(crate) fn add_source_arc(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        let mask = source.create_mask(&self.grid);
        let mode = source_injection::determine_injection_mode(&mask);

        let grad_mask: Option<ndarray::Array3<f64>> = match source.source_type() {
            SourceField::VelocityX => {
                if let Some(ops) = &self.kspace_operators {
                    Some(ops.spectral_grad_x(&mask)?)
                } else {
                    None
                }
            }
            SourceField::VelocityY => {
                if let Some(ops) = &self.kspace_operators {
                    Some(ops.spectral_grad_y(&mask)?)
                } else {
                    None
                }
            }
            SourceField::VelocityZ => {
                if let Some(ops) = &self.kspace_operators {
                    Some(ops.spectral_grad_z(&mask)?)
                } else {
                    None
                }
            }
            _ => None,
        };

        self.dynamic_sources.push((source, mask));
        self.source_injection_modes.push(mode);
        self.velocity_source_grad_masks.push(grad_mask);
        Ok(())
    }
}
