//! Source illumination accumulation for RTM normalisation.
//!
//! The source illumination `Φ(x)` estimates how well the source wavefield
//! illuminates each subsurface point:
//! ```text
//! Φ(x) = Σ_t  S²(x, t)
//! ```
//! After migration, the image is divided by √Φ to compensate for uneven
//! illumination (see [`super::mod::post_process_image`]).
//!
//! Reference: Kaelin & Guitton (2006), "Imaging condition for nonlinear
//! scattering-based inversion", SEG Technical Program Expanded Abstracts.

use crate::core::error::KwaversResult;
use ndarray::{s, Array4, Zip};

use super::super::types::ReverseTimeMigration;

impl ReverseTimeMigration {
    /// Accumulate `Σ_t S²(x,t)` into `self.source_illumination`.
    pub(super) fn update_source_illumination(
        &mut self,
        source_wavefield: &Array4<f64>,
    ) -> KwaversResult<()> {
        let n_time_steps = source_wavefield.shape()[0];

        for t in 0..n_time_steps {
            let src = source_wavefield.slice(s![t, .., .., ..]);
            Zip::from(&mut self.source_illumination)
                .and(&src)
                .for_each(|illum, &s| *illum += s * s);
        }

        Ok(())
    }
}
