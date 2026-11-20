use crate::forward::viscoacoustic::solver::ViscoacousticMemorySolver;
use kwavers_core::error::KwaversError;
use kwavers_core::error::KwaversResult;
use leto::Array3;

impl ViscoacousticMemorySolver {
    /// Enable an **absorbing boundary layer** (sponge) of `thickness` cells on
    /// every face whose axis is long enough to hold it. Outgoing waves entering
    /// the layer are damped before they reach (and wrap around) the periodic
    /// boundary, suppressing artificial reflections.
    ///
    /// `gamma_max` \[s⁻¹] is the peak damping rate at the outermost cell; the
    /// rate ramps in as a quadratic profile `γ(d) = γ_max ((L-d)/L)²` over the
    /// layer depth `d ∈ [0, L)` (zero in the interior), summed across axes so
    /// corners damp in every direction. A smooth ramp keeps layer reflection low.
    /// Calling again rebuilds the profile; `thickness = 0` disables it, and an
    /// axis too short to host a layer contributes exactly zero damping as
    /// before. When no axis can host a layer of the requested thickness the
    /// retained damping state is cleared — the established no-layer result.
    /// The per-axis extent test is overflow-free: it compares the thickness
    /// against `n / 2` instead of doubling it, so an extreme `usize` input
    /// neither wraps the guard nor reaches `n - thickness` underflow.
    ///
    /// # Errors
    /// - Non-finite `gamma_max`: rejected before the decay field is replaced,
    ///   so a rejected reconfiguration leaves the existing damping state and
    ///   subsequent pressure evolution untouched.
    pub fn enable_absorbing_layer(
        &mut self,
        thickness: usize,
        gamma_max: f64,
    ) -> KwaversResult<()> {
        if !gamma_max.is_finite() {
            return Err(KwaversError::InvalidInput(
                "absorbing-layer gamma_max must be finite".to_owned(),
            ));
        }
        if thickness == 0 || gamma_max <= 0.0 {
            self.damping_decay = None;
            return Ok(());
        }
        // No axis can host a layer of this thickness: the summed γ profile is
        // identically zero, so retain the established no-layer result instead
        // of an all-ones decay grid. (An axis hosts the layer iff
        // `thickness < n.div_ceil(2)`, which is the original `n > 2 * thickness`
        // test computed without the `usize` overflow.)
        if thickness >= self.nx.div_ceil(2)
            && thickness >= self.ny.div_ceil(2)
            && thickness >= self.nz.div_ceil(2)
        {
            self.damping_decay = None;
            return Ok(());
        }
        // Per-axis ramp: γ contribution at index `i` along an axis of extent `n`.
        let ramp = |i: usize, n: usize| -> f64 {
            // Overflow-free extent test: `thickness` may be any `usize`, so it
            // must never be doubled (`2 * thickness` wraps, and the wrapped
            // guard lets `n - thickness` underflow). `thickness >= n.div_ceil(2)`
            // holds for exactly the axes the wrapped expression rejected.
            if thickness >= n.div_ceil(2) {
                return 0.0; // axis too short to host the layer (e.g. singleton)
            }
            let l = thickness as f64;
            let depth = if i < thickness {
                (thickness - i) as f64 // distance from the low boundary
            } else if i >= n - thickness {
                (i - (n - thickness) + 1) as f64 // distance from the high boundary
            } else {
                0.0
            };
            let frac = depth / l;
            gamma_max * frac * frac
        };

        let dt = self.dt;
        let decay = Array3::from_shape_fn((self.nx, self.ny, self.nz), |[i, j, k]| {
            let gamma = ramp(i, self.nx) + ramp(j, self.ny) + ramp(k, self.nz);
            (-gamma * dt).exp()
        });
        self.damping_decay = Some(decay);
        Ok(())
    }

    /// Test-only view of the retained damping field for the finite-domain
    /// state-preservation assertions: a rejected reconfiguration must leave
    /// this value untouched.
    #[cfg(test)]
    pub(crate) fn damping_decay_state(&self) -> Option<&Array3<f64>> {
        self.damping_decay.as_ref()
    }
}
