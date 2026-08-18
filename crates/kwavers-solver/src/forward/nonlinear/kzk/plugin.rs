//! Plugin adapter wrapping the correct complex-field [`KZKSolver`] for the
//! [`Plugin`](crate::plugin::Plugin) trait system.
//!
//! The [`KZKSolver`] is a parabolic beam-propagation solver that advances
//! through z-planes using Strang operator splitting.  This adapter runs the
//! full z-propagation on the first [`Plugin::update`] call and serves cached
//! z-slices thereafter, matching the therapy `execution.rs` pattern where the
//! entire volume is computed before per-plane readout.
//!
//! # Coordinate mapping
//!
//! The therapy frame uses axial = x (`grid.nx`), transverse = (y, z).
//! The KZK solver uses axial = z, transverse = (x, y).  The adapter remaps:
//!
//! | KZK field | Therapy field |
//! |-----------|---------------|
//! | `nx`      | `grid.ny`     |
//! | `ny`      | `grid.nz`     |
//! | `nz`      | `grid.nx`     |
//! | `dx`      | `grid.dy`     |
//! | `dz`      | `grid.dx`     |

use kwavers_core::error::KwaversError;
use kwavers_core::error::KwaversResult;
use kwavers_field::mapping::UnifiedFieldType;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::acoustics::wave_propagation::KZKSolverTrait;
use leto::{Array2, Array3, Array4};

use super::solver::KZKSolver;
use super::KZKConfig;
use crate::plugin::{PluginMetadata, PluginState};

/// Plugin adapter for the correct complex-field KZK solver.
///
/// Wraps [`KZKSolver`] behind the [`crate::plugin::Plugin`] trait so the
/// [`PhysicsCatalog`](crate::plugin::catalog::PhysicsCatalog) can instantiate it.
pub struct KzkPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    solver: Option<KZKSolver>,
    config: Option<KZKConfig>,
    /// Cached RMS volume from full propagation, shape `(grid.ny, grid.nz, grid.nx)`.
    cached_volume: Option<Array3<f64>>,
    /// Current z-slice index into the cached volume.
    current_step: usize,
}

impl std::fmt::Debug for KzkPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KzkPlugin")
            .field("metadata", &self.metadata)
            .field("state", &self.state)
            .field("has_solver", &self.solver.is_some())
            .field("has_volume", &self.cached_volume.is_some())
            .field("current_step", &self.current_step)
            .finish()
    }
}

// SAFETY: KZKSolver owns all its data (Array3<Complex64>, FFTW plans,
// operators).  All inner types are Send + Sync — complex arrays, real
// arrays, and FFTW forward/inverse plan wrappers that are safe to share
// across threads.  KZKSolver contains no raw pointers or thread-local
// state.
unsafe impl Send for KzkPlugin {}
unsafe impl Sync for KzkPlugin {}

impl Default for KzkPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl KzkPlugin {
    /// Create new KZK plugin adapter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "kzk_solver".to_owned(),
                name: "KZK Beam Propagation Solver".to_owned(),
                version: "1.0.0".to_owned(),
                author: "Kwavers Team".to_owned(),
                description: "Complex-field KZK beam propagation using Strang splitting".to_owned(),
                license: "MIT".to_owned(),
            },
            state: PluginState::Created,
            solver: None,
            config: None,
            cached_volume: None,
            current_step: 0,
        }
    }

    /// Build a [`KZKConfig`] from grid and medium parameters.
    ///
    /// Applies the coordinate mapping: KZK axial = therapy x, KZK transverse =
    /// therapy (y, z).
    fn build_config(grid: &Grid, medium: &dyn Medium) -> KZKConfig {
        let c0 = kwavers_medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
        let rho0 = kwavers_medium::density_at(medium, 0.0, 0.0, 0.0, grid);
        let b_over_a = medium.nonlinearity_parameter(0.0, 0.0, 0.0, grid);
        let alpha0 = medium.alpha_coefficient(0.0, 0.0, 0.0, grid);
        let alpha_power = medium.alpha_power(0.0, 0.0, 0.0, grid);

        // CFL constraint: c0 * dt / dz <= 0.5 for the parabolic approximation.
        let dz = grid.dx; // therapy axial spacing → KZK axial step
        let mut dt = 0.4 * dz / c0; // conservative CFL = 0.4

        // Nyquist: dt <= 1 / (20 * f_max), where f_max ≈ 10 * fundamental.
        let frequency = 1.0e6; // 1 MHz default
        let f_max = 10.0 * frequency;
        let dt_nyquist = 1.0 / (20.0 * f_max);
        if dt > dt_nyquist {
            dt = dt_nyquist;
        }

        // Number of retarded-time samples: enough to resolve the waveform.
        let period = 1.0 / frequency;
        let mut nt = (period / dt).ceil() as usize;
        if nt < 64 {
            nt = 64; // minimum for spectral accuracy
        }

        KZKConfig {
            // Coordinate mapping: KZK nx,ny = therapy ny,nz (transverse);
            // KZK nz = therapy nx (axial).
            nx: grid.ny,
            ny: grid.nz,
            nz: grid.nx,
            dx: grid.dy,
            dz,
            dt,
            nt,
            c0,
            rho0,
            b_over_a,
            alpha0,
            alpha_power,
            include_diffraction: true,
            include_absorption: true,
            include_nonlinearity: true,
            frequency,
        }
    }

    /// Extract a 2D source amplitude map from the current 3D pressure field.
    ///
    /// Collapses the therapy axial (x) dimension by summing absolute values,
    /// producing a `(grid.ny, grid.nz)` amplitude map that maps to KZK
    /// transverse coordinates.
    fn extract_source(fields: &Array4<f64>, grid: &Grid) -> Array2<f64> {
        let kzk_nx = grid.ny;
        let kzk_ny = grid.nz;
        let mut source = Array2::<f64>::zeros((kzk_nx, kzk_ny));

        let pressure_field = fields
            .index_axis::<3>(0, UnifiedFieldType::Pressure.index())
            .expect("invariant: pressure field index within field stack");

        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let mut amp = 0.0_f64;
                for ix in 0..grid.nx {
                    amp += pressure_field[[ix, iy, iz]].abs();
                }
                source[[iy, iz]] = amp;
            }
        }

        // Normalise so the peak amplitude is 1 Pa (unit source).
        let peak = source.iter().cloned().fold(0.0_f64, f64::max);
        if peak > 0.0 {
            for v in source.iter_mut() {
                *v /= peak;
            }
        }

        source
    }
}

impl crate::plugin::Plugin for KzkPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        let config = Self::build_config(grid, medium);

        let solver = KZKSolver::new(config.clone())
            .map_err(|msg| KwaversError::InternalError(format!("KZKSolver::new failed: {msg}")))?;

        self.solver = Some(solver);
        self.config = Some(config);
        self.cached_volume = None;
        self.current_step = 0;
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        _context: &mut crate::plugin::PluginContext<'_>,
    ) -> KwaversResult<()> {
        // On first call: run full z-propagation and cache the RMS volume.
        if self.cached_volume.is_none() {
            let solver = self.solver.as_mut().ok_or_else(|| {
                KwaversError::InternalError("KzkPlugin::update called before initialize".to_owned())
            })?;

            let source = Self::extract_source(fields, grid);
            solver.set_source(source, 1.0e6);

            // Collect RMS at each z-plane → volume shape (kzk_nx, kzk_ny, nz).
            let kzk_nx = solver.config.nx;
            let kzk_ny = solver.config.ny;
            let total_z = solver.config.nz;
            let mut volume = Array3::<f64>::zeros((kzk_nx, kzk_ny, total_z));

            for iz in 0..total_z {
                solver.step();
                let rms_2d = solver.current_field(); // (kzk_nx, kzk_ny)
                for i in 0..kzk_nx {
                    for j in 0..kzk_ny {
                        volume[[i, j, iz]] = rms_2d[[i, j]];
                    }
                }
            }

            self.cached_volume = Some(volume);
        }

        // Write the current z-slice back into the pressure field.
        if let Some(ref volume) = self.cached_volume {
            let total_z = volume.shape()[2];
            let step = self.current_step.min(total_z.saturating_sub(1));
            let kzk_nx = volume.shape()[0];
            let kzk_ny = volume.shape()[1];

            let mut pressure_slice = fields
                .index_axis_mut::<3>(0, UnifiedFieldType::Pressure.index())
                .expect("invariant: pressure field index within field stack");

            // Write RMS at the current axial plane (therapy x = step).
            let therapy_x = step.min(grid.nx.saturating_sub(1));
            for iy in 0..grid.ny.min(kzk_nx) {
                for iz in 0..grid.nz.min(kzk_ny) {
                    pressure_slice[[therapy_x, iy, iz]] = volume[[iy, iz, step]];
                }
            }

            if self.current_step < total_z {
                self.current_step += 1;
            }
        }

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.solver = None;
        self.config = None;
        self.cached_volume = None;
        self.current_step = 0;
        self.state = PluginState::Finalized;
        Ok(())
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.solver = None;
        self.config = None;
        self.cached_volume = None;
        self.current_step = 0;
        self.state = PluginState::Created;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::test_support::{make_context, null_plugin_fields, NullBoundary};
    use crate::plugin::Plugin;
    use kwavers_core::constants::acoustic_parameters::NP_TO_DB;
    use kwavers_core::constants::numerical::{CM_TO_M, MHZ_TO_HZ};
    use kwavers_core::constants::tissue_acoustics::B_OVER_A_WATER;
    use kwavers_core::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;
    use leto::Array4;

    /// Attenuation coefficient used by the plane-wave oracles, in the clinical
    /// unit the `Medium` trait and [`KZKConfig::alpha0`] both carry.
    const ALPHA0_DB_PER_CM_MHZ: f64 = 0.5;
    /// Power-law exponent y in α(f) = α₀·f^y.  y = 1 keeps α(f₀) = α₀ so the
    /// oracle needs no frequency scaling.
    const ALPHA_POWER: f64 = 1.0;

    /// Round-off bound for a 32-step plane-wave propagation.
    ///
    /// ## Derivation
    ///
    /// Every term below is a relative error on the measured RMS ratio.
    ///
    /// 1. **Absorption is spectrally exact.**  The source is `sin(ω₀τ)`
    ///    sampled at `nt = 200`, `Δτ = 5 ns`, `f₀ = 1 MHz`, so the fundamental
    ///    sits on DFT bin `k₀ = f₀·nt·Δτ = 1` exactly (asserted below as a
    ///    precondition).  With zero spectral leakage the per-step factor is
    ///    exactly `exp(−α(f₀)·Δz)` (see `absorption.rs`, spectral-exactness
    ///    theorem), so absorption contributes no truncation error at all.
    /// 2. **Diffraction is the identity.**  A transversely uniform field has
    ///    all energy at `k_T = 0`, where `H = exp(−i·0·Δz/2k₀) = 1`.
    /// 3. **FFT round-off.**  Higham (2002) §24.1 bounds a radix-2 FFT of
    ///    length N by `3·log₂(N)·ε`.  Per z-step: absorption runs 4 length-200
    ///    transforms (`4 × 3 × 7.65 × ε ≈ 2.0e−14`) and diffraction 4
    ///    length-256 2-D transforms (`4 × 3 × 8 × ε ≈ 2.1e−14`).  Over 32
    ///    steps: `≈ 1.3e−12`.
    /// 4. **RMS reduction.**  A sequential sum of `nt = 200` terms is bounded
    ///    by `nt·ε ≈ 4.4e−14`, taken twice (numerator and denominator):
    ///    `≈ 8.9e−14`.
    /// 5. **Strang commutator.**  Absorption and nonlinearity do not commute.
    ///    The per-step defect scales as `(α·Δz)²·σ/12` where the nonlinear
    ///    distortion per step is `σ = β·ω₀·p₀·Δz/(ρ₀c₀³) ≈ 7e−9` for the
    ///    1 Pa peak-normalised source, giving `≈ 2e−14` per step, `6e−13`
    ///    over 32 steps.
    /// 6. **Harmonic content in the RMS.**  Second-harmonic amplitude
    ///    `≈ σ_total/2 ≈ 8e−8` enters the RMS quadratically: `≈ 3e−15`.
    ///
    /// Sum ≈ `2e−12`.  The bound below carries a factor-5 margin over that
    /// sum, covering the unproven constant in the FFT error model.  It is a
    /// round-off bound, not a physics tolerance: any real defect in the
    /// absorption law, the unit conversion, or the axial mapping moves the
    /// ratio by parts in a thousand or more.
    const PLANE_WAVE_ROUNDOFF: f64 = 1.0e-11;

    fn small_grid() -> Grid {
        // Axial-elongated: grid.nx (therapy axial) > grid.ny so the KZK
        // parabolic angle check (atan(KZK_nx·dx / (2·KZK_nz·dz))) passes.
        // KZK_nx = grid.ny = 16, KZK_nz = grid.nx = 32 → atan(16/64) ≈ 14°.
        Grid::new(32, 16, 16, 1e-3, 1e-3, 1e-3).expect("grid")
    }

    /// Water with a power-law absorption of α₀ = 0.5 dB/(cm·MHz), y = 1.
    fn absorbing_water(grid: &Grid) -> HomogeneousMedium {
        let mut medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, grid);
        medium
            .set_acoustic_properties(ALPHA0_DB_PER_CM_MHZ, ALPHA_POWER, B_OVER_A_WATER)
            .expect("set_acoustic_properties");
        medium
    }

    /// Plane-wave attenuation coefficient in Np/m at `frequency_hz`.
    ///
    /// Mirrors the clinical → SI conversion in
    /// [`super::super::absorption::KzkAbsorptionOperator::new`]:
    /// `α₀[Np/(m·Hz^y)] = α₀[dB/(cm·MHz^y)] / CM_TO_M / NP_TO_DB / (1e6)^y`.
    fn alpha_np_per_m(frequency_hz: f64) -> f64 {
        ALPHA0_DB_PER_CM_MHZ / CM_TO_M / NP_TO_DB * (frequency_hz / MHZ_TO_HZ).powf(ALPHA_POWER)
    }

    /// Assert the preconditions the round-off tolerance rests on.
    ///
    /// The derivation of [`PLANE_WAVE_ROUNDOFF`] assumes the fundamental lands
    /// on an exact DFT bin.  If [`KzkPlugin::build_config`] ever changes `dt`
    /// or `nt`, spectral leakage would spread energy onto bins with a
    /// different α and the bound would no longer hold — so the assumption is
    /// checked rather than trusted.
    fn assert_exact_fft_bin(config: &KZKConfig) {
        let bin = config.frequency * config.nt as f64 * config.dt;
        assert!(
            (bin - bin.round()).abs() < 1.0e-12,
            "tolerance precondition: fundamental must sit on an exact DFT bin, \
             got k₀ = {bin} (nt = {}, dt = {} s, f₀ = {} Hz)",
            config.nt,
            config.dt,
            config.frequency
        );
        assert!(
            bin >= 1.0,
            "tolerance precondition: the retarded-time window must span at \
             least one period of the fundamental, got k₀ = {bin}"
        );
    }

    /// Drive the plugin over every axial plane and return the field it wrote.
    ///
    /// [`KzkPlugin::update`] runs the whole z-propagation on the first call and
    /// then writes cached plane `step` at therapy index `x = step`.  After
    /// `grid.nx` updates the therapy x-axis therefore holds the axial RMS
    /// profile `p_rms(z = (x+1)·Δz)` at every transverse cell, which is what
    /// the absorption and uniformity oracles read.
    fn sweep_all_planes(
        grid: &Grid,
        medium: &HomogeneousMedium,
        source: &Array4<f64>,
    ) -> Array4<f64> {
        let mut plugin = KzkPlugin::new();
        plugin.initialize(grid, medium).expect("initialize");

        let mut fields = source.clone();
        let extra = null_plugin_fields(grid);
        let mut boundary = NullBoundary;
        let mut ctx = make_context(&extra, &mut boundary);
        let dt = 1.0e-7;

        for step in 0..grid.nx {
            plugin
                .update(&mut fields, grid, medium, dt, step as f64 * dt, &mut ctx)
                .expect("update");
        }
        fields
    }

    /// Uniform 1 Pa plane wave over the whole therapy volume.
    fn uniform_plane_wave(grid: &Grid) -> Array4<f64> {
        let mut fields = Array4::<f64>::zeros((1, grid.nx, grid.ny, grid.nz));
        for value in fields.iter_mut() {
            *value = 1.0;
        }
        fields
    }

    /// Beer–Lambert oracle: a plane wave in a homogeneous absorber decays as
    /// `exp(−α·z)`.
    ///
    /// ## Why a ratio
    ///
    /// [`KzkPlugin::extract_source`] normalises the source plane to unit peak,
    /// which destroys absolute scale but leaves the axial profile untouched:
    /// every plane is scaled by the same constant.  The ratio between two
    /// depths is therefore scale-free and is the strongest oracle the adapter
    /// admits:
    ///
    /// ```text
    /// p_rms(z₂) / p_rms(z₁) = exp(−α(f₀)·(z₂ − z₁))
    /// ```
    ///
    /// with α(f₀) = 0.5 dB/(cm·MHz) × 1 MHz = 5.7565 Np/m and
    /// z₂ − z₁ = 24 mm, giving 0.87097.
    ///
    /// ## Reference
    ///
    /// Szabo TL (1994). J. Acoust. Soc. Am. 96(1), 491–500.
    #[test]
    fn plane_wave_decays_at_the_beer_lambert_rate() {
        let grid = small_grid();
        let medium = absorbing_water(&grid);
        let config = KzkPlugin::build_config(&grid, &medium);
        assert_exact_fft_bin(&config);

        let fields = sweep_all_planes(&grid, &medium, &uniform_plane_wave(&grid));

        // Two interior planes, clear of the first and last written slice.
        let (x1, x2) = (4_usize, 28_usize);
        let (cy, cz) = (grid.ny / 2, grid.nz / 2);
        let p1 = fields[[0, x1, cy, cz]];
        let p2 = fields[[0, x2, cy, cz]];

        assert!(
            p1 > 0.0,
            "plane wave must carry positive RMS amplitude at z₁, got {p1} Pa"
        );

        let span = (x2 - x1) as f64 * grid.dx;
        let expected = (-alpha_np_per_m(config.frequency) * span).exp();
        let measured = p2 / p1;
        let relative_error = (measured - expected).abs() / expected;

        assert!(
            relative_error < PLANE_WAVE_ROUNDOFF,
            "plane-wave absorption: expected p(z₂)/p(z₁) = {expected:.15}, got \
             {measured:.15} (relative error {relative_error:.3e} exceeds the \
             derived round-off bound {PLANE_WAVE_ROUNDOFF:.1e}); \
             α = {:.6} Np/m over {:.3} m",
            alpha_np_per_m(config.frequency),
            span
        );
    }

    /// Transverse-uniformity oracle: the parabolic propagator is exactly the
    /// identity on a transversely uniform field.
    ///
    /// ## Theorem
    ///
    /// A field with no transverse variation has all its energy in the
    /// `k_T = 0` bin, where the propagator is
    /// `H(0) = exp(−i·0²·Δz/(2k₀)) = 1`.  Absorption acts identically at every
    /// transverse cell and the nonlinear operator is pointwise, so a uniform
    /// plane wave stays uniform to round-off at every z-plane.
    ///
    /// This is the oracle that a diffraction operator indexed by real-space
    /// position instead of transverse wavenumber cannot satisfy: such an
    /// operator multiplies each cell by a different factor and shreds the
    /// uniformity immediately.
    #[test]
    fn plane_wave_stays_transversely_uniform() {
        let grid = small_grid();
        let medium = absorbing_water(&grid);
        let config = KzkPlugin::build_config(&grid, &medium);
        assert_exact_fft_bin(&config);

        let fields = sweep_all_planes(&grid, &medium, &uniform_plane_wave(&grid));

        for x in 0..grid.nx {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for iy in 0..grid.ny {
                for iz in 0..grid.nz {
                    let value = fields[[0, x, iy, iz]];
                    min = min.min(value);
                    max = max.max(value);
                }
            }
            assert!(
                max > 0.0,
                "plane {x}: uniform plane wave must retain positive amplitude, got max {max} Pa"
            );
            let spread = (max - min) / max;
            assert!(
                spread < PLANE_WAVE_ROUNDOFF,
                "plane {x}: transverse spread {spread:.3e} exceeds the derived \
                 round-off bound {PLANE_WAVE_ROUNDOFF:.1e} (min {min:.15} Pa, \
                 max {max:.15} Pa) — the k_T = 0 propagator is not the identity"
            );
        }

        // The propagation must be non-trivial: absorption removes amplitude.
        let (cy, cz) = (grid.ny / 2, grid.nz / 2);
        let first = fields[[0, 0, cy, cz]];
        let last = fields[[0, grid.nx - 1, cy, cz]];
        let span = (grid.nx - 1) as f64 * grid.dx;
        let expected = (-alpha_np_per_m(config.frequency) * span).exp();
        let measured = last / first;
        assert!(
            (measured - expected).abs() / expected < PLANE_WAVE_ROUNDOFF,
            "uniform plane wave must still be attenuated end to end: expected \
             {expected:.15}, got {measured:.15}"
        );
    }

    /// Adapter oracle: the plugin must reproduce the reference [`KZKSolver`]
    /// exactly, under the documented axis remap.
    ///
    /// The plugin owns no physics — it owns the therapy ↔ KZK coordinate map
    /// (`therapy (x, y, z) → KZK (z, x, y)`), the peak-normalised source
    /// extraction, and the cached per-plane readout.  Its contract is therefore
    /// differential: for every therapy index `(x, iy, iz)` the written value
    /// must equal `KZKSolver::current_field()[[iy, iz]]` after `x + 1` steps of
    /// a reference solver built from the same config and source.
    ///
    /// The source is deliberately anisotropic in y and z, so a transposed or
    /// rotated mapping fails rather than cancelling out.
    ///
    /// Equality is asserted bit for bit: both paths execute the same
    /// deterministic operator sequence on the same data, and every reduction
    /// in that sequence is sequential within a disjoint slice, so the chunking
    /// chosen by the parallel scheduler cannot change a result.
    #[test]
    fn adapter_reproduces_the_reference_solver_under_the_axis_remap() {
        let grid = Grid::new(16, 8, 8, 1e-3, 1e-3, 1e-3).expect("grid");
        let medium = absorbing_water(&grid);

        // Anisotropic in (y, z) and non-uniform along x so that the source
        // extraction's sum over the therapy axial axis is exercised.
        let mut fields = Array4::<f64>::zeros((1, grid.nx, grid.ny, grid.nz));
        let (cy, cz) = (grid.ny as f64 / 2.0, grid.nz as f64 / 2.0);
        let (sy, sz) = (grid.ny as f64 / 3.0, grid.nz as f64 / 6.0);
        for ix in 0..grid.nx {
            for iy in 0..grid.ny {
                for iz in 0..grid.nz {
                    let ry = (iy as f64 - cy) / sy;
                    let rz = (iz as f64 - cz) / sz;
                    fields[[0, ix, iy, iz]] = (1.0 + ix as f64) * (-(ry * ry + rz * rz)).exp();
                }
            }
        }

        let config = KzkPlugin::build_config(&grid, &medium);
        let source = KzkPlugin::extract_source(&fields, &grid);
        let mut reference = KZKSolver::new(config.clone()).expect("reference solver");
        reference.set_source(source, config.frequency);

        let written = sweep_all_planes(&grid, &medium, &fields);

        for x in 0..grid.nx {
            reference.step();
            let plane = reference.current_field();
            for iy in 0..grid.ny {
                for iz in 0..grid.nz {
                    let actual = written[[0, x, iy, iz]];
                    let expected = plane[[iy, iz]];
                    assert!(
                        actual.to_bits() == expected.to_bits(),
                        "adapter mismatch at therapy (x = {x}, y = {iy}, z = {iz}) \
                         ↔ KZK (i = {iy}, j = {iz}, step = {x}): plugin wrote \
                         {actual:.17e} Pa, reference solver gives {expected:.17e} Pa"
                    );
                }
            }
        }

        // The comparison is only meaningful if the reference field is alive.
        let peak = reference
            .current_field()
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        assert!(
            peak > 0.0,
            "reference solver must carry non-zero amplitude at the final plane, got {peak} Pa"
        );
    }
}
