//! `Stimulus` trait — per-tile stimulation-program abstraction (Phase 5).
//!
//! The experiment orchestrator depends on [`Stimulus`] only; the concrete type
//! ([`DefaultStimulus`]) is injected at the call site. That is the DIP boundary: the
//! runner loop calls [`Stimulus::profile_for`] generically, agnostic of whether the
//! profiles come from a manifest, a test fixture, or a sweep generator.

use crate::manifest::{DriverManifest, TileStimulationProfile};

/// Per-tile stimulation-program abstraction — the orchestrator's DIP seam.
///
/// Implementors supply [`TileStimulationProfile`] slices on demand. The runner
/// does not know whether the profiles come from a [`DriverManifest`], a synthetic
/// test fixture, or a sweep generator.
pub trait Stimulus {
    /// Borrow the [`TileStimulationProfile`] for tile `tile` (0-indexed).
    /// Returns `None` when `tile >= self.tile_count()`.
    fn profile_for(&self, tile: usize) -> Option<&TileStimulationProfile>;

    /// Number of tiles this stimulus covers.
    fn tile_count(&self) -> usize;
}

/// Default implementation — borrows the manifest's `tile_profiles` slice verbatim.
/// Owns a shared reference; the manifest lives at least as long as the stimulus.
pub struct DefaultStimulus<'m> {
    manifest: &'m DriverManifest,
}

impl<'m> DefaultStimulus<'m> {
    /// Wrap `manifest` so the runner can call [`Stimulus::profile_for`] generically.
    #[must_use]
    pub fn new(manifest: &'m DriverManifest) -> Self {
        Self { manifest }
    }
}

impl<'m> Stimulus for DefaultStimulus<'m> {
    fn profile_for(&self, tile: usize) -> Option<&TileStimulationProfile> {
        self.manifest.tile_profiles.get(tile)
    }

    fn tile_count(&self) -> usize {
        self.manifest.tile_profiles.len()
    }
}
