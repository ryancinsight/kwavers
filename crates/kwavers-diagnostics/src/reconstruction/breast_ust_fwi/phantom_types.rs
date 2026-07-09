//! Shared phantom-domain types for Ali et al. breast UST FWI ingest.

use leto::Array3;
use std::path::PathBuf;

/// Unit of stored sound-speed samples.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreastUstSoundSpeedUnit {
    /// Dataset values are already in meters per second.
    MetersPerSecond,
    /// Dataset values are in kilometers per second.
    KilometersPerSecond,
}

impl BreastUstSoundSpeedUnit {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::MetersPerSecond => "meters_per_second",
            Self::KilometersPerSecond => "kilometers_per_second",
        }
    }

    pub(super) const fn scale_to_meters_per_second(self) -> f64 {
        match self {
            Self::MetersPerSecond => 1.0,
            Self::KilometersPerSecond => 1000.0,
        }
    }
}

/// Linearization order used by an external tensor payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreastUstPhantomStorageOrder {
    /// C-contiguous order: `k` is the fastest-varying index.
    CContiguous,
    /// MATLAB/Fortran-contiguous order: `i` is the fastest-varying index.
    FortranContiguous,
}

impl BreastUstPhantomStorageOrder {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::CContiguous => "c_contiguous",
            Self::FortranContiguous => "fortran_contiguous",
        }
    }
}

/// Clinical sound-speed phantom decoded from an external source.
#[derive(Clone, Debug)]
pub struct BreastUstAliPhantom {
    /// Sound-speed volume [m/s].
    pub sound_speed_m_s: Array3<f64>,
    /// Uniform voxel spacing [m].
    pub spacing_m: f64,
    /// Resolved dataset or variable path.
    pub dataset_path: String,
    /// Filesystem path of the loaded source.
    pub source_path: PathBuf,
    /// Storage order used to interpret the raw payload.
    pub storage_order: BreastUstPhantomStorageOrder,
    /// Clinical phantom model identifier.
    pub model_family: &'static str,
}

impl BreastUstAliPhantom {
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        self.sound_speed_m_s.dim()
    }

    #[must_use]
    pub fn physical_extent_m(&self) -> [f64; 3] {
        let (nx, ny, nz) = self.sound_speed_m_s.dim();
        [
            nx as f64 * self.spacing_m,
            ny as f64 * self.spacing_m,
            nz as f64 * self.spacing_m,
        ]
    }
}
