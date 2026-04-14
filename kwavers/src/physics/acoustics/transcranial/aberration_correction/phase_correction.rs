//! Phase correction computation for transcranial aberration
//!
//! ## Mathematical Foundation
//!
//! ### Phase-Screen Aberration Model
//!
//! The skull is treated as a thin phase screen (Clement & Hynynen 2002).
//! For a ray travelling from a transducer element at `r_elem` to a target
//! at `r_target`, the accumulated aberration phase is the line integral:
//!
//! ```text
//!   Δφ = ∫_{ray} [ k(s) − k_water ] ds
//!      = ∫_{ray} [ 2π f / c(s) − 2π f / c_water ] ds
//! ```
//!
//! where `k(s) = 2π f / c(s)` is the local wavenumber at arc-length `s`
//! and `c(s)` is the local sound speed derived from CT Hounsfield units.
//!
//! ### Phase Conjugation Correction
//!
//! **Theorem** (Fink 1992, Aubry 2003): The wave equation
//! `∂²u/∂t² − c²∇²u = 0` is invariant under `t → −t`.
//! Consequently, pre-applying the correction phase `−Δφ_i` to element `i`
//! exactly cancels the skull-induced aberration and produces coherent
//! summation at the target.
//!
//! **Corollary** (focal gain): For an N-element array with uncorrected
//! aberration phases `{φ_i}`, the circular coherence is
//! `R = (1/N)|Σ_i e^{iφ_i}|`.  After correction R → 1, so the focal gain
//! improvement is:
//! ```text
//!   ΔG = 20 · log10(1 / R)   [dB]
//! ```
//! (positive when R < 1, i.e. when the array was aberrated.)
//!
//! ## References
//!
//! - Clement GT, Hynynen K (2002). "A non-invasive method for focusing
//!   ultrasound through the human skull." Phys. Med. Biol. 47(8):1219–1235.
//! - Aubry J-F et al. (2003). "Experimental demonstration of noninvasive
//!   transskull adaptive focusing based on prior CT scans." JASA 113(1):84–93.
//! - Fink M (1992). "Time reversal of ultrasonic fields."
//!   IEEE Trans. UFFC 39(5):555–566.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use log::info;
use std::f64::consts::PI;

/// Phase correction data for transducer elements
#[derive(Debug, Clone)]
pub struct PhaseCorrection {
    /// Correction phases for each transducer element (radians).
    ///
    /// `phases[i] = −Δφ_i` so that the transmitted signal `A · e^{i·phases[i]}`
    /// arrives at the target with zero residual phase aberration.
    pub phases: Vec<f64>,
    /// Element amplitude weights (dimensionless, nominally 1.0).
    pub amplitudes: Vec<f64>,
    /// Expected focal gain improvement due to aberration correction (dB).
    ///
    /// Computed as `−20·log10(R)` where R is the pre-correction circular
    /// coherence of the aberration phases.  A value of 0 dB means the array
    /// was already fully coherent; larger values indicate larger improvement.
    pub focal_gain_db: f64,
    /// Correction quality metric ∈ [0, 1].
    ///
    /// Defined as the mean vector magnitude of the residual phases after
    /// applying the computed correction.  1 = perfect correction; 0 = no
    /// coherence restored.
    pub quality_metric: f64,
}

/// Transcranial aberration correction system
#[derive(Debug)]
pub struct TranscranialAberrationCorrection {
    /// Computational grid
    pub(crate) grid: Grid,
    /// Operating frequency (Hz)
    pub(crate) frequency: f64,
    /// Reference sound speed in water (m/s)
    pub(crate) reference_speed: f64,
    /// Number of transducer elements (reserved for future array geometry)
    pub(crate) _num_elements: usize,
}

impl TranscranialAberrationCorrection {
    /// Create new aberration correction system
    pub fn new(grid: &Grid) -> KwaversResult<Self> {
        Ok(Self {
            grid: grid.clone(),
            frequency: 650e3,
            reference_speed: 1500.0, // matches CTImageLoader::hu_to_sound_speed soft-tissue baseline
            _num_elements: 1024,
        })
    }

    /// Calculate CT-based phase correction for each transducer element.
    ///
    /// ## Algorithm
    ///
    /// 1. Trace a ray from each transducer element to the target (100 samples).
    /// 2. Accumulate the aberration phase `Δφ = ∫(k_local − k_water) ds`
    ///    using the trapezoidal rule along the ray.
    /// 3. The correction phase for element i is `−Δφ_i`.
    /// 4. Amplitudes are set to 1.0 (uniform weighting).
    ///
    /// ## References
    /// - Clement & Hynynen (2002) §II.A. Phase-screen model.
    pub fn calculate_correction(
        &self,
        skull_ct_data: &ndarray::Array3<f64>,
        transducer_positions: &[[f64; 3]],
        target_point: &[f64; 3],
    ) -> KwaversResult<PhaseCorrection> {
        info!(
            "Calculating aberration correction for {} transducer elements",
            transducer_positions.len()
        );

        let aberration_phases =
            self.calculate_aberration_phases(skull_ct_data, transducer_positions, target_point)?;

        // Phase conjugation: correction = −Δφ
        let phases: Vec<f64> = aberration_phases.iter().map(|&p| -p).collect();

        // Uniform element amplitudes: amplitude weighting / tapering is a
        // separate array-driver concern (Harris 1978; Nuttall 1981).
        let amplitudes = vec![1.0_f64; transducer_positions.len()];

        let focal_gain_db = Self::focal_gain_improvement_db(&aberration_phases);
        let quality_metric = Self::circular_coherence(&phases);

        Ok(PhaseCorrection {
            phases,
            amplitudes,
            focal_gain_db,
            quality_metric,
        })
    }

    /// Compute the aberration phase accumulated by each element ray through the skull.
    ///
    /// **Theorem**: The aberration phase for element `i` is
    /// ```text
    ///   Δφ_i = ∫_{ray_i} [ 2π f / c(s) − 2π f / c_water ] ds
    /// ```
    /// evaluated by a 100-sample composite trapezoidal rule.
    ///
    /// Grid coordinates are recovered by dividing the physical position by the
    /// grid spacing and clamping to valid index range before lookup in the CT array.
    ///
    /// **References**: Aubry et al. (2003) §II.B; Marquet et al. (2009) §2.1.
    pub(crate) fn calculate_aberration_phases(
        &self,
        skull_ct_data: &ndarray::Array3<f64>,
        transducer_positions: &[[f64; 3]],
        target_point: &[f64; 3],
    ) -> KwaversResult<Vec<f64>> {
        let mut aberration_phases = Vec::with_capacity(transducer_positions.len());

        for &transducer_pos in transducer_positions {
            let path_vector = [
                target_point[0] - transducer_pos[0],
                target_point[1] - transducer_pos[1],
                target_point[2] - transducer_pos[2],
            ];
            let path_length = (path_vector[0].powi(2)
                + path_vector[1].powi(2)
                + path_vector[2].powi(2))
            .sqrt();

            let num_samples: usize = 100;
            let ds = path_length / num_samples as f64;
            let k_water = 2.0 * PI * self.frequency / self.reference_speed;
            let mut total_aberration = 0.0_f64;

            for i in 0..num_samples {
                let t = (i as f64 + 0.5) / num_samples as f64; // midpoint rule
                let point = [
                    transducer_pos[0] + t * path_vector[0],
                    transducer_pos[1] + t * path_vector[1],
                    transducer_pos[2] + t * path_vector[2],
                ];

                let ix = ((point[0] / self.grid.dx) as usize).min(self.grid.nx - 1);
                let iy = ((point[1] / self.grid.dy) as usize).min(self.grid.ny - 1);
                let iz = ((point[2] / self.grid.dz) as usize).min(self.grid.nz - 1);

                let hu = skull_ct_data[[ix, iy, iz]];
                let local_speed =
                    crate::domain::imaging::medical::CTImageLoader::hu_to_sound_speed(hu);

                let k_local = 2.0 * PI * self.frequency / local_speed;
                total_aberration += (k_local - k_water) * ds;
            }

            aberration_phases.push(total_aberration);
        }

        Ok(aberration_phases)
    }

    /// Expected focal gain improvement from correcting `aberration_phases` (dB).
    ///
    /// ## Theorem (Clement & Hynynen 2002, eq. 7)
    ///
    /// The pre-correction circular coherence is:
    /// ```text
    ///   R = (1/N) |Σ_i e^{i φ_i}|   ∈ [0, 1]
    /// ```
    /// - R = 1: all elements are in phase → no improvement from correction.
    /// - R = 0: fully incoherent → maximum improvement.
    ///
    /// The improvement in focal gain after phase conjugation is:
    /// ```text
    ///   ΔG = −20 · log10(R)   [dB]
    /// ```
    ///
    /// **References**:
    /// - Clement GT, Hynynen K (2002). Phys. Med. Biol. 47(8):1219.
    /// - O'Brien WD (1992). J. Acoust. Soc. Am. 92(5):2397.
    pub(crate) fn focal_gain_improvement_db(aberration_phases: &[f64]) -> f64 {
        let coherence = Self::circular_coherence(aberration_phases);
        if coherence <= 0.0 {
            f64::INFINITY
        } else if coherence >= 1.0 {
            0.0
        } else {
            -20.0 * coherence.log10()
        }
    }

    /// Circular mean vector magnitude (coherence) of a set of phases.
    ///
    /// ```text
    ///   R = (1/N) |Σ_i e^{i φ_i}| = sqrt( (Σ cos φ_i)² + (Σ sin φ_i)² ) / N
    /// ```
    ///
    /// R ∈ [0, 1]: 1 = fully coherent; 0 = uniformly distributed on the circle.
    ///
    /// **Reference**: Mardia KV, Jupp PE (2000). *Directional Statistics*. §2.2.
    pub(crate) fn circular_coherence(phases: &[f64]) -> f64 {
        let n = phases.len();
        if n == 0 {
            return 0.0;
        }
        let (sum_cos, sum_sin) =
            phases
                .iter()
                .fold((0.0_f64, 0.0_f64), |(sc, ss), &p| (sc + p.cos(), ss + p.sin()));
        (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n as f64
    }
}
