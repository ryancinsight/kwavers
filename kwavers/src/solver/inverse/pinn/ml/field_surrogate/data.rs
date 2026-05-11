//! Sampling helpers that turn cached PSTD kernels into Burn
//! `TrainingBatch`es.
//!
//! The trainer ([`super::training::ParamFieldPINNTrainer`]) is data-
//! source agnostic — it only needs a `make_batch(step)` closure
//! producing tensors of the right shape. This module provides a
//! [`KernelCubeSampler`] that draws random voxels from a list of
//! [`FocalKernel`]s, applies the same `[-1, 1]` input/output
//! normalisation the network expects, and emits a [`TrainingBatch`]
//! ready for `step()`.
//!
//! Per-channel target scales (`p_min_scale_pa`, `p_max_scale_pa`,
//! `p_rms_scale_pa`) are computed once from the cube's max
//! envelope amplitudes and reused across all batches so the
//! normalisation is consistent.

use burn::tensor::{backend::AutodiffBackend, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::physics::field_surrogate::FocalKernel;

use super::training::TrainingBatch;

/// Per-channel scale factors used to map physical Pa to the
/// network's `[-1, 1]` output space.
#[derive(Debug, Clone, Copy)]
pub struct OutputScales {
    pub p_min_pa: f32,
    pub p_max_pa: f32,
    pub p_rms_pa: f32,
}

/// Per-axis spatial half-extents (m) used to map physical
/// coordinates to the network's `[-1, 1]` input space.
#[derive(Debug, Clone, Copy)]
pub struct CoordHalves {
    pub hx_m: f32,
    pub hy_m: f32,
    pub hz_m: f32,
}

/// `(f0_min, f0_max)` and `(pnp_min, pnp_max)` for input-side
/// parameter normalisation.
#[derive(Debug, Clone, Copy)]
pub struct ParamRanges {
    pub f0_hz: (f32, f32),
    pub pnp_pa: (f32, f32),
}

/// How [`KernelCubeSampler::batch`] selects voxels.
///
/// `Uniform` — every voxel equally likely. Simple but biases the
/// network toward predicting near-zero everywhere because most
/// voxels are far from focus and have `|p| ≈ 0`.
///
/// `ImportanceByMagnitude { exponent }` — voxel probability ∝
/// `|p|^exponent`. With `exponent = 1.0` the network sees each
/// voxel weighted by its envelope magnitude (so the focal peak is
/// sampled ~the same total amount as the far rim); `2.0` further
/// concentrates samples near the peak. Closes the focal-peak
/// underprediction observed under uniform sampling.
#[derive(Debug, Clone, Copy)]
pub enum SamplingMode {
    Uniform,
    ImportanceByMagnitude { exponent: f32 },
}

impl Default for SamplingMode {
    fn default() -> Self {
        SamplingMode::Uniform
    }
}

/// A precomputed flat dataset built from a stack of
/// [`FocalKernel`]s. Holds the normalised inputs and targets in
/// host memory so each batch is a cheap random gather.
#[derive(Debug)]
pub struct KernelCubeSampler {
    /// Normalised inputs `[N, 5]` flat row-major.
    inputs: Vec<f32>,
    /// Normalised targets `[N, 3]` flat row-major.
    targets: Vec<f32>,
    /// Per-sample physical `f0` (Hz) — used by the Helmholtz
    /// residual to compute per-sample `k`.
    f0_phys_hz: Vec<f32>,
    /// Per-sample `|p_max_norm|` magnitude (used for importance
    /// sampling). Same length as `f0_phys_hz`.
    p_magnitude: Vec<f32>,
    /// Number of samples = total interior voxel count across all
    /// kernels.
    n: usize,
    pub coord_halves: CoordHalves,
    pub output_scales: OutputScales,
    pub param_ranges: ParamRanges,
    /// Active sampling strategy. Mutable so callers can switch
    /// strategies between training phases (e.g., uniform warm-up,
    /// then importance-sampled fine-tune).
    pub sampling: SamplingMode,
    /// Cached cumulative-weight table for importance sampling — only
    /// populated when `sampling = ImportanceByMagnitude`.
    cumulative_weights: Vec<f32>,
}

impl KernelCubeSampler {
    /// Build a flat dataset from a stack of kernels.
    ///
    /// Each kernel's interior voxels (one-cell shell stripped) are
    /// flattened into rows of (input, target). Spatial coordinates
    /// are normalised by the kernel's own half-extent (focus voxel
    /// at the centre, edges at ±1). `f0` and `pnp` are normalised
    /// by the global min/max across the stack.
    ///
    /// `coord_halves` is the planner-side spatial half-extent used
    /// at inference time — it must match the planner's expected
    /// input range. Pass `None` to derive it from the kernels'
    /// largest spatial extent.
    pub fn new(
        kernels: &[FocalKernel],
        coord_halves_override: Option<CoordHalves>,
    ) -> Self {
        let (f0_min, f0_max) = kernels.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(lo, hi), k| (lo.min(k.f0 as f32), hi.max(k.f0 as f32)),
        );
        let (pnp_min, pnp_max) = kernels.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(lo, hi), k| (lo.min(k.pnp_realised as f32), hi.max(k.pnp_realised as f32)),
        );
        let f0_range = if (f0_max - f0_min).abs() < 1.0 {
            // Single-frequency cube — clamp to a 1-Hz range so
            // normalisation stays stable.
            (f0_min, f0_min + 1.0)
        } else {
            (f0_min, f0_max)
        };
        let pnp_range = if (pnp_max - pnp_min).abs() < 1.0 {
            (pnp_min, pnp_min + 1.0)
        } else {
            (pnp_min, pnp_max)
        };

        // Output scales: per-channel max absolute values across all
        // kernels. The kernel field stores p_neg (positive Pa for
        // peak rarefactional); take that as p_min_scale magnitude
        // and reuse it for p_max + p_rms since the cube doesn't
        // currently store separate channels — the trainer feeds
        // p_neg into all three target slots and the network learns
        // the channel-specific shapes from supervised data later.
        let p_max_pa = kernels.iter().fold(0.0_f64, |acc, k| {
            acc.max(k.field.iter().fold(0.0_f64, |a, &v| a.max(v.abs())))
        }) as f32;
        let output_scales = OutputScales {
            p_min_pa: p_max_pa,
            p_max_pa,
            p_rms_pa: p_max_pa * 0.7, // typical p_rms ≈ p_max / sqrt(2)
        };

        // Spatial halves: largest extent across all kernels (in m).
        let coord_halves = coord_halves_override.unwrap_or_else(|| {
            let (mut hx, mut hy, mut hz) = (0.0_f32, 0.0_f32, 0.0_f32);
            for k in kernels {
                let (nx, ny, nz) = k.shape();
                let dx = k.dx_m as f32;
                hx = hx.max((nx as f32) * dx * 0.5);
                hy = hy.max((ny as f32) * dx * 0.5);
                hz = hz.max((nz as f32) * dx * 0.5);
            }
            CoordHalves { hx_m: hx, hy_m: hy, hz_m: hz }
        });

        // Flatten interior voxels of every kernel into the dataset.
        let mut inputs: Vec<f32> = Vec::new();
        let mut targets: Vec<f32> = Vec::new();
        let mut f0_phys: Vec<f32> = Vec::new();
        let mut p_magnitude: Vec<f32> = Vec::new();
        for kernel in kernels {
            let (nx, ny, nz) = kernel.shape();
            if nx < 3 || ny < 3 || nz < 3 {
                continue;
            }
            let dx = kernel.dx_m as f32;
            let (fx, fy, fz) = kernel.focus_idx;
            let f0 = kernel.f0 as f32;
            let pnp = kernel.pnp_realised as f32;
            let f0_norm =
                2.0 * (f0 - f0_range.0) / (f0_range.1 - f0_range.0) - 1.0;
            let pnp_norm =
                2.0 * (pnp - pnp_range.0) / (pnp_range.1 - pnp_range.0) - 1.0;
            for i in 1..(nx - 1) {
                for j in 1..(ny - 1) {
                    for kk in 1..(nz - 1) {
                        let x_m = ((i as f32) - (fx as f32)) * dx;
                        let y_m = ((j as f32) - (fy as f32)) * dx;
                        let z_m = ((kk as f32) - (fz as f32)) * dx;
                        let x_n = (x_m / coord_halves.hx_m).clamp(-1.0, 1.0);
                        let y_n = (y_m / coord_halves.hy_m).clamp(-1.0, 1.0);
                        let z_n = (z_m / coord_halves.hz_m).clamp(-1.0, 1.0);
                        inputs.push(x_n);
                        inputs.push(y_n);
                        inputs.push(z_n);
                        inputs.push(f0_norm);
                        inputs.push(pnp_norm);

                        // The cached PSTD kernel field stores peak
                        // rarefactional pressure as positive Pa.
                        // Map this onto all three target channels
                        // until per-channel kernels are generated.
                        let p_pa = kernel.field[[i, j, kk]] as f32;
                        let p_n = (p_pa / output_scales.p_max_pa).clamp(-1.0, 1.0);
                        targets.push(-p_n); // p_min channel: negative envelope
                        targets.push(p_n); // p_max channel: positive envelope
                        targets.push(p_n * 0.7); // p_rms ≈ |p|/sqrt(2)
                        f0_phys.push(f0);
                        p_magnitude.push(p_n.abs());
                    }
                }
            }
        }
        let n = inputs.len() / 5;

        Self {
            inputs,
            targets,
            f0_phys_hz: f0_phys,
            p_magnitude,
            n,
            coord_halves,
            output_scales,
            param_ranges: ParamRanges {
                f0_hz: f0_range,
                pnp_pa: pnp_range,
            },
            sampling: SamplingMode::Uniform,
            cumulative_weights: Vec::new(),
        }
    }

    /// Switch sampling strategy. Recomputes the cumulative-weight
    /// table when entering an `ImportanceByMagnitude` mode.
    pub fn set_sampling(&mut self, mode: SamplingMode) {
        self.sampling = mode;
        match mode {
            SamplingMode::Uniform => {
                self.cumulative_weights.clear();
            }
            SamplingMode::ImportanceByMagnitude { exponent } => {
                // Weight per voxel: |p|^exponent + epsilon, so even
                // far-rim voxels retain a small probability of being
                // sampled (the network needs some negative-class
                // training to learn the field decays to zero).
                let eps = 1.0e-4_f32;
                let mut cum = Vec::with_capacity(self.n);
                let mut acc = 0.0_f64;
                for &m in &self.p_magnitude {
                    let w = (m.max(0.0).powf(exponent) + eps) as f64;
                    acc += w;
                    cum.push(acc as f32);
                }
                self.cumulative_weights = cum;
            }
        }
    }

    /// Binary-search the cumulative-weight CDF for a random draw.
    fn sample_one(&self, u: f32) -> usize {
        match self.sampling {
            SamplingMode::Uniform => {
                // u ∈ [0, 1); map to index uniformly.
                let idx = (u * self.n as f32) as usize;
                idx.min(self.n - 1)
            }
            SamplingMode::ImportanceByMagnitude { .. } => {
                debug_assert_eq!(self.cumulative_weights.len(), self.n);
                let total = *self.cumulative_weights.last().unwrap_or(&0.0);
                let target = u * total;
                let result = self
                    .cumulative_weights
                    .binary_search_by(|w| {
                        w.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal)
                    });
                let idx = match result {
                    Ok(i) | Err(i) => i,
                };
                idx.min(self.n - 1)
            }
        }
    }

    /// Total number of voxel samples in the flat dataset.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// True if the dataset has zero samples. Trivially false after
    /// construction from any non-empty kernel.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Draw a random batch of `batch_size` samples seeded by `step`.
    ///
    /// Returns inputs `[batch, 5]`, targets `[batch, 3]`, and per-
    /// sample physical `f0` `[batch]` for the Helmholtz residual.
    pub fn batch<B: AutodiffBackend>(
        &self,
        device: &B::Device,
        step: u64,
        batch_size: usize,
    ) -> TrainingBatch<B> {
        let mut rng = StdRng::seed_from_u64(step);
        let mut input_buf = Vec::with_capacity(batch_size * 5);
        let mut target_buf = Vec::with_capacity(batch_size * 3);
        let mut f0_buf = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let u: f32 = rng.r#gen();
            let idx = self.sample_one(u);
            input_buf.extend_from_slice(&self.inputs[idx * 5..idx * 5 + 5]);
            target_buf.extend_from_slice(&self.targets[idx * 3..idx * 3 + 3]);
            f0_buf.push(self.f0_phys_hz[idx]);
        }
        let inputs = Tensor::<B, 2>::from_data(
            TensorData::new(input_buf, [batch_size, 5]),
            device,
        );
        let targets = Tensor::<B, 2>::from_data(
            TensorData::new(target_buf, [batch_size, 3]),
            device,
        );
        let f0_phys_hz = Tensor::<B, 1>::from_data(
            TensorData::new(f0_buf, [batch_size]),
            device,
        );
        TrainingBatch {
            inputs,
            targets,
            f0_phys_hz,
            coord_half_m: (
                self.coord_halves.hx_m,
                self.coord_halves.hy_m,
                self.coord_halves.hz_m,
            ),
            p_max_scale_pa: self.output_scales.p_max_pa,
        }
    }
}
