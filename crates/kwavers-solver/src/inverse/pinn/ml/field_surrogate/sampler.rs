//! Sampling helpers that turn cached PSTD kernels into
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

use coeus_autograd::Var;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use kwavers_core::error::KwaversResult;
use kwavers_physics::field_surrogate::FocalKernel;

use super::target_transform::OutputTransforms;
use super::training::TrainingBatch;
use super::types::{CoordHalves, OutputScales, ParamRanges, SamplingMode};

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
    /// Per-sample source-kernel index. Voxel `i` originated from
    /// `kernels[group_ids[i]]`. Consumed by the per-kernel-scoped
    /// peak-prominence loss (Phase C-10).
    group_ids: Vec<f32>,
    /// Number of distinct source kernels — propagated to the
    /// `TrainingBatch` so the trainer can size its per-group loop.
    num_groups: usize,
    /// Number of samples = total interior voxel count across all
    /// kernels.
    n: usize,
    pub coord_halves: CoordHalves,
    pub output_scales: OutputScales,
    /// Per-channel transforms used to map physical Pa into the
    /// network's `[-1, 1]` output space. Defaults to a linear
    /// transform derived from `output_scales`; pass a
    /// signed-log1p variant via [`KernelCubeSampler::with_transforms`]
    /// to balance loss across the dynamic range and close the
    /// focal-peak underprediction.
    pub output_transforms: OutputTransforms,
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
    pub fn new(kernels: &[FocalKernel], coord_halves_override: Option<CoordHalves>) -> Self {
        // Default construction uses the legacy linear transform so
        // existing tests and downstream callers keep their semantics.
        // For the C-8 signed-log1p path, use
        // [`Self::with_transforms`] and pass a precomputed
        // [`OutputTransforms`].
        Self::build(kernels, coord_halves_override, None)
            .expect("default linear transform construction cannot fail")
    }

    /// Build a sampler with an explicit per-channel target transform.
    ///
    /// The supplied `transforms` is used both at construction time
    /// (to forward-map every voxel's Pa into the network's `[-1, 1]`
    /// output space) and stored on the sampler so the inference path
    /// can apply the matching inverse.
    ///
    /// # Errors
    /// Returns [`crate::KwaversError::InvalidInput`] when the dataset is
    /// empty (no kernels with shape ≥ 3 on every axis).
    pub fn with_transforms(
        kernels: &[FocalKernel],
        coord_halves_override: Option<CoordHalves>,
        transforms: OutputTransforms,
    ) -> KwaversResult<Self> {
        Self::build(kernels, coord_halves_override, Some(transforms))
    }

    fn build(
        kernels: &[FocalKernel],
        coord_halves_override: Option<CoordHalves>,
        transforms_override: Option<OutputTransforms>,
    ) -> KwaversResult<Self> {
        let (f0_min, f0_max) = kernels
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), k| {
                (lo.min(k.f0 as f32), hi.max(k.f0 as f32))
            });
        let (pnp_min, pnp_max) = kernels
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), k| {
                (lo.min(k.pnp_realised as f32), hi.max(k.pnp_realised as f32))
            });
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

        // Per-channel transforms: default to the legacy linear divide
        // when no override is supplied; the C-8 signed-log1p path is
        // selected by `with_transforms`.
        let output_transforms = match transforms_override {
            Some(t) => t,
            None => OutputTransforms::linear(
                output_scales.p_min_pa,
                output_scales.p_max_pa,
                output_scales.p_rms_pa,
            )?,
        };

        // Spatial halves: largest extent across all kernels (in m).
        let coord_halves = coord_halves_override.unwrap_or_else(|| {
            let (mut hx, mut hy, mut hz) = (0.0_f32, 0.0_f32, 0.0_f32);
            for k in kernels {
                let [nx, ny, nz] = k.shape();
                let dx = k.dx_m as f32;
                hx = hx.max((nx as f32) * dx * 0.5);
                hy = hy.max((ny as f32) * dx * 0.5);
                hz = hz.max((nz as f32) * dx * 0.5);
            }
            CoordHalves {
                hx_m: hx,
                hy_m: hy,
                hz_m: hz,
            }
        });

        // Pre-compute total interior voxel count to avoid Vec reallocations.
        let total_voxels: usize = kernels
            .iter()
            .map(|k| {
                let [nx, ny, nz] = k.shape();
                if nx < 3 || ny < 3 || nz < 3 {
                    0
                } else {
                    (nx - 2) * (ny - 2) * (nz - 2)
                }
            })
            .sum();

        // Flatten interior voxels of every kernel into the dataset.
        let mut inputs: Vec<f32> = Vec::with_capacity(total_voxels * 5);
        let mut targets: Vec<f32> = Vec::with_capacity(total_voxels * 3);
        let mut f0_phys: Vec<f32> = Vec::with_capacity(total_voxels);
        let mut p_magnitude: Vec<f32> = Vec::with_capacity(total_voxels);
        let mut group_ids: Vec<f32> = Vec::with_capacity(total_voxels);
        let mut active_kernel_count: usize = 0;
        for kernel in kernels.iter() {
            let [nx, ny, nz] = kernel.shape();
            if nx < 3 || ny < 3 || nz < 3 {
                continue;
            }
            // Use the dense index of active kernels so downstream
            // per-group loops iterate 0..num_groups consecutively
            // without holes from skipped (too-small) kernels.
            let group_id = active_kernel_count as f32;
            active_kernel_count += 1;
            let dx = kernel.dx_m as f32;
            let (fx, fy, fz) = kernel.focus_idx;
            let f0 = kernel.f0 as f32;
            let pnp = kernel.pnp_realised as f32;
            let f0_norm = 2.0 * (f0 - f0_range.0) / (f0_range.1 - f0_range.0) - 1.0;
            let pnp_norm = 2.0 * (pnp - pnp_range.0) / (pnp_range.1 - pnp_range.0) - 1.0;
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
                        // rarefactional pressure as positive Pa. Map
                        // it onto all three target channels through
                        // the per-channel forward transform — the
                        // sign convention matches the inference path
                        // (`p_min` is the negative envelope, `p_max`
                        // the positive envelope, `p_rms ≈ |p|/√2`).
                        let p_pa = kernel.field[[i, j, kk]] as f32;
                        let p_min_n = output_transforms.p_min.forward(-p_pa);
                        let p_max_n = output_transforms.p_max.forward(p_pa);
                        let p_rms_n = output_transforms.p_rms.forward(p_pa * 0.7);
                        targets.push(p_min_n);
                        targets.push(p_max_n);
                        targets.push(p_rms_n);
                        f0_phys.push(f0);
                        group_ids.push(group_id);
                        // Importance-sampling magnitude is the
                        // *un-transformed* normalised pressure
                        // `|p|/p_max`. Computing it from the
                        // post-transform target would lift rim
                        // voxels under signed-log1p and undo the
                        // focal-peak concentration the importance
                        // CDF is built for.
                        let raw_mag = (p_pa / output_scales.p_max_pa).abs().clamp(0.0, 1.0);
                        p_magnitude.push(raw_mag);
                    }
                }
            }
        }
        let n = (inputs.len()) / 5;

        Ok(Self {
            inputs,
            targets,
            f0_phys_hz: f0_phys,
            p_magnitude,
            group_ids,
            num_groups: active_kernel_count,
            n,
            coord_halves,
            output_scales,
            output_transforms,
            param_ranges: ParamRanges {
                f0_hz: f0_range,
                pnp_pa: pnp_range,
            },
            sampling: SamplingMode::Uniform,
            cumulative_weights: Vec::new(),
        })
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
                debug_assert_eq!((self.cumulative_weights.len()), self.n);
                let total = *self.cumulative_weights.last().unwrap_or(&0.0);
                let target = u * total;
                let result = self
                    .cumulative_weights
                    .binary_search_by(|w| w.total_cmp(&target));
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
    pub fn batch<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
        &self,
        step: u64,
        batch_size: usize,
    ) -> TrainingBatch<B> {
        let mut rng = StdRng::seed_from_u64(step);
        let mut input_buf = Vec::with_capacity(batch_size * 5);
        let mut target_buf = Vec::with_capacity(batch_size * 3);
        let mut f0_buf = Vec::with_capacity(batch_size);
        let mut group_buf = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let u: f32 = rng.r#gen();
            let idx = self.sample_one(u);
            input_buf.extend_from_slice(&self.inputs[idx * 5..idx * 5 + 5]);
            target_buf.extend_from_slice(&self.targets[idx * 3..idx * 3 + 3]);
            f0_buf.push(self.f0_phys_hz[idx]);
            group_buf.push(self.group_ids[idx]);
        }
        let backend = B::default();
        let inputs = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size, 5], &input_buf, &backend),
            false,
        );
        let targets = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size, 3], &target_buf, &backend),
            false,
        );
        let f0_phys_hz = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size], &f0_buf, &backend),
            false,
        );
        let group_ids = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size], &group_buf, &backend),
            false,
        );
        TrainingBatch {
            inputs,
            targets,
            f0_phys_hz,
            group_ids,
            num_groups: self.num_groups,
            coord_half_m: (
                self.coord_halves.hx_m,
                self.coord_halves.hy_m,
                self.coord_halves.hz_m,
            ),
            p_max_scale_pa: self.output_scales.p_max_pa,
        }
    }

    /// Number of distinct source-kernel groups in the flat dataset.
    #[must_use]
    pub fn num_groups(&self) -> usize {
        self.num_groups
    }
}
