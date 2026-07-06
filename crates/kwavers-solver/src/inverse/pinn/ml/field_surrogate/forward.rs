//! Vectorised forward inference over a planner grid.
//!
//! The treatment-planner asks for a full 3D `(p_min, p_max, p_rms)`
//! field on its target grid, parameterised by `(f0, pnp)`. This module
//! turns a [`super::ParamFieldPINNNetwork`] forward pass on a stream
//! of voxel-coordinates into three `Array3<f64>` outputs without
//! committing the planner to Burn's tensor API.

use coeus_autograd::Var;
use ndarray::Array3;

use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use kwavers_core::error::{KwaversError, KwaversResult};

use super::config::OUTPUT_DIM;
use super::network::ParamFieldPINNNetwork;
use super::target_transform::{OutputTransforms, TargetTransform};

/// Geometry + parameter inputs for [`infer_grid`].
///
/// All physical inputs are passed in raw SI units; the function
/// applies the network's input normalisation internally so callers
/// do not need to know the training-time scale factors.
#[derive(Debug, Clone)]
pub struct GridQueryParams {
    /// Output grid shape `(nx, ny, nz)`.
    pub shape: (usize, usize, usize),
    /// Index of the focal voxel within the output grid.
    pub focus_idx: (usize, usize, usize),
    /// Isotropic grid spacing [m].
    pub dx_m: f64,
    /// Source centre frequency [Hz].
    pub f0: f64,
    /// Target peak rarefactional pressure [Pa].
    pub pnp: f64,
    /// Per-axis half-extent used to normalise spatial coordinates to
    /// `[-1, 1]`. Should match the training-time bounding box; for
    /// kernel-cube training set this to the cube extent.
    pub coord_half_m: (f64, f64, f64),
    /// `(f0_min, f0_max)` for input-side `f0` normalisation.
    pub f0_range: (f64, f64),
    /// `(pnp_min, pnp_max)` for input-side `pnp` normalisation.
    pub pnp_range: (f64, f64),
    /// Per-channel `(p_min, p_max, p_rms)` inverse transforms. The
    /// network outputs `[-1, 1]`; each channel's transform inverts
    /// that back to physical Pa. For the legacy linear path this is
    /// equivalent to multiplying by a per-channel scale; for the
    /// C-8 signed-log1p path it inverts the dynamic-range compression.
    pub output_transforms: OutputTransforms,
    /// Inference batch size; trades memory against throughput. Default
    /// 65536 voxels per call (`~5 MB` of f32 input tensor on CPU).
    pub batch_size: usize,
}

impl Default for GridQueryParams {
    fn default() -> Self {
        Self {
            shape: (1, 1, 1),
            focus_idx: (0, 0, 0),
            dx_m: 1.0e-3,
            f0: MHZ_TO_HZ,
            pnp: 30.0 * MPA_TO_PA,
            coord_half_m: (50.0e-3, 50.0e-3, 50.0e-3),
            f0_range: (0.5 * MHZ_TO_HZ, MHZ_TO_HZ),
            pnp_range: (15.0 * MPA_TO_PA, 30.0 * MPA_TO_PA),
            output_transforms: OutputTransforms {
                p_min: TargetTransform::Linear {
                    scale_pa: 30.0 * MPA_TO_PA as f32,
                },
                p_max: TargetTransform::Linear {
                    scale_pa: 30.0 * MPA_TO_PA as f32,
                },
                p_rms: TargetTransform::Linear {
                    scale_pa: 21.0 * MPA_TO_PA as f32,
                },
            },
            batch_size: 65_536,
        }
    }
}

impl GridQueryParams {
    fn validate(&self) -> KwaversResult<()> {
        if self.shape.0 == 0 || self.shape.1 == 0 || self.shape.2 == 0 {
            return Err(KwaversError::InvalidInput(
                "GridQueryParams.shape components must be > 0".into(),
            ));
        }
        if self.focus_idx.0 >= self.shape.0
            || self.focus_idx.1 >= self.shape.1
            || self.focus_idx.2 >= self.shape.2
        {
            return Err(KwaversError::InvalidInput(format!(
                "focus_idx {:?} out of bounds for shape {:?}",
                self.focus_idx, self.shape
            )));
        }
        if self.dx_m <= 0.0 {
            return Err(KwaversError::InvalidInput("dx_m must be > 0".into()));
        }
        if self.coord_half_m.0 <= 0.0 || self.coord_half_m.1 <= 0.0 || self.coord_half_m.2 <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "coord_half_m components must be > 0".into(),
            ));
        }
        if self.f0_range.0 >= self.f0_range.1 || self.pnp_range.0 >= self.pnp_range.1 {
            return Err(KwaversError::InvalidInput(
                "f0_range / pnp_range must satisfy min < max".into(),
            ));
        }
        if self.batch_size == 0 {
            return Err(KwaversError::InvalidInput("batch_size must be > 0".into()));
        }
        Ok(())
    }
}

/// Map a value in `[lo, hi]` to `[-1, 1]`.
#[inline]
fn normalise_to_unit(v: f64, lo: f64, hi: f64) -> f64 {
    2.0 * (v - lo) / (hi - lo) - 1.0
}

/// Run the network over every voxel of the planner grid and return
/// `(p_min, p_max, p_rms)` Pa fields.
///
/// Voxels are streamed in batches of `params.batch_size` to keep
/// peak memory bounded; per-batch tensors are constructed on
/// `device` and the f32 outputs are denormalised to f64 Pa as they
/// are written back to the output `Array3`s.
///
/// # Errors
///
/// Returns `KwaversError::InvalidInput` when [`GridQueryParams`]
/// fails validation.
pub fn infer_grid<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
    network: &ParamFieldPINNNetwork<B>,
    params: &GridQueryParams,
) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    params.validate()?;
    let (nx, ny, nz) = params.shape;
    let (fx, fy, fz) = params.focus_idx;
    let (hx, hy, hz) = params.coord_half_m;
    let dx = params.dx_m;

    let f0_n = normalise_to_unit(params.f0, params.f0_range.0, params.f0_range.1) as f32;
    let pnp_n = normalise_to_unit(params.pnp, params.pnp_range.0, params.pnp_range.1) as f32;

    let mut p_min = Array3::<f64>::zeros((nx, ny, nz));
    let mut p_max = Array3::<f64>::zeros((nx, ny, nz));
    let mut p_rms = Array3::<f64>::zeros((nx, ny, nz));

    let inv_min = params.output_transforms.p_min;
    let inv_max = params.output_transforms.p_max;
    let inv_rms = params.output_transforms.p_rms;

    // Pre-build the (i, j, k) index sequence and stream in batches.
    let total = nx * ny * nz;
    let bs = params.batch_size.min(total).max(1);
    let mut buf_in = Vec::<f32>::with_capacity(bs * 5);
    let mut linear_idx = 0usize;
    while linear_idx < total {
        let end = (linear_idx + bs).min(total);
        let n = end - linear_idx;
        buf_in.clear();
        buf_in.reserve(n * 5);

        for off in 0..n {
            let lin = linear_idx + off;
            let k = lin % nz;
            let j = (lin / nz) % ny;
            let i = lin / (ny * nz);
            let x = ((i as f64) - (fx as f64)) * dx;
            let y = ((j as f64) - (fy as f64)) * dx;
            let z = ((k as f64) - (fz as f64)) * dx;
            buf_in.push((x / hx).clamp(-1.0, 1.0) as f32);
            buf_in.push((y / hy).clamp(-1.0, 1.0) as f32);
            buf_in.push((z / hz).clamp(-1.0, 1.0) as f32);
            buf_in.push(f0_n);
            buf_in.push(pnp_n);
        }

        let backend = B::default();
        let input = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n, 5], &buf_in, &backend),
            false,
        );
        let output = network.forward(&input);
        // Output is `[n, 3]`; already host-resident f32.
        let out_vec = output.tensor.as_slice();
        debug_assert_eq!(out_vec.len(), n * OUTPUT_DIM);

        for off in 0..n {
            let lin = linear_idx + off;
            let k = lin % nz;
            let j = (lin / nz) % ny;
            let i = lin / (ny * nz);
            let base = off * OUTPUT_DIM;
            p_min[[i, j, k]] = inv_min.inverse(out_vec[base]) as f64;
            p_max[[i, j, k]] = inv_max.inverse(out_vec[base + 1]) as f64;
            p_rms[[i, j, k]] = inv_rms.inverse(out_vec[base + 2]) as f64;
        }

        linear_idx = end;
    }

    Ok((p_min, p_max, p_rms))
}
