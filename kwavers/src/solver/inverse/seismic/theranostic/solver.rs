//! Same-device exposure synthesis, FWI, RTM, and harmonic reconstruction.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;

use super::config::{TheranosticFwiConfig, C_REF_M_S};
use super::geometry::{angle_span, build_device_layout, DeviceLayout};
use super::medium::{target_contrast, PreparedTheranosticSlice};
use super::metrics::{metrics_for, ReconstructionMetrics};
use super::operator::{
    active_grid, build_fundamental_matrix, build_harmonic_matrix, build_passive_matrix,
    build_ultraharmonic_matrix, exposure_map, image_from_vector, normalize_positive,
    vector_from_image, ActiveGrid, RowMatrix,
};

#[derive(Clone, Debug)]
pub struct TheranosticFwiResult {
    pub prepared: PreparedTheranosticSlice,
    pub layout: DeviceLayout,
    pub exposure: Array2<f64>,
    pub lesion_target: Array2<f64>,
    pub anatomy_reconstruction: Array2<f64>,
    pub active_lesion_reconstruction: Array2<f64>,
    pub subharmonic_reconstruction: Array2<f64>,
    pub harmonic_reconstruction: Array2<f64>,
    pub ultraharmonic_reconstruction: Array2<f64>,
    pub fused_reconstruction: Array2<f64>,
    pub anatomy_metrics: ReconstructionMetrics,
    pub active_metrics: ReconstructionMetrics,
    pub subharmonic_metrics: ReconstructionMetrics,
    pub harmonic_metrics: ReconstructionMetrics,
    pub ultraharmonic_metrics: ReconstructionMetrics,
    pub fused_metrics: ReconstructionMetrics,
    pub objective_history: Vec<f64>,
    pub measurements: usize,
    pub active_voxels: usize,
}

pub fn run_theranostic_fwi(
    prepared: PreparedTheranosticSlice,
    config: &TheranosticFwiConfig,
) -> KwaversResult<TheranosticFwiResult> {
    config.validate()?;
    let layout = build_device_layout(
        config,
        &prepared.body_mask,
        &prepared.target_mask,
        prepared.spacing_m,
    );
    let active_mask = &prepared.body_mask;
    let active = active_grid(active_mask, prepared.spacing_m);
    if active.indices.len() < 16 {
        return Err(KwaversError::InvalidInput(
            "theranostic active support has fewer than 16 voxels".to_owned(),
        ));
    }
    let fundamental = build_fundamental_matrix(&prepared, &layout, &active, config);
    let harmonic = build_harmonic_matrix(&prepared, &layout, &active, config);
    let ultraharmonic = build_ultraharmonic_matrix(&prepared, &layout, &active, config);
    let passive = build_passive_matrix(&prepared, &layout, &active, config);
    let exposure = exposure_map(&prepared, &layout, config);
    let lesion_target = lesion_source(&prepared, &exposure);

    let anatomy_target = target_contrast(&prepared);
    let anatomy_vec = vector_from_image(&anatomy_target, &active);
    let (anatomy_recon_vec, mut history) = solve_inverse(
        &fundamental,
        &anatomy_vec,
        &active,
        active_mask.dim(),
        config,
    );
    let anatomy_reconstruction = image_from_vector(&anatomy_recon_vec, &active, active_mask.dim());

    let mut lesion_speed = lesion_target.clone();
    lesion_speed.mapv_inplace(|v| v * config.lesion_delta_c_m_s / C_REF_M_S);
    let lesion_speed_vec = vector_from_image(&lesion_speed, &active);
    let (active_vec, active_history) = solve_inverse(
        &fundamental,
        &lesion_speed_vec,
        &active,
        active_mask.dim(),
        config,
    );
    history.extend(active_history);
    let active_lesion_reconstruction = normalize_positive(
        &image_from_vector(&negated(&active_vec), &active, active_mask.dim()),
        active_mask,
    );

    let sub_target_vec = vector_from_image(&lesion_target, &active);
    let (sub_vec, sub_history) = solve_inverse(
        &passive,
        &sub_target_vec,
        &active,
        active_mask.dim(),
        config,
    );
    history.extend(sub_history);
    let subharmonic_reconstruction = normalize_positive(
        &image_from_vector(&sub_vec, &active, active_mask.dim()),
        active_mask,
    );

    let harmonic_target = harmonic_target(&prepared, &lesion_target);
    let harmonic_vec = vector_from_image(&harmonic_target, &active);
    let (harmonic_out, harmonic_history) =
        solve_inverse(&harmonic, &harmonic_vec, &active, active_mask.dim(), config);
    history.extend(harmonic_history);
    let harmonic_reconstruction = normalize_positive(
        &image_from_vector(&harmonic_out, &active, active_mask.dim()),
        active_mask,
    );

    let ultraharmonic_target = ultraharmonic_target(&prepared, &lesion_target);
    let ultraharmonic_vec = vector_from_image(&ultraharmonic_target, &active);
    let (ultra_out, ultra_history) = solve_inverse(
        &ultraharmonic,
        &ultraharmonic_vec,
        &active,
        active_mask.dim(),
        config,
    );
    history.extend(ultra_history);
    let ultraharmonic_reconstruction = normalize_positive(
        &image_from_vector(&ultra_out, &active, active_mask.dim()),
        active_mask,
    );

    let fused_reconstruction = fuse_maps(
        &active_lesion_reconstruction,
        &subharmonic_reconstruction,
        &harmonic_reconstruction,
        &ultraharmonic_reconstruction,
        active_mask,
    );
    let anatomy_metrics = metrics_for(&anatomy_target, &anatomy_reconstruction, active_mask);
    let active_metrics = metrics_for(&lesion_target, &active_lesion_reconstruction, active_mask);
    let subharmonic_metrics = metrics_for(&lesion_target, &subharmonic_reconstruction, active_mask);
    let harmonic_metrics = metrics_for(&harmonic_target, &harmonic_reconstruction, active_mask);
    let ultraharmonic_metrics = metrics_for(
        &ultraharmonic_target,
        &ultraharmonic_reconstruction,
        active_mask,
    );
    let fused_metrics = metrics_for(&lesion_target, &fused_reconstruction, active_mask);

    let _span = angle_span(&layout);
    Ok(TheranosticFwiResult {
        prepared,
        layout,
        exposure,
        lesion_target,
        anatomy_reconstruction,
        active_lesion_reconstruction,
        subharmonic_reconstruction,
        harmonic_reconstruction,
        ultraharmonic_reconstruction,
        fused_reconstruction,
        anatomy_metrics,
        active_metrics,
        subharmonic_metrics,
        harmonic_metrics,
        ultraharmonic_metrics,
        fused_metrics,
        objective_history: history,
        measurements: fundamental.rows + passive.rows + harmonic.rows + ultraharmonic.rows,
        active_voxels: active.indices.len(),
    })
}

fn solve_inverse(
    matrix: &RowMatrix,
    target: &[f32],
    active: &ActiveGrid,
    shape: (usize, usize),
    config: &TheranosticFwiConfig,
) -> (Vec<f32>, Vec<f64>) {
    let mut data = vec![0.0; matrix.rows];
    matrix.matvec(target, &mut data);
    add_deterministic_noise(&mut data, config.noise_fraction);
    let mut rhs = vec![0.0; matrix.cols];
    matrix.t_matvec(&data, &mut rhs);
    let mut diag = matrix.normal_diag();
    for value in &mut diag {
        *value += config.regularization as f32 + 1.0e-6;
    }
    let mut x = vec![0.0; matrix.cols];
    let mut hx = vec![0.0; matrix.cols];
    normal_apply(matrix, &x, active, shape, config, &mut hx);
    let mut r = rhs
        .iter()
        .zip(hx.iter())
        .map(|(b, h)| b - h)
        .collect::<Vec<_>>();
    let mut z = r
        .iter()
        .zip(diag.iter())
        .map(|(rv, dv)| rv / dv)
        .collect::<Vec<_>>();
    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);
    let mut history = vec![objective(matrix, &x, &data, active, shape, config)];
    for _ in 0..config.iterations {
        let mut ap = vec![0.0; matrix.cols];
        normal_apply(matrix, &p, active, shape, config, &mut ap);
        let denom = dot(&p, &ap);
        if denom <= 0.0 {
            break;
        }
        let alpha = rz_old / denom;
        axpy(alpha, &p, &mut x);
        axpy(-alpha, &ap, &mut r);
        z.iter_mut()
            .zip(r.iter().zip(diag.iter()))
            .for_each(|(zv, (rv, dv))| *zv = rv / dv);
        let rz_new = dot(&r, &z);
        history.push(objective(matrix, &x, &data, active, shape, config));
        if rz_new <= 1.0e-16 * rz_old.max(1.0) {
            break;
        }
        let beta = rz_new / rz_old;
        for (pv, zv) in p.iter_mut().zip(z.iter()) {
            *pv = *zv + beta * *pv;
        }
        rz_old = rz_new;
    }
    (x, history)
}

fn normal_apply(
    matrix: &RowMatrix,
    x: &[f32],
    active: &ActiveGrid,
    shape: (usize, usize),
    config: &TheranosticFwiConfig,
    out: &mut [f32],
) {
    let mut tmp = vec![0.0; matrix.rows];
    matrix.matvec(x, &mut tmp);
    matrix.t_matvec(&tmp, out);
    let lap = laplacian(active, shape, x);
    for ((dst, xv), lv) in out.iter_mut().zip(x.iter()).zip(lap.iter()) {
        *dst += config.regularization as f32 * *xv + config.smoothness_weight as f32 * *lv;
    }
}

fn objective(
    matrix: &RowMatrix,
    x: &[f32],
    data: &[f32],
    active: &ActiveGrid,
    shape: (usize, usize),
    config: &TheranosticFwiConfig,
) -> f64 {
    let mut prediction = vec![0.0; matrix.rows];
    matrix.matvec(x, &mut prediction);
    let residual = prediction
        .iter()
        .zip(data.iter())
        .map(|(p, d)| (*p - *d).powi(2) as f64)
        .sum::<f64>();
    let norm = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    let lap = laplacian(active, shape, x);
    let smooth = x
        .iter()
        .zip(lap.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum::<f64>();
    0.5 * residual + 0.5 * config.regularization * norm + 0.5 * config.smoothness_weight * smooth
}

fn laplacian(active: &ActiveGrid, shape: (usize, usize), values: &[f32]) -> Vec<f32> {
    let mut image = Array2::<f32>::zeros(shape);
    let mut active_image = Array2::<bool>::from_elem(shape, false);
    for ((ix, iy), value) in active.indices.iter().zip(values.iter()) {
        image[[*ix, *iy]] = *value;
        active_image[[*ix, *iy]] = true;
    }
    let mut out = Vec::with_capacity(active.indices.len());
    for (ix, iy) in &active.indices {
        let mut degree = 0.0;
        let mut sum = 0.0;
        for (nx, ny) in neighbors(*ix, *iy, shape) {
            if active_image[[nx, ny]] {
                degree += 1.0;
                sum += image[[nx, ny]];
            }
        }
        out.push(degree * image[[*ix, *iy]] - sum);
    }
    out
}

fn neighbors(ix: usize, iy: usize, shape: (usize, usize)) -> impl Iterator<Item = (usize, usize)> {
    let mut out = Vec::with_capacity(4);
    if ix > 0 {
        out.push((ix - 1, iy));
    }
    if iy > 0 {
        out.push((ix, iy - 1));
    }
    if ix + 1 < shape.0 {
        out.push((ix + 1, iy));
    }
    if iy + 1 < shape.1 {
        out.push((ix, iy + 1));
    }
    out.into_iter()
}

fn lesion_source(prepared: &PreparedTheranosticSlice, exposure: &Array2<f64>) -> Array2<f64> {
    let target_peak = exposure
        .iter()
        .zip(prepared.target_mask.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .fold(0.0, f64::max)
        .max(1.0e-12);
    Array2::from_shape_fn(exposure.dim(), |idx| {
        if prepared.target_mask[idx] {
            (exposure[idx] / target_peak).clamp(0.0, 1.0)
        } else {
            0.0
        }
    })
}

fn harmonic_target(prepared: &PreparedTheranosticSlice, lesion: &Array2<f64>) -> Array2<f64> {
    let median = prepared
        .sound_speed_m_s
        .iter()
        .zip(prepared.body_mask.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .sum::<f64>()
        / prepared.body_mask.iter().filter(|v| **v).count().max(1) as f64;
    Array2::from_shape_fn(lesion.dim(), |idx| {
        let contrast = ((prepared.sound_speed_m_s[idx] - median).abs() / 120.0).clamp(0.0, 1.0);
        lesion[idx] * (0.8 + 0.2 * contrast)
    })
}

fn ultraharmonic_target(prepared: &PreparedTheranosticSlice, lesion: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(lesion.dim(), |idx| {
        let attenuation = prepared.attenuation_np_per_m_mhz[idx];
        lesion[idx] * (0.7 + 0.3 * (attenuation / 18.0).clamp(0.0, 1.0))
    })
}

fn fuse_maps(
    a: &Array2<f64>,
    s: &Array2<f64>,
    h: &Array2<f64>,
    u: &Array2<f64>,
    mask: &Array2<bool>,
) -> Array2<f64> {
    let fused = Array2::from_shape_fn(a.dim(), |idx| {
        if mask[idx] {
            (0.40 * a[idx] + 0.25 * s[idx] + 0.20 * h[idx] + 0.15 * u[idx])
                * (0.25 + 0.75 * s[idx].max(u[idx]))
        } else {
            0.0
        }
    });
    normalize_positive(&fused, mask)
}

fn add_deterministic_noise(data: &mut [f32], fraction: f64) {
    if fraction <= 0.0 || data.is_empty() {
        return;
    }
    let rms = (data.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
    let sigma = (fraction * rms) as f32;
    for (idx, value) in data.iter_mut().enumerate() {
        *value += sigma * ((idx as f32 * 12.9898).sin() + 0.5 * (idx as f32 * 78.233).cos());
    }
}

fn negated(values: &[f32]) -> Vec<f32> {
    values.iter().map(|v| -*v).collect()
}

fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (yv, xv) in y.iter_mut().zip(x.iter()) {
        *yv += alpha * *xv;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
