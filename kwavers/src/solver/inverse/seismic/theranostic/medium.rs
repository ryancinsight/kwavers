//! CT and segmentation preprocessing for theranostic FWI slices.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array2, Array3};
use std::collections::VecDeque;

use super::config::{AnatomyKind, C_REF_M_S};

#[derive(Clone, Debug)]
pub struct PreparedTheranosticSlice {
    pub anatomy: AnatomyKind,
    pub ct_hu: Array2<f64>,
    pub label: Array2<i16>,
    pub sound_speed_m_s: Array2<f64>,
    pub attenuation_np_per_m_mhz: Array2<f64>,
    pub body_mask: Array2<bool>,
    pub organ_mask: Array2<bool>,
    pub target_mask: Array2<bool>,
    pub spacing_m: f64,
    pub source_slice_index: usize,
}

pub fn prepare_brain_slice(
    ct_hu: Array2<f64>,
    spacing_m: f64,
    source_slice_index: usize,
) -> KwaversResult<PreparedTheranosticSlice> {
    let (nx, ny) = ct_hu.dim();
    let mut label = Array2::<i16>::zeros((nx, ny));
    let mut sound_speed = Array2::<f64>::from_elem((nx, ny), 1480.0);
    let mut attenuation = Array2::<f64>::from_elem((nx, ny), soft_attenuation());
    let mut body = Array2::<bool>::from_elem((nx, ny), false);
    let mut organ = Array2::<bool>::from_elem((nx, ny), false);
    let centroid = head_centroid(&ct_hu);
    let radius = 0.34 * nx.min(ny) as f64;
    for ix in 0..nx {
        for iy in 0..ny {
            let hu = ct_hu[[ix, iy]];
            let in_body = hu > -300.0;
            body[[ix, iy]] = in_body;
            if hu >= 300.0 {
                let phi = (hu / 1000.0).clamp(0.0, 1.0);
                sound_speed[[ix, iy]] = 1500.0 * (1.0 - phi) + 2900.0 * phi;
                attenuation[[ix, iy]] = soft_attenuation() * (1.0 - phi) + 70.0 * phi;
                label[[ix, iy]] = 4;
            } else if in_body {
                sound_speed[[ix, iy]] = brain_speed(hu);
                label[[ix, iy]] = 1;
            }
            let dx = ix as f64 - centroid.0;
            let dy = iy as f64 - centroid.1;
            let central = (dx * dx + dy * dy).sqrt() <= radius;
            organ[[ix, iy]] = central && (-40.0..=140.0).contains(&hu);
        }
    }
    let target = synthetic_deep_target(&organ, spacing_m);
    validate_masks(&body, &target)?;
    Ok(PreparedTheranosticSlice {
        anatomy: AnatomyKind::Brain,
        ct_hu,
        label,
        sound_speed_m_s: sound_speed,
        attenuation_np_per_m_mhz: attenuation,
        body_mask: body,
        organ_mask: organ,
        target_mask: target,
        spacing_m,
        source_slice_index,
    })
}

pub fn prepare_abdominal_slice(
    anatomy: AnatomyKind,
    ct_volume_hu: &Array3<f64>,
    label_volume: &Array3<i16>,
    spacing_mm: [f64; 3],
    grid_size: usize,
) -> KwaversResult<PreparedTheranosticSlice> {
    if ct_volume_hu.dim() != label_volume.dim() {
        return Err(KwaversError::InvalidInput(format!(
            "CT shape {:?} does not match segmentation shape {:?}",
            ct_volume_hu.dim(),
            label_volume.dim()
        )));
    }
    let slice_index = largest_target_slice(label_volume)?;
    let ct_slice = ct_volume_hu.slice(s![.., .., slice_index]).to_owned();
    let label_slice = label_volume.slice(s![.., .., slice_index]).to_owned();
    let target_seed = target_seed_index(&label_slice)?;
    let body_component = connected_body_component(&ct_slice, &label_slice, target_seed)?;
    let bbox = square_bbox_from_mask(&body_component, 6)?;
    let ct_crop = ct_slice
        .slice(s![bbox.0..=bbox.1, bbox.2..=bbox.3])
        .to_owned();
    let label_crop = label_slice
        .slice(s![bbox.0..=bbox.1, bbox.2..=bbox.3])
        .to_owned();
    let ct = resample_f64(&ct_crop, grid_size);
    let label = resample_labels_max(&label_crop, grid_size);
    let spacing_m = ((bbox.1 - bbox.0 + 1) as f64 * spacing_mm[0] * 1.0e-3)
        .max((bbox.3 - bbox.2 + 1) as f64 * spacing_mm[1] * 1.0e-3)
        / grid_size as f64;
    let (sound_speed, attenuation, body, organ, target) =
        abdominal_properties(anatomy, &ct, &label);
    validate_masks(&body, &target)?;
    Ok(PreparedTheranosticSlice {
        anatomy,
        ct_hu: ct,
        label,
        sound_speed_m_s: sound_speed,
        attenuation_np_per_m_mhz: attenuation,
        body_mask: body,
        organ_mask: organ,
        target_mask: target,
        spacing_m,
        source_slice_index: slice_index,
    })
}

pub fn target_contrast(prepared: &PreparedTheranosticSlice) -> Array2<f64> {
    let reference =
        median_in_mask(&prepared.sound_speed_m_s, &prepared.body_mask).unwrap_or(C_REF_M_S);
    let mut out = Array2::<f64>::zeros(prepared.sound_speed_m_s.dim());
    for ((idx, value), active) in prepared
        .sound_speed_m_s
        .indexed_iter()
        .zip(prepared.body_mask.iter())
    {
        if *active {
            out[idx] = (*value - reference) / C_REF_M_S;
        }
    }
    out
}

fn abdominal_properties(
    anatomy: AnatomyKind,
    ct: &Array2<f64>,
    label: &Array2<i16>,
) -> (
    Array2<f64>,
    Array2<f64>,
    Array2<bool>,
    Array2<bool>,
    Array2<bool>,
) {
    let (nx, ny) = ct.dim();
    let mut speed = Array2::<f64>::from_elem((nx, ny), 343.0);
    let mut attenuation = Array2::<f64>::from_elem((nx, ny), 0.05);
    let mut body = Array2::<bool>::from_elem((nx, ny), false);
    let mut organ = Array2::<bool>::from_elem((nx, ny), false);
    let mut target = Array2::<bool>::from_elem((nx, ny), false);
    let organ_speed = match anatomy {
        AnatomyKind::Liver => 1595.0,
        AnatomyKind::Kidney => 1567.0,
        AnatomyKind::Brain => 1540.0,
    };
    for ix in 0..nx {
        for iy in 0..ny {
            let hu = ct[[ix, iy]];
            let lab = label[[ix, iy]];
            body[[ix, iy]] = hu > -450.0 || lab > 0;
            organ[[ix, iy]] = lab == 1 || lab == 2;
            target[[ix, iy]] = lab == 2;
            if body[[ix, iy]] {
                speed[[ix, iy]] = 1480.0 + 0.18 * hu.clamp(-150.0, 250.0);
                attenuation[[ix, iy]] = 0.55;
            }
            if organ[[ix, iy]] {
                speed[[ix, iy]] = organ_speed + 0.10 * hu.clamp(-100.0, 200.0);
                attenuation[[ix, iy]] = 0.8;
            }
            if target[[ix, iy]] {
                speed[[ix, iy]] = organ_speed - 22.0 + 0.12 * hu.clamp(-50.0, 220.0);
                attenuation[[ix, iy]] = 1.05;
            }
            if hu > 250.0 {
                speed[[ix, iy]] = 2450.0 + 0.42 * (hu - 250.0).clamp(0.0, 1400.0);
                attenuation[[ix, iy]] = 18.0;
            }
        }
    }
    (speed, attenuation, body, organ, target)
}

pub(crate) fn largest_target_slice(label: &Array3<i16>) -> KwaversResult<usize> {
    let (_, _, nz) = label.dim();
    let mut best = None;
    for z in 0..nz {
        let count = label
            .slice(s![.., .., z])
            .iter()
            .filter(|v| **v == 2)
            .count();
        if count > best.map_or(0, |(_, c)| c) {
            best = Some((z, count));
        }
    }
    best.filter(|(_, count)| *count > 0)
        .map(|(z, _)| z)
        .ok_or_else(|| {
            KwaversError::InvalidInput("segmentation contains no label-2 target".to_owned())
        })
}

fn target_seed_index(label: &Array2<i16>) -> KwaversResult<(usize, usize)> {
    for ((ix, iy), value) in label.indexed_iter() {
        if *value == 2 {
            return Ok((ix, iy));
        }
    }
    Err(KwaversError::InvalidInput(
        "target seed is empty".to_owned(),
    ))
}

fn connected_body_component(
    ct: &Array2<f64>,
    label: &Array2<i16>,
    seed: (usize, usize),
) -> KwaversResult<Array2<bool>> {
    let (nx, ny) = ct.dim();
    let mut component = Array2::<bool>::from_elem((nx, ny), false);
    if !is_abdominal_body_candidate(ct[[seed.0, seed.1]], label[[seed.0, seed.1]]) {
        return Err(KwaversError::InvalidInput(
            "target seed is not inside abdominal body support".to_owned(),
        ));
    }
    let mut queue = VecDeque::from([seed]);
    component[[seed.0, seed.1]] = true;
    while let Some((ix, iy)) = queue.pop_front() {
        for (nx_i, ny_i) in body_neighbors(ix, iy, nx, ny) {
            if component[[nx_i, ny_i]]
                || !is_abdominal_body_candidate(ct[[nx_i, ny_i]], label[[nx_i, ny_i]])
            {
                continue;
            }
            component[[nx_i, ny_i]] = true;
            queue.push_back((nx_i, ny_i));
        }
    }
    let count = component.iter().filter(|active| **active).count();
    if count < 16 {
        return Err(KwaversError::InvalidInput(format!(
            "abdominal body component is too small: {count}"
        )));
    }
    Ok(component)
}

fn is_abdominal_body_candidate(hu: f64, label: i16) -> bool {
    hu > -450.0 || label > 0
}

fn body_neighbors(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let mut neighbors = [(ix, iy); 4];
    let mut count = 0;
    if ix > 0 {
        neighbors[count] = (ix - 1, iy);
        count += 1;
    }
    if ix + 1 < nx {
        neighbors[count] = (ix + 1, iy);
        count += 1;
    }
    if iy > 0 {
        neighbors[count] = (ix, iy - 1);
        count += 1;
    }
    if iy + 1 < ny {
        neighbors[count] = (ix, iy + 1);
        count += 1;
    }
    neighbors.into_iter().take(count)
}

fn square_bbox_from_mask(
    mask: &Array2<bool>,
    margin: usize,
) -> KwaversResult<(usize, usize, usize, usize)> {
    let (nx, ny) = mask.dim();
    let mut bbox: Option<(usize, usize, usize, usize)> = None;
    for ix in 0..nx {
        for iy in 0..ny {
            if mask[[ix, iy]] {
                bbox = Some(match bbox {
                    None => (ix, ix, iy, iy),
                    Some((x0, x1, y0, y1)) => (x0.min(ix), x1.max(ix), y0.min(iy), y1.max(iy)),
                });
            }
        }
    }
    let (x0, x1, y0, y1) =
        bbox.ok_or_else(|| KwaversError::InvalidInput("body bbox is empty".to_owned()))?;
    let x0 = x0.saturating_sub(margin);
    let x1 = (x1 + margin).min(nx - 1);
    let y0 = y0.saturating_sub(margin);
    let y1 = (y1 + margin).min(ny - 1);
    let side = (x1 - x0 + 1).max(y1 - y0 + 1).min(nx).min(ny);
    let cx2 = x0 + x1;
    let cy2 = y0 + y1;
    let mut sx = ((cx2 + 1).saturating_sub(side)) / 2;
    let mut sy = ((cy2 + 1).saturating_sub(side)) / 2;
    sx = sx.min(nx - side);
    sy = sy.min(ny - side);
    Ok((sx, sx + side - 1, sy, sy + side - 1))
}

fn resample_f64(input: &Array2<f64>, size: usize) -> Array2<f64> {
    let (nx, ny) = input.dim();
    Array2::from_shape_fn((size, size), |(ix, iy)| {
        let x = ix as f64 * (nx - 1) as f64 / (size - 1) as f64;
        let y = iy as f64 * (ny - 1) as f64 / (size - 1) as f64;
        bilinear(input, x, y)
    })
}

fn resample_labels_max(input: &Array2<i16>, size: usize) -> Array2<i16> {
    let (nx, ny) = input.dim();
    Array2::from_shape_fn((size, size), |(ix, iy)| {
        let x0 = (ix * nx) / size;
        let x1 = (((ix + 1) * nx).saturating_sub(1)) / size;
        let y0 = (iy * ny) / size;
        let y1 = (((iy + 1) * ny).saturating_sub(1)) / size;
        let mut label = 0;
        for x in x0..=x1.min(nx - 1) {
            for y in y0..=y1.min(ny - 1) {
                label = label.max(input[[x, y]]);
            }
        }
        label
    })
}

fn bilinear(input: &Array2<f64>, x: f64, y: f64) -> f64 {
    let (nx, ny) = input.dim();
    let x0 = x.floor().clamp(0.0, (nx - 1) as f64) as usize;
    let y0 = y.floor().clamp(0.0, (ny - 1) as f64) as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;
    (1.0 - tx) * (1.0 - ty) * input[[x0, y0]]
        + tx * (1.0 - ty) * input[[x1, y0]]
        + (1.0 - tx) * ty * input[[x0, y1]]
        + tx * ty * input[[x1, y1]]
}

fn synthetic_deep_target(organ: &Array2<bool>, spacing_m: f64) -> Array2<bool> {
    let (nx, ny) = organ.dim();
    let center = head_centroid_bool(organ);
    let rx = 6.0e-3 / spacing_m;
    let ry = 8.0e-3 / spacing_m;
    Array2::from_shape_fn((nx, ny), |(ix, iy)| {
        organ[[ix, iy]]
            && ((ix as f64 - center.0) / rx).powi(2) + ((iy as f64 - center.1) / ry).powi(2) <= 1.0
    })
}

fn validate_masks(body: &Array2<bool>, target: &Array2<bool>) -> KwaversResult<()> {
    let body_count = body.iter().filter(|v| **v).count();
    let target_count = target.iter().filter(|v| **v).count();
    if body_count < 16 || target_count < 4 {
        return Err(KwaversError::InvalidInput(format!(
            "insufficient active support: body={body_count}, target={target_count}"
        )));
    }
    Ok(())
}

fn median_in_mask(values: &Array2<f64>, mask: &Array2<bool>) -> Option<f64> {
    let mut selected = values
        .iter()
        .zip(mask.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return None;
    }
    selected.sort_by(f64::total_cmp);
    Some(selected[selected.len() / 2])
}

fn head_centroid(ct_hu: &Array2<f64>) -> (f64, f64) {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut n = 0.0;
    for ((ix, iy), hu) in ct_hu.indexed_iter() {
        if *hu > -300.0 {
            sx += ix as f64;
            sy += iy as f64;
            n += 1.0;
        }
    }
    if n > 0.0 {
        (sx / n, sy / n)
    } else {
        let (nx, ny) = ct_hu.dim();
        ((nx - 1) as f64 * 0.5, (ny - 1) as f64 * 0.5)
    }
}

fn head_centroid_bool(mask: &Array2<bool>) -> (f64, f64) {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut n = 0.0;
    for ((ix, iy), active) in mask.indexed_iter() {
        if *active {
            sx += ix as f64;
            sy += iy as f64;
            n += 1.0;
        }
    }
    if n > 0.0 {
        (sx / n, sy / n)
    } else {
        let (nx, ny) = mask.dim();
        ((nx - 1) as f64 * 0.5, (ny - 1) as f64 * 0.5)
    }
}

fn brain_speed(hu: f64) -> f64 {
    1510.0 + 55.0 * ((hu + 20.0) / 140.0).clamp(0.0, 1.0)
}

fn soft_attenuation() -> f64 {
    0.5 * 100.0 * std::f64::consts::LN_10 / 20.0
}
