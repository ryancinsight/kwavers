use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

type PmlArraysFlat = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

pub(super) fn build_pml_arrays(
    pml_size_xyz: Option<(usize, usize, usize)>,
    kgrid: &kwavers_grid::Grid,
    c_ref: f64,
    dt: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> PyResult<PmlArraysFlat> {
    use kwavers_boundary::cpml::{CPMLConfig, CPMLProfiles};

    let total = nx * ny * nz;
    let (pml_x_sz, pml_y_sz, pml_z_sz) = pml_size_xyz.unwrap_or((10, 10, 10));
    let pml_config = CPMLConfig::with_per_dimension_thickness(pml_x_sz, pml_y_sz, pml_z_sz);
    let profiles = CPMLProfiles::new(&pml_config, kgrid, c_ref, dt)
        .map_err(|e| PyRuntimeError::new_err(format!("PML init failed: {e}")))?;

    let pml_sgx_1d: Vec<f32> = profiles
        .sigma_x_sgx
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();
    let pml_sgy_1d: Vec<f32> = profiles
        .sigma_y_sgy
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();
    let pml_sgz_1d: Vec<f32> = profiles
        .sigma_z_sgz
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();
    let pml_x_1d: Vec<f32> = profiles
        .sigma_x
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();
    let pml_y_1d: Vec<f32> = profiles
        .sigma_y
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();
    let pml_z_1d: Vec<f32> = profiles
        .sigma_z
        .iter()
        .map(|&s| (-s * dt * 0.5).exp() as f32)
        .collect();

    let mut pml_x_3d = vec![1.0f32; total];
    let mut pml_y_3d = vec![1.0f32; total];
    let mut pml_z_3d = vec![1.0f32; total];
    let mut pml_sgx_3d = vec![1.0f32; total];
    let mut pml_sgy_3d = vec![1.0f32; total];
    let mut pml_sgz_3d = vec![1.0f32; total];
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let flat = ix * ny * nz + iy * nz + iz;
                pml_sgx_3d[flat] = pml_sgx_1d[ix];
                pml_sgy_3d[flat] = pml_sgy_1d[iy];
                pml_sgz_3d[flat] = pml_sgz_1d[iz];
                pml_x_3d[flat] = pml_x_1d[ix];
                pml_y_3d[flat] = pml_y_1d[iy];
                pml_z_3d[flat] = pml_z_1d[iz];
            }
        }
    }

    Ok((
        pml_x_3d, pml_y_3d, pml_z_3d, pml_sgx_3d, pml_sgy_3d, pml_sgz_3d,
    ))
}
