//! Brain tissue prior construction for the planar seismic workflow.

use anyhow::Context as _;
use coeus_core::MoiraiBackend;
use leto::{Array2, Array3};
use ritk_io::ImageReader;
use ritk_io::format::nifti::native::NiftiReader as NativeNiftiReader;
use std::path::Path;

use super::seismic_imaging::medium::SkullModel;
use super::{
    BONE_VELOCITY_THRESHOLD, BrainPriorMode, C_CSF, C_GRAY, C_WHITE, DX, MNI_INNER_SKULL_RADIUS_MM,
    NX, NY, NZ, R_SKULL_IN, SOUND_SPEED_WATER_SIM,
};

/// Estimate intracranial brain support from a CT-derived HU map.
pub(super) fn brain_support_from_hu(hu: &Array3<f64>) -> Array2<bool> {
    let mut mask = Array2::<bool>::from_elem((NX, NZ), false);
    for iz in 0..NZ {
        let bone: Vec<usize> = (0..NX).filter(|&ix| hu[[ix, 0, iz]] >= 250.0).collect();
        if bone.len() < 2 {
            continue;
        }
        let left = bone[0];
        let right = *bone.last().expect("bone len checked");
        if right <= left + 2 {
            continue;
        }
        for ix in (left + 1)..right {
            if hu[[ix, 0, iz]] < 250.0 {
                mask[[ix, iz]] = true;
            }
        }
    }
    mask
}

// Stage-2 brain tissue helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a per-voxel FWI frozen mask for Stage-2 brain tissue inversion.
///
/// Frozen voxels are restored from the CT reference after every gradient step.
/// Only high-velocity bone voxels (c > BONE_VELOCITY_THRESHOLD) are frozen;
/// scalp, diploe-transition, and water coupling remain free.  This keeps the
/// free region large enough for the FWI gradient to converge while still
/// preventing updates to the cortical bone wall.
///
/// Visualization alignment is handled separately in `write_brain_tissue_png`
/// using the geometric r < R_SKULL_IN criterion, independent of this mask.
pub(super) fn build_skull_mask(sound_speed: &Array3<f64>) -> Array3<bool> {
    sound_speed.mapv(|c| c > BONE_VELOCITY_THRESHOLD)
}

/// Load MNI ICBM 2009c tissue probability maps and resample them onto the FWI
/// 2-D coronal grid, returning a brain tissue velocity model.
///
/// # Tissue velocity mapping (Duck 1990)
///
/// For each free (non-bone) FWI voxel the velocity is a probability-weighted
/// mixture of the three soft-tissue classes:
///
/// ```text
/// c(x) = p_gm(x) × C_GRAY + p_wm(x) × C_WHITE + p_csf(x) × C_CSF
///       + (1 − p_gm − p_wm − p_csf) × c_water
/// ```
///
/// Bone voxels are left at their CT-derived velocities (they will be frozen by
/// `build_skull_mask` during FWI and are never updated).
///
/// # Spatial mapping
///
/// The MNI ICBM 2009c atlas is sampled at the mid-coronal slice (y ≈ rows/2,
/// near the anterior commissure).  Each FWI voxel is mapped to MNI coordinates
/// by scaling: `mni_offset = fwi_offset_mm × (MNI_INNER_SKULL_RADIUS_MM / fwi_inner_skull_mm)`.
fn build_brain_velocity_model(
    skull_phantom: &SkullModel,
    mni_dir: &Path,
) -> anyhow::Result<Array3<f64>> {
    let backend = MoiraiBackend;

    // Load the three probability maps via ritk NIfTI reader.
    // CtVolume.hu stores probability values [0,1] (HU clamping [-1024,3071] is harmless).
    let load = |name: &str| -> anyhow::Result<Array3<f64>> {
        let path = mni_dir.join(name);
        anyhow::ensure!(
            path.exists(),
            "MNI tissue map not found: '{}' — download from {}",
            path.display(),
            "https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09c_nifti.zip"
        );
        // Load through ritk (returns [Z,Y,X] tensor → transposed to hu[X,Y,Z]).
        let img = ImageReader::read(&NativeNiftiReader::new(backend), &path)
            .with_context(|| format!("NIfTI load failed: '{}'", path.display()))?;
        let [depth, rows, cols] = img.shape();
        let vals = img
            .data_slice()
            .map_err(|e| anyhow::anyhow!("NIfTI data not f32: {e:?}"))?;
        let mut vol = Array3::<f64>::zeros((cols, rows, depth));
        for z in 0..depth {
            for y in 0..rows {
                for x in 0..cols {
                    vol[[x, y, z]] =
                        f64::from(vals[z * rows * cols + y * cols + x]).clamp(0.0, 1.0);
                }
            }
        }
        Ok(vol)
    };

    let gm = load("mni_icbm152_gm_tal_nlin_sym_09c.nii")?;
    let wm = load("mni_icbm152_wm_tal_nlin_sym_09c.nii")?;
    let csf = load("mni_icbm152_csf_tal_nlin_sym_09c.nii")?;

    let [mni_nx, mni_ny, mni_nz] = gm.shape();
    // MNI centroid voxel (brain centre-of-mass in MNI space ≈ [nx/2, ny/2, nz/2]).
    let cx_mni = mni_nx / 2; // ~90
    let cy_mni = mni_ny / 2; // ~108 — mid coronal slice (near AC)
    let cz_mni = mni_nz / 2; // ~90

    // Scale factor: project FWI physical offset (mm) into MNI voxel offset.
    // At the inner skull boundary: R_SKULL_IN × DX × 1e3 mm (FWI) ↔ MNI_INNER_SKULL_RADIUS_MM.
    let fwi_inner_mm = R_SKULL_IN * DX * 1e3; // 36 mm
    let fwi_to_mni = MNI_INNER_SKULL_RADIUS_MM / fwi_inner_mm; // 82/36 ≈ 2.28

    let cx_fwi = NX / 2; // 32
    let cz_fwi = NZ / 2; // 32

    let mut brain_model = skull_phantom.acoustic().sound_speed.clone();

    for iz in 0..NZ {
        for ix in 0..NX {
            // Only assign MNI tissue velocities inside the inner skull surface
            // (r < R_SKULL_IN = 36 mm).  The skull wall, scalp, and water coupling
            // bath all retain their CT-derived velocities — they are frozen during
            // Stage-2 FWI and must not carry tissue-velocity artefacts.
            //
            // Using the velocity threshold c_ct > BONE_VELOCITY_THRESHOLD is
            // insufficient because bilinear interpolation at the 15 CT-px / FWI-voxel
            // scale blurs the thin skull wall into intermediate-HU voxels that fall
            // below the threshold, leaking MNI velocities into the scalp ring and
            // creating a false yellow band in the tight [1480,1560] m/s colormap.
            let dx_fwi = ix as f64 - cx_fwi as f64;
            let dz_fwi = iz as f64 - cz_fwi as f64;
            let r_fwi = (dx_fwi * dx_fwi + dz_fwi * dz_fwi).sqrt();
            if r_fwi >= R_SKULL_IN {
                continue; // skull wall, scalp, water bath — keep CT velocity
            }

            // FWI voxel physical offset from grid centre [mm].
            let dx_mm = dx_fwi * DX * 1e3;
            let dz_mm = dz_fwi * DX * 1e3;

            // Map to MNI voxel coordinate.
            let mni_x = (cx_mni as f64 + dx_mm * fwi_to_mni).round() as isize;
            let mni_z = (cz_mni as f64 + dz_mm * fwi_to_mni).round() as isize;

            // Out-of-bounds → keep water velocity.
            if mni_x < 0 || mni_x >= mni_nx as isize || mni_z < 0 || mni_z >= mni_nz as isize {
                continue;
            }
            let mx = mni_x as usize;
            let mz = mni_z as usize;

            // Sample mid-coronal MNI slice.
            let p_gm = gm[[mx, cy_mni, mz]];
            let p_wm = wm[[mx, cy_mni, mz]];
            let p_csf = csf[[mx, cy_mni, mz]];
            let p_rest = (1.0 - p_gm - p_wm - p_csf).clamp(0.0, 1.0);

            let c_tissue =
                p_gm * C_GRAY + p_wm * C_WHITE + p_csf * C_CSF + p_rest * SOUND_SPEED_WATER_SIM;

            for iy in 0..NY {
                brain_model[[ix, iy, iz]] = c_tissue;
            }
        }
    }

    Ok(brain_model)
}

/// Build a deterministic homogeneous brain prior without external datasets.
fn build_uniform_brain_velocity_model(skull_phantom: &SkullModel) -> Array3<f64> {
    let mut brain_model = skull_phantom.acoustic().sound_speed.clone();
    let cx = NX as f64 / 2.0;
    let cz = NZ as f64 / 2.0;
    for ix in 0..NX {
        for iz in 0..NZ {
            let dx = ix as f64 - cx;
            let dz = iz as f64 - cz;
            if (dx * dx + dz * dz).sqrt() >= R_SKULL_IN {
                continue;
            }
            for iy in 0..NY {
                brain_model[[ix, iy, iz]] = SOUND_SPEED_WATER_SIM;
            }
        }
    }
    brain_model
}

pub(super) fn build_brain_prior(
    skull_phantom: &SkullModel,
    prior: &BrainPriorMode,
) -> anyhow::Result<Array3<f64>> {
    match prior {
        BrainPriorMode::Uniform => Ok(build_uniform_brain_velocity_model(skull_phantom)),
        BrainPriorMode::Mni(path) => build_brain_velocity_model(skull_phantom, path),
        BrainPriorMode::T1(_) | BrainPriorMode::MniT1 { .. } => {
            anyhow::bail!("the 2-D workflow accepts uniform or mni:<directory> brain priors")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
