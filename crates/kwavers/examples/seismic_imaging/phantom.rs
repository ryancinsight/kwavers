//! Synthetic and explicit CT phantom construction for the planar workflow.

use anyhow::Context as _;
use leto::Array3;

use super::seismic_imaging::ct::{
    load_ct_volume, skull_centroid_2d, skull_equator_z, skull_outer_radius_ct, CtVolume,
};
use super::seismic_imaging::medium::SkullModel;
use super::seismic_input::SeismicInputMode;
use super::{
    HU_BRAIN, HU_CORTICAL_IN, HU_CORTICAL_OUT, HU_DIPLOE, HU_SCALP, HU_WATER, NX, NY, NZ, R_BRAIN,
    R_DIPLOE, R_HEAD, R_SKULL_IN, R_SKULL_OUT,
};

/// Build the synthetic concentric-shell skull phantom.
pub(super) fn build_skull_phantom() -> super::KwaversResult<SkullModel> {
    let cx = (NX / 2) as f64;
    let cz = (NZ / 2) as f64;
    let mut hu = Array3::<f64>::from_elem((NX, NY, NZ), HU_WATER);

    for i in 0..NX {
        for k in 0..NZ {
            let dx = i as f64 - cx;
            let dz = k as f64 - cz;
            let r = (dx * dx + dz * dz).sqrt();
            let voxel_hu = if r > R_HEAD {
                HU_WATER
            } else if r > R_SKULL_OUT {
                HU_SCALP
            } else if r > R_DIPLOE {
                HU_CORTICAL_OUT
            } else if r > R_SKULL_IN {
                HU_DIPLOE
            } else if r > R_BRAIN {
                HU_CORTICAL_IN
            } else {
                HU_BRAIN
            };
            for j in 0..NY {
                hu[[i, j, k]] = voxel_hu;
            }
        }
    }

    SkullModel::from_hu(hu)
}

fn bilinear_hu(hu: &Array3<f64>, x: f64, y: f64, z: usize) -> f64 {
    let [nx, ny, nz] = hu.shape();
    if z >= nz {
        return 0.0;
    }
    let clamp_x = |index: isize| index.clamp(0, nx as isize - 1) as usize;
    let clamp_y = |index: isize| index.clamp(0, ny as isize - 1) as usize;
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let h00 = hu[[clamp_x(x0), clamp_y(y0), z]];
    let h10 = hu[[clamp_x(x0 + 1), clamp_y(y0), z]];
    let h01 = hu[[clamp_x(x0), clamp_y(y0 + 1), z]];
    let h11 = hu[[clamp_x(x0 + 1), clamp_y(y0 + 1), z]];
    h00 * (1.0 - fx) * (1.0 - fy) + h10 * fx * (1.0 - fy) + h01 * (1.0 - fx) * fy + h11 * fx * fy
}

fn resample_ct_to_fwi_grid(vol: &CtVolume) -> Array3<f64> {
    let hu = vol.hu();
    let z_eq = skull_equator_z(hu);
    let (cx, cy) = skull_centroid_2d(hu, z_eq);
    let r_skull_ct = skull_outer_radius_ct(hu, z_eq, cx, cy);
    let spacing_mm = vol.spacing_mm();
    let scale = r_skull_ct / R_HEAD;

    println!(
        "  CT skull radius : {r_skull_ct:.1} px × {:.2} mm/px = {:.0} mm",
        spacing_mm[0],
        r_skull_ct * spacing_mm[0]
    );
    println!(
        "  FWI fit scale   : {scale:.2} CT px / FWI voxel  \
              (skull outer edge → R_HEAD={R_HEAD} voxels)"
    );

    let mut result = Array3::<f64>::zeros((NX, NY, NZ));
    for ix in 0..NX {
        for iz in 0..NZ {
            let x_ct = cx + (ix as f64 - NX as f64 / 2.0) * scale;
            let y_ct = cy + (iz as f64 - NZ as f64 / 2.0) * scale;
            let hu_val = bilinear_hu(hu, x_ct, y_ct, z_eq);
            for iy in 0..NY {
                result[[ix, iy, iz]] = hu_val;
            }
        }
    }

    let brain = super::seismic_brain_model::brain_support_from_hu(&result);
    for ix in 0..NX {
        for iz in 0..NZ {
            if brain[[ix, iz]] && result[[ix, 0, iz]] < 250.0 {
                for iy in 0..NY {
                    result[[ix, iy, iz]] = HU_BRAIN;
                }
            }
        }
    }
    result
}

/// Build the synthetic or explicit CT phantom selected by the workflow mode.
pub(super) fn build_phantom_for_demo(
    input: &SeismicInputMode,
) -> anyhow::Result<(SkullModel, Option<CtVolume>)> {
    let SeismicInputMode::Ct(path) = input else {
        if matches!(input, SeismicInputMode::CtMri { .. }) {
            anyhow::bail!("the 2-D seismic workflow accepts synthetic or ct:<path> input only");
        }
        println!("  Phantom         : synthetic analytical skull");
        return Ok((build_skull_phantom()?, None));
    };

    print!("  CT source       : {}  ", path.display());
    let vol = load_ct_volume(path)
        .with_context(|| format!("explicit CT input could not be loaded: {}", path.display()))?;
    let [cx, cy, nz] = vol.hu().shape();
    let spacing_mm = vol.spacing_mm();
    println!(
        "({cx}×{cy}×{nz} voxels @ [{:.2},{:.2},{:.2}] mm)",
        spacing_mm[0], spacing_mm[1], spacing_mm[2]
    );
    let hu_fwi = resample_ct_to_fwi_grid(&vol);
    let phantom = SkullModel::from_hu(hu_fwi)?;
    Ok((phantom, Some(vol)))
}
