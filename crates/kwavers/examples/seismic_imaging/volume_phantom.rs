//! Full-volume skull phantom construction and CT interpolation.

use anyhow::Context as _;
use leto::Array3;

use super::seismic_imaging::ct::{
    CtVolume, load_ct_volume, skull_centroid_2d, skull_equator_z, skull_outer_radius_ct,
};
use super::seismic_imaging::medium::SkullModel;
use super::seismic_input::SeismicInputMode;
use super::{
    DX, HU_BRAIN, HU_CORTICAL_IN, HU_CORTICAL_OUT, HU_DIPLOE, HU_SCALP, HU_WATER, NX, NY, NZ,
    R_BRAIN, R_DIPLOE, R_HEAD, R_SKULL_IN, R_SKULL_OUT,
};

/// Interpolate a volume at a fractional coordinate with clamped boundaries.
pub(super) fn trilinear_hu(hu: &Array3<f64>, x: f64, y: f64, z: f64) -> f64 {
    let [nx, ny, nz] = hu.shape();
    if nx == 0 || ny == 0 || nz == 0 {
        return 0.0;
    }
    let clamp_x = |index: isize| index.clamp(0, nx as isize - 1) as usize;
    let clamp_y = |index: isize| index.clamp(0, ny as isize - 1) as usize;
    let clamp_z = |index: isize| index.clamp(0, nz as isize - 1) as usize;
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let z0 = z.floor() as isize;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let fz = z - z.floor();
    let h000 = hu[[clamp_x(x0), clamp_y(y0), clamp_z(z0)]];
    let h100 = hu[[clamp_x(x0 + 1), clamp_y(y0), clamp_z(z0)]];
    let h010 = hu[[clamp_x(x0), clamp_y(y0 + 1), clamp_z(z0)]];
    let h110 = hu[[clamp_x(x0 + 1), clamp_y(y0 + 1), clamp_z(z0)]];
    let h001 = hu[[clamp_x(x0), clamp_y(y0), clamp_z(z0 + 1)]];
    let h101 = hu[[clamp_x(x0 + 1), clamp_y(y0), clamp_z(z0 + 1)]];
    let h011 = hu[[clamp_x(x0), clamp_y(y0 + 1), clamp_z(z0 + 1)]];
    let h111 = hu[[clamp_x(x0 + 1), clamp_y(y0 + 1), clamp_z(z0 + 1)]];
    let h00 = h000 * (1.0 - fx) + h100 * fx;
    let h10 = h010 * (1.0 - fx) + h110 * fx;
    let h01 = h001 * (1.0 - fx) + h101 * fx;
    let h11 = h011 * (1.0 - fx) + h111 * fx;
    let h0 = h00 * (1.0 - fy) + h10 * fy;
    let h1 = h01 * (1.0 - fy) + h11 * fy;
    h0 * (1.0 - fz) + h1 * fz
}

fn resample_ct_to_fwi_grid_3d(vol: &CtVolume) -> Array3<f64> {
    let hu = vol.hu();
    let z_eq = skull_equator_z(hu);
    let (cx_ct, cy_ct) = skull_centroid_2d(hu, z_eq);
    let r_skull_ct = skull_outer_radius_ct(hu, z_eq, cx_ct, cy_ct);
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
    println!(
        "  Grid            : {NX}×{NY}×{NZ} voxels @ {:.0} mm",
        DX * 1e3
    );
    println!(
        "  Domain          : {:.0}×{:.0}×{:.0} mm",
        NX as f64 * DX * 1e3,
        NY as f64 * DX * 1e3,
        NZ as f64 * DX * 1e3
    );

    let mut result = Array3::<f64>::zeros((NX, NY, NZ));
    for ix in 0..NX {
        for iy in 0..NY {
            for iz in 0..NZ {
                let x_ct = cx_ct + (ix as f64 - NX as f64 / 2.0) * scale;
                let y_ct = cy_ct + (iz as f64 - NZ as f64 / 2.0) * scale;
                let z_ct = z_eq as f64 + (iy as f64 - NY as f64 / 2.0) * scale;
                result[[ix, iy, iz]] = trilinear_hu(hu, x_ct, y_ct, z_ct);
            }
        }
    }

    for ix in 0..NX {
        for iy in 0..NY {
            for iz in 0..NZ {
                let dx = ix as f64 - NX as f64 / 2.0;
                let dy = iy as f64 - NY as f64 / 2.0;
                let dz = iz as f64 - NZ as f64 / 2.0;
                let radius = (dx * dx + dy * dy + dz * dz).sqrt();
                if radius < R_SKULL_IN && result[[ix, iy, iz]] < 250.0 {
                    result[[ix, iy, iz]] = HU_BRAIN;
                }
            }
        }
    }

    let hu_min = result.iter().copied().fold(f64::INFINITY, f64::min);
    let hu_max = result.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!("  HU range        : [{hu_min:.0}, {hu_max:.0}]");
    println!(
        "  Head radius     : {:.0} mm  (R_HEAD = {R_HEAD} voxels)",
        R_HEAD * DX * 1e3
    );
    println!(
        "  Skull thickness : ~{:.0} mm (outer cortical → inner cortical)",
        (R_SKULL_OUT - R_SKULL_IN) * DX * 1e3
    );
    println!("  Brain radius    : {:.0} mm", R_BRAIN * DX * 1e3);
    println!("  Layers          : water coupling / scalp / cortical bone / diploe / brain");
    result
}

fn build_skull_phantom_3d() -> super::KwaversResult<SkullModel> {
    let cx = (NX / 2) as f64;
    let cy = (NY / 2) as f64;
    let cz = (NZ / 2) as f64;
    let mut hu = Array3::<f64>::from_elem((NX, NY, NZ), HU_WATER);

    for ix in 0..NX {
        for iy in 0..NY {
            for iz in 0..NZ {
                let dx = ix as f64 - cx;
                let dy = iy as f64 - cy;
                let dz = iz as f64 - cz;
                let radius = (dx * dx + dy * dy + dz * dz).sqrt();
                hu[[ix, iy, iz]] = if radius > R_HEAD {
                    HU_WATER
                } else if radius > R_SKULL_OUT {
                    HU_SCALP
                } else if radius > R_DIPLOE {
                    HU_CORTICAL_OUT
                } else if radius > R_SKULL_IN {
                    HU_DIPLOE
                } else if radius > R_BRAIN {
                    HU_CORTICAL_IN
                } else {
                    HU_BRAIN
                };
            }
        }
    }

    SkullModel::from_hu(hu)
}

/// Build the 3-D phantom selected by the input mode.
pub(super) fn build_phantom_3d(
    input: &SeismicInputMode,
) -> anyhow::Result<(SkullModel, Option<CtVolume>)> {
    let (SeismicInputMode::Ct(path) | SeismicInputMode::CtMri { ct: path, .. }) = input else {
        println!("  Phantom         : synthetic 3D spherical model");
        return Ok((build_skull_phantom_3d()?, None));
    };

    print!("  CT source       : {}  ", path.display());
    let volume = load_ct_volume(path)
        .with_context(|| format!("explicit CT input could not be loaded: {}", path.display()))?;
    let [cx, cy, nz] = volume.hu().shape();
    let spacing_mm = volume.spacing_mm();
    println!(
        "({cx}×{cy}×{nz} voxels @ [{:.2},{:.2},{:.2}] mm)",
        spacing_mm[0], spacing_mm[1], spacing_mm[2]
    );
    let hu_fwi = resample_ct_to_fwi_grid_3d(&volume);
    let phantom = SkullModel::from_hu(hu_fwi)?;
    Ok((phantom, Some(volume)))
}

#[cfg(test)]
mod tests {
    use super::trilinear_hu;
    use leto::Array3;

    #[test]
    fn trilinear_interpolation_matches_the_eight_corner_average() {
        let values = Array3::from_shape_fn((2, 2, 2), |[x, y, z]| {
            1.0 + x as f64 + 2.0 * y as f64 + 4.0 * z as f64
        });

        assert_eq!(trilinear_hu(&values, 0.0, 0.0, 0.0), 1.0);
        assert_eq!(trilinear_hu(&values, 0.5, 0.5, 0.5), 4.5);
    }
}
