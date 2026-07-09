//! `.npz` loader for cached [`FocalKernel`]s produced by the Python
//! kernel generator (`pykwavers/examples/book/cavitation_kernel.py`).
//!
//! On-disk schema (compressed `.npz`):
//!
//! | field            | dtype     | shape          | semantics                              |
//! |------------------|-----------|----------------|----------------------------------------|
//! | `p_min`          | `float64` | `[nx, ny, nz]` | peak rarefactional (signed; `≤ 0`)     |
//! | `p_max`          | `float64` | `[nx, ny, nz]` | peak compressional (signed; `≥ 0`)     |
//! | `p_rms`          | `float64` | `[nx, ny, nz]` | RMS pressure (non-negative)            |
//! | `dx`             | `float64` | scalar         | isotropic grid spacing (m)             |
//! | `f0`             | `float64` | scalar         | source centre frequency (Hz)           |
//! | `pnp_realised`   | `float64` | scalar         | realised peak rarefactional (Pa, ≥ 0)  |
//! | `source_pa`      | `float64` | scalar         | drive pressure at bowl surface (Pa)    |
//! | `focus_idx`      | `int64`   | `[3]`          | focal voxel `(i, j, k)`                |
//! | `fwhm_lat_m`     | `float64` | scalar         | Penttinen lateral focal FWHM (m)       |
//! | `fwhm_ax_m`      | `float64` | scalar         | Penttinen axial focal FWHM (m)         |
//!
//! [`FocalKernel`] stores `field` as **non-negative peak rarefactional
//! pressure**; this loader negates the on-disk `p_min` so the in-memory
//! representation matches the rest of the field-surrogate stack.
//!
//! Linear rescaling: pass `target_pnp_pa` to [`load_focal_kernel`] to
//! scale the entire field by `target_pnp_pa / pnp_realised`. This is
//! exact in linear water (B/A = 0); for non-linear media use a kernel
//! generated at the target amplitude instead.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use leto::{
    Array1,
    Array3,
};
use ndarray_npy::NpzReader;

use kwavers_core::error::{KwaversError, KwaversResult};

use super::kernel::FocalKernel;

/// Map an `ndarray-npy` read error into a `KwaversError::InvalidInput`
/// that names the missing/incompatible array and the source path.
fn map_read_err<E: std::fmt::Debug>(name: &str, path: &Path, err: E) -> KwaversError {
    KwaversError::InvalidInput(format!(
        "FocalKernel npz `{}`: failed to read `{name}`: {err:?}",
        path.display()
    ))
}

fn read_field<R: std::io::Read + std::io::Seek>(
    npz: &mut NpzReader<R>,
    path: &Path,
) -> KwaversResult<Array3<f64>> {
    let arr: Array3<f64> = npz
        .by_name("p_min")
        .map_err(|e| map_read_err("p_min", path, e))?;
    Ok(arr)
}

fn read_scalar<R: std::io::Read + std::io::Seek>(
    npz: &mut NpzReader<R>,
    name: &str,
    path: &Path,
) -> KwaversResult<f64> {
    let arr: Array1<f64> = npz.by_name(name).map_err(|e| map_read_err(name, path, e))?;
    if arr.len() != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "FocalKernel npz `{}`: expected scalar `{name}`, got shape {:?}",
            path.display(),
            arr.shape()
        )));
    }
    Ok(arr[0])
}

fn read_focus_idx<R: std::io::Read + std::io::Seek>(
    npz: &mut NpzReader<R>,
    path: &Path,
) -> KwaversResult<(usize, usize, usize)> {
    // Python's `np.array((i, j, k), dtype=np.int64)` round-trips as
    // either i64 or i32 depending on the writer; try i64 first, then
    // fall back to i32 for older fixtures.
    let arr: Vec<i64> = match npz.by_name::<Vec<i64>, usize>("focus_idx") {
        Ok(a) => a.iter().copied().collect(),
        Err(_) => {
            let a: Array1<i32> = npz
                .by_name("focus_idx")
                .map_err(|e| map_read_err("focus_idx", path, e))?;
            a.iter().map(|&v| i64::from(v)).collect()
        }
    };
    if arr.len() != 3 {
        return Err(KwaversError::InvalidInput(format!(
            "FocalKernel npz `{}`: focus_idx has {} components, want 3",
            path.display(),
            arr.len()
        )));
    }
    if arr.iter().any(|&v| v < 0) {
        return Err(KwaversError::InvalidInput(format!(
            "FocalKernel npz `{}`: focus_idx contains negative components: {:?}",
            path.display(),
            arr
        )));
    }
    Ok((arr[0] as usize, arr[1] as usize, arr[2] as usize))
}

/// Load a `FocalKernel` from a single compressed `.npz` file.
///
/// `target_pnp_pa` optionally rescales the field by
/// `target_pnp_pa / pnp_realised`. Passing `None` returns the kernel
/// at its stored amplitude.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] when:
/// - The file cannot be opened.
/// - A required array is missing or has an incompatible dtype/shape.
/// - The stored `focus_idx` falls outside `field`'s shape.
/// - `target_pnp_pa` is requested but `pnp_realised` is non-positive.
pub fn load_focal_kernel(path: &Path, target_pnp_pa: Option<f64>) -> KwaversResult<FocalKernel> {
    let file = File::open(path).map_err(|e| {
        KwaversError::InvalidInput(format!(
            "FocalKernel npz `{}`: open failed: {e}",
            path.display()
        ))
    })?;
    let mut npz = NpzReader::new(BufReader::new(file)).map_err(|e| {
        KwaversError::InvalidInput(format!(
            "FocalKernel npz `{}`: not a valid .npz archive: {e:?}",
            path.display()
        ))
    })?;

    // `p_min` is non-positive on disk (peak rarefactional is the
    // *minimum* of the signed pressure record). Negate to recover the
    // non-negative magnitude FocalKernel expects.
    let p_min: Array3<f64> = read_field(&mut npz, path)?;
    let field: Array3<f64> = p_min.mapv(|v| -v);

    let dx_m = read_scalar(&mut npz, "dx", path)?;
    let f0 = read_scalar(&mut npz, "f0", path)?;
    let mut pnp_realised = read_scalar(&mut npz, "pnp_realised", path)?;
    let mut source_pa = read_scalar(&mut npz, "source_pa", path)?;
    let fwhm_lat_m = read_scalar(&mut npz, "fwhm_lat_m", path)?;
    let fwhm_ax_m = read_scalar(&mut npz, "fwhm_ax_m", path)?;
    let focus_idx = read_focus_idx(&mut npz, path)?;

    if dx_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "FocalKernel npz `{}`: dx={dx_m} must be > 0",
            path.display()
        )));
    }
    if f0 <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "FocalKernel npz `{}`: f0={f0} must be > 0",
            path.display()
        )));
    }
    let (nx, ny, nz) = field.dim();
    if focus_idx.0 >= nx || focus_idx.1 >= ny || focus_idx.2 >= nz {
        return Err(KwaversError::InvalidInput(format!(
            "FocalKernel npz `{}`: focus_idx {:?} out of bounds for shape ({nx}, {ny}, {nz})",
            path.display(),
            focus_idx
        )));
    }

    let mut field = field;
    if let Some(target) = target_pnp_pa {
        if pnp_realised <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "FocalKernel npz `{}`: cannot rescale to {target} Pa from pnp_realised={pnp_realised}",
                path.display()
            )));
        }
        let scale = target / pnp_realised;
        field.mapv_inplace(|p| p * scale);
        source_pa *= scale;
        pnp_realised = target;
    }

    Ok(FocalKernel::new(
        field,
        dx_m,
        focus_idx,
        f0,
        pnp_realised,
        source_pa,
        fwhm_lat_m,
        fwhm_ax_m,
    ))
}

/// Scan `dir` for files matching `kernel_*.npz` and load each one.
///
/// The scan is non-recursive and the result preserves filesystem
/// iteration order; callers that want a deterministic ordering should
/// sort by `(f0, pnp_realised)` after loading. Files that fail to
/// parse abort the call — partial results are not returned, so a
/// corrupt fixture surfaces immediately.
///
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if `dir` cannot be read
///   or if any matching file fails to parse.
pub fn discover_focal_kernels(dir: &Path) -> KwaversResult<Vec<FocalKernel>> {
    let entries = std::fs::read_dir(dir).map_err(|e| {
        KwaversError::InvalidInput(format!(
            "discover_focal_kernels: cannot read `{}`: {e}",
            dir.display()
        ))
    })?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("kernel_") && name.ends_with(".npz")
        })
        .collect();
    paths.sort();
    let mut out = Vec::with_capacity(paths.len());
    for path in &paths {
        out.push(load_focal_kernel(path, None)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
    use array;
    use ndarray_npy::NpzWriter;
    use std::io::Cursor;

    fn write_fixture_npz() -> Vec<u8> {
        // Build a tiny 4×3×3 kernel with the focal voxel at (2, 1, 1)
        // carrying p_min = -10 MPa. Off-focus voxels carry small
        // values so the round-trip can verify shape preservation.
        let mut p_min = leto::Array3::<f64>::from_elem((4, 3, 3), -1.0e3);
        p_min[[2, 1, 1]] = -1.0e7;

        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = NpzWriter::new(&mut buf);
            w.add_array("p_min", &p_min).unwrap();
            w.add_array("dx", &array![5.0e-4_f64]).unwrap();
            w.add_array("f0", &array![MHZ_TO_HZ]).unwrap();
            w.add_array("pnp_realised", &array![10.0 * MPA_TO_PA])
                .unwrap();
            w.add_array("source_pa", &array![1.5 * MPA_TO_PA]).unwrap();
            w.add_array("fwhm_lat_m", &array![2.0e-3_f64]).unwrap();
            w.add_array("fwhm_ax_m", &array![6.0e-3_f64]).unwrap();
            w.add_array("focus_idx", &array![2_i64, 1, 1]).unwrap();
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    fn write_to_tempfile(bytes: &[u8], name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("kwavers_npz_loader_tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn round_trip_preserves_field_and_metadata() {
        let bytes = write_fixture_npz();
        let path = write_to_tempfile(&bytes, "kernel_roundtrip.npz");
        let kernel = load_focal_kernel(&path, None).expect("load");
        assert_eq!(kernel.shape(), (4, 3, 3));
        // The loader negates p_min, so the focal magnitude must be +1e7.
        assert!((kernel.focal_pressure() - 10.0 * MPA_TO_PA).abs() < 1e-3);
        assert!((kernel.dx_m - 5.0e-4).abs() < 1e-12);
        assert!((kernel.f0 - MHZ_TO_HZ).abs() < 1e-3);
        assert!((kernel.pnp_realised - 10.0 * MPA_TO_PA).abs() < 1e-3);
        assert!((kernel.source_pa - 1.5 * MPA_TO_PA).abs() < 1e-3);
        assert!((kernel.fwhm_lat_m - 2.0e-3).abs() < 1e-9);
        assert!((kernel.fwhm_ax_m - 6.0e-3).abs() < 1e-9);
        assert_eq!(kernel.focus_idx, (2, 1, 1));
    }

    #[test]
    fn rescaling_scales_field_and_pnp_linearly() {
        let bytes = write_fixture_npz();
        let path = write_to_tempfile(&bytes, "kernel_rescale.npz");
        let target = 30.0 * MPA_TO_PA;
        let kernel = load_focal_kernel(&path, Some(target)).expect("rescaled load");
        // Field was rescaled by 3×: focal magnitude = 30 MPa.
        assert!((kernel.focal_pressure() - target).abs() < 1.0);
        assert!((kernel.pnp_realised - target).abs() < 1.0);
        // Source pressure scales by the same factor: 1.5 MPa × 3 = 4.5 MPa.
        assert!((kernel.source_pa - 4.5 * MPA_TO_PA).abs() < 1.0);
    }

    #[test]
    fn missing_array_surfaces_invalid_input() {
        // Write a fixture missing `f0`; loader must reject with an
        // InvalidInput error that names the missing field.
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = NpzWriter::new(&mut buf);
            let p_min = leto::Array3::<f64>::zeros((2, 2, 2));
            w.add_array("p_min", &p_min).unwrap();
            w.add_array("dx", &array![5.0e-4_f64]).unwrap();
            w.add_array("pnp_realised", &array![10.0 * MPA_TO_PA])
                .unwrap();
            w.add_array("source_pa", &array![MPA_TO_PA]).unwrap();
            w.add_array("fwhm_lat_m", &array![2.0e-3_f64]).unwrap();
            w.add_array("fwhm_ax_m", &array![6.0e-3_f64]).unwrap();
            w.add_array("focus_idx", &array![1_i64, 1, 1]).unwrap();
            w.finish().unwrap();
        }
        let bytes = buf.into_inner();
        let path = write_to_tempfile(&bytes, "kernel_missing_f0.npz");
        let err = load_focal_kernel(&path, None).expect_err("must fail");
        let msg = err.to_string();
        assert!(msg.contains("f0"), "error message must name `f0`: {msg}");
    }

    #[test]
    fn focus_out_of_bounds_rejected() {
        let mut buf = Cursor::new(Vec::<u8>::new());
        {
            let mut w = NpzWriter::new(&mut buf);
            let p_min = leto::Array3::<f64>::zeros((2, 2, 2));
            w.add_array("p_min", &p_min).unwrap();
            w.add_array("dx", &array![1.0e-3_f64]).unwrap();
            w.add_array("f0", &array![MHZ_TO_HZ]).unwrap();
            w.add_array("pnp_realised", &array![10.0 * MPA_TO_PA])
                .unwrap();
            w.add_array("source_pa", &array![MPA_TO_PA]).unwrap();
            w.add_array("fwhm_lat_m", &array![2.0e-3_f64]).unwrap();
            w.add_array("fwhm_ax_m", &array![6.0e-3_f64]).unwrap();
            w.add_array("focus_idx", &array![5_i64, 0, 0]).unwrap(); // 5 ≥ nx=2
            w.finish().unwrap();
        }
        let path = write_to_tempfile(&buf.into_inner(), "kernel_oob.npz");
        let err = load_focal_kernel(&path, None).expect_err("must fail");
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn discover_loads_all_matching_files_sorted() {
        let dir = std::env::temp_dir().join("kwavers_npz_discover");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Two valid fixtures plus a non-matching file that must be
        // ignored by the glob.
        let bytes = write_fixture_npz();
        std::fs::write(dir.join("kernel_a.npz"), &bytes).unwrap();
        std::fs::write(dir.join("kernel_b.npz"), &bytes).unwrap();
        std::fs::write(dir.join("README.txt"), "not a kernel").unwrap();
        let kernels = discover_focal_kernels(&dir).expect("discover");
        assert_eq!(kernels.len(), 2);
    }
}
