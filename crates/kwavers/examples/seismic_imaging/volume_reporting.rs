//! 3-D reconstruction artifact writing and output-directory handling.

use super::{
    seismic_volume_artifacts, Array3, KwaversError, KwaversResult, BRAIN_C_MAX, BRAIN_C_MIN,
};
use kwavers_core::constants::acoustic_parameters::SOUND_SPEED_SKULL_CORTICAL;
use std::path::PathBuf;

/// Write reconstructed skull, T1-derived, and brain-tissue volume artifacts.
pub(super) fn write_outputs(
    output_dir: PathBuf,
    reconstructed: &Array3<f64>,
    t1_brain_model: Option<&Array3<f64>>,
    brain_reconstructed: Option<&Array3<f64>>,
) -> KwaversResult<()> {
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| KwaversError::InvalidInput(format!("cannot create output dir: {e}")))?;

    let abs_dir = std::fs::canonicalize(&output_dir).map_err(|error| {
        KwaversError::InvalidInput(format!("cannot canonicalize output dir: {error}"))
    })?;

    let c_lo = kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    let c_hi = SOUND_SPEED_SKULL_CORTICAL;
    let skull_path = abs_dir.join("brain3d_fwi_orthogonal.png");
    let t1_tissue_path = abs_dir.join("brain3d_t1_tissue.png");
    let brain_tissue_path = abs_dir.join("brain3d_brain_tissue.png");

    seismic_volume_artifacts::write_orthogonal_slices_png(&skull_path, reconstructed, c_lo, c_hi)
        .map_err(|e| KwaversError::InvalidInput(format!("skull PNG write failed: {e}")))?;

    if let Some(t1_brain) = t1_brain_model {
        seismic_volume_artifacts::write_orthogonal_slices_png(
            &t1_tissue_path,
            t1_brain,
            BRAIN_C_MIN,
            BRAIN_C_MAX,
        )
        .map_err(|e| KwaversError::InvalidInput(format!("T1 tissue PNG write failed: {e}")))?;
        println!("  T1 tissue image  : {}", t1_tissue_path.display());
    }

    if let Some(bt_recon) = brain_reconstructed {
        seismic_volume_artifacts::write_orthogonal_slices_png(
            &brain_tissue_path,
            bt_recon,
            BRAIN_C_MIN,
            BRAIN_C_MAX,
        )
        .map_err(|e| KwaversError::InvalidInput(format!("brain tissue PNG write failed: {e}")))?;
        println!("  Brain tissue image: {}", brain_tissue_path.display());
    }

    println!("\n  Output directory  : {}", abs_dir.display());
    println!("\n  Wrote images:");
    println!(
        "    {}  (axial+coronal+sagittal skull velocity, reconstructed)",
        skull_path.display()
    );
    Ok(())
}
