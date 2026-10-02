//! Planar seismic output-directory and artifact orchestration.

use super::seismic_imaging::ct::CtVolume;
use super::seismic_imaging::medium::SkullModel;
use super::{
    seismic_acquisition, seismic_planar_artifacts, seismic_planar_auxiliary, Array3, KwaversError,
    KwaversResult, BRAIN_C_MAX, BRAIN_C_MIN, COLORBAR_H, C_HI, C_LO, PANEL,
};
use std::io::Write;
use std::path::PathBuf;

/// Inputs to the planar artifact writer.
pub(super) struct PlanarOutput<'a> {
    pub(super) output_dir: PathBuf,
    pub(super) phantom: &'a SkullModel,
    pub(super) ct_vol: Option<&'a CtVolume>,
    pub(super) true_model: &'a Array3<f64>,
    pub(super) initial_model: &'a Array3<f64>,
    pub(super) reconstructed: &'a Array3<f64>,
    pub(super) brain_true: Option<&'a Array3<f64>>,
    pub(super) brain_reconstructed: Option<&'a Array3<f64>>,
    pub(super) rtm_image: &'a Array3<f64>,
}

/// Write all planar reconstruction artifacts and print their value semantics.
pub(super) fn write_outputs(request: PlanarOutput<'_>) -> KwaversResult<()> {
    let PlanarOutput {
        output_dir,
        phantom,
        ct_vol,
        true_model,
        initial_model,
        reconstructed,
        brain_true,
        brain_reconstructed,
        rtm_image,
    } = request;
    std::fs::create_dir_all(&output_dir).map_err(|error| {
        KwaversError::InvalidInput(format!("cannot create output dir: {error}"))
    })?;
    let absolute_dir = std::fs::canonicalize(&output_dir).map_err(|error| {
        KwaversError::InvalidInput(format!("cannot canonicalize output dir: {error}"))
    })?;

    let base = "brain_fwi";
    let three_plane_path = absolute_dir.join(format!("{base}_three_plane.png"));
    let velocity_ppm_path = absolute_dir.join(format!("{base}.ppm"));
    let rtm_path = absolute_dir.join(format!("{base}_rtm.ppm"));
    let brain_prior_path = absolute_dir.join(format!("{base}_ct_brain_prior.png"));
    let csv_path = absolute_dir.join(format!("{base}.csv"));
    let brain_tissue_path = absolute_dir.join(format!("{base}_brain_tissue.png"));

    let shot_positions = seismic_acquisition::transmit_positions();
    let active_elements = seismic_acquisition::ACTIVE_TRANSDUCER_POSITIONS.to_vec();
    let acquisition = seismic_planar_artifacts::AcquisitionMarkers {
        shot_positions: &shot_positions,
        active_elements: &active_elements,
    };

    seismic_planar_artifacts::write_three_plane_png(
        &three_plane_path,
        true_model,
        reconstructed,
        seismic_planar_artifacts::VelocityScale { lo: C_LO, hi: C_HI },
        acquisition,
        ct_vol,
    )
    .map_err(|error| KwaversError::InvalidInput(format!("PNG write failed: {error}")))?;
    seismic_planar_artifacts::write_velocity_panels(
        &velocity_ppm_path,
        true_model,
        initial_model,
        reconstructed,
        &shot_positions,
        &active_elements,
    )
    .map_err(|error| KwaversError::InvalidInput(format!("velocity panel write failed: {error}")))?;
    seismic_planar_auxiliary::write_brain_prior_png(
        &brain_prior_path,
        phantom.hu(),
        &shot_positions,
        &active_elements,
    )
    .map_err(|error| {
        KwaversError::InvalidInput(format!("brain prior PNG write failed: {error}"))
    })?;
    seismic_planar_auxiliary::write_rtm_panel(&rtm_path, rtm_image)
        .map_err(|error| KwaversError::InvalidInput(format!("RTM panel write failed: {error}")))?;
    seismic_planar_auxiliary::write_velocity_csv(
        &csv_path,
        true_model,
        initial_model,
        reconstructed,
    )
    .map_err(|error| KwaversError::InvalidInput(format!("CSV write failed: {error}")))?;

    if let (Some(brain_true), Some(brain_reconstructed)) = (brain_true, brain_reconstructed) {
        seismic_planar_auxiliary::write_brain_tissue_png(
            &brain_tissue_path,
            brain_true,
            brain_reconstructed,
        )
        .map_err(|error| {
            KwaversError::InvalidInput(format!("brain tissue PNG write failed: {error}"))
        })?;
    }

    let _ = writeln!(
        std::io::stdout().lock(),
        "\n  Output directory  : {}",
        absolute_dir.display()
    );
    let _ = writeln!(std::io::stdout().lock(), "\n  Wrote images and data:");
    let three_plane_desc = if ct_vol.is_some() {
        "PNG 3×2: CT coronal|axial|sagittal (top) / FWI true|reconstructed|difference (bottom)"
    } else {
        "PNG: true skull (FWI grid) | FWI reconstructed | difference — coronal x-z"
    };
    let _ = writeln!(
        std::io::stdout().lock(),
        "    {}  ({three_plane_desc})",
        three_plane_path.display()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    {}  (PPM 4-panel: true | initial | reconstructed | error)",
        velocity_ppm_path.display()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    {}  (PNG CT-derived brain/skull prior + transducer)",
        brain_prior_path.display()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    {}  (PPM RTM zero-lag cross-correlation)",
        rtm_path.display()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "    {}  (CSV depth profile at x = NX/2)",
        csv_path.display()
    );
    if brain_reconstructed.is_some() {
        let _ = writeln!(std::io::stdout().lock(),
            "    {}  (PNG brain tissue: true|reconstructed|difference, [{BRAIN_C_MIN:.0},{BRAIN_C_MAX:.0}] m/s colormap)",
            brain_tissue_path.display()
        );
    }
    if ct_vol.is_some() {
        let _ = writeln!(
            std::io::stdout().lock(),
            "  Image size        : {}×{} px (3×{PANEL} wide, 2×({PANEL}+{COLORBAR_H}) tall)",
            3 * PANEL,
            2 * (PANEL + COLORBAR_H)
        );
    } else {
        let _ =
            writeln!(std::io::stdout().lock(),
            "  Image size        : {PANEL}×{PANEL} px per panel, 3 panels, {COLORBAR_H}px colorbar"
        );
    }
    let _ =
        writeln!(std::io::stdout().lock(),
        "  Colormap          : blue (1500 m/s, water/brain) → red ({C_HI:.0} m/s, cortical bone)"
    );
    if ct_vol.is_some() {
        let _ = writeln!(std::io::stdout().lock(),
            "  PNG layout        : 3×2 grid — top: CT coronal | axial | sagittal (bone window); bottom: FWI true | reconstructed | difference"
        );
    } else {
        let _ = writeln!(
            std::io::stdout().lock(),
            "  PNG panels        : true skull | reconstructed | difference (x-z coronal, y=0)"
        );
    }
    let _ =
        writeln!(std::io::stdout().lock(),
        "  Markers           : white = transmitting elements | yellow = active transducer samples"
    );
    Ok(())
}
