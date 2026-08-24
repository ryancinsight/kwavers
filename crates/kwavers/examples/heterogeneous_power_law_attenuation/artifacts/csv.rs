use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::Result;

use super::super::model::SweepRow;

pub(crate) fn write_sweep_csv(path: &Path, rows: &[SweepRow]) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "alpha0_db_cm_mhz_gamma,gamma,frequency_hz,prescribed_np_m,measured_np_m,relative_error"
    )?;
    for row in rows {
        writeln!(
            file,
            "{},{},{:e},{:.6},{:.6},{:.6}",
            row.alpha0_db,
            row.gamma,
            row.frequency_hz,
            row.prescribed_np_m,
            row.measured_np_m,
            row.relative_error()
        )?;
    }
    Ok(())
}

pub(crate) fn write_layered_csv(path: &Path, rows: &[(f64, f64, f64)]) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "frequency_hz,path_weighted_np_m,measured_np_m,relative_error"
    )?;
    for &(frequency_hz, predicted, measured) in rows {
        writeln!(
            file,
            "{frequency_hz:e},{predicted:.6},{measured:.6},{:.6}",
            (measured - predicted).abs() / predicted
        )?;
    }
    Ok(())
}
