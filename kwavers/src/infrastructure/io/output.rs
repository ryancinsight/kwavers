// output/mod.rs
use crate::core::time::Time;
use crate::domain::field::indices::LIGHT_IDX;
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::sensor::recorder::Recorder;
use log::info;
use ndarray::{Array3, Axis};
use std::fs::File;
use std::io::{self, Write};

// Note: Field indices imported from physics::field_indices for SSOT

pub fn save_pressure_data(recorder: &Recorder, time: &Time, filename: &str) -> io::Result<()> {
    info!("Saving pressure data to {}", filename);
    let mut file = File::create(filename)?;

    write!(file, "Time")?;
    for i in 0..recorder.sensor().num_sensors() {
        write!(file, ",Sensor{}", i + 1)?;
    }
    writeln!(file)?;

    match recorder.pressure_data() {
        Some(data) => {
            let max_steps = recorder.recorded_steps.len().min(data.ncols());
            for (t, &time_val) in recorder.recorded_steps.iter().take(max_steps).enumerate() {
                write!(file, "{}", time.time_vector()[t].min(time_val))?;
                data.column(t)
                    .iter()
                    .try_for_each(|&val: &f64| write!(file, ",{val}"))?;
                writeln!(file)?;
            }
        }
        _ => {
            writeln!(file, "No pressure data recorded")?;
        }
    }
    Ok(())
}

pub fn save_light_data(recorder: &Recorder, time: &Time, filename: &str) -> io::Result<()> {
    info!("Saving light data to {}", filename);
    let mut file = File::create(filename)?;

    write!(file, "Time")?;
    for i in 0..recorder.sensor().num_sensors() {
        write!(file, ",Sensor{}", i + 1)?;
    }
    writeln!(file)?;

    match recorder.light_data() {
        Some(data) => {
            let max_steps = recorder.recorded_steps.len().min(data.ncols());
            for (t, &time_val) in recorder.recorded_steps.iter().take(max_steps).enumerate() {
                write!(file, "{}", time.time_vector()[t].min(time_val))?;
                data.column(t)
                    .iter()
                    .try_for_each(|&val: &f64| write!(file, ",{val}"))?;
                writeln!(file)?;
            }
        }
        _ => {
            writeln!(file, "No light data recorded")?;
        }
    }
    Ok(())
}

pub fn generate_summary(recorder: &Recorder, filename: &str) -> io::Result<()> {
    info!("Generating summary to {}", filename);
    let mut file = File::create(filename)?;

    writeln!(file, "Metric,Value")?;

    if let Some((step, fields)) = recorder.fields_snapshots.last() {
        let pressure: ndarray::ArrayView3<f64> =
            fields.index_axis(Axis(0), UnifiedFieldType::Pressure.index());
        let light: ndarray::ArrayView3<f64> = fields.index_axis(Axis(0), LIGHT_IDX);

        let max_pressure = pressure.iter().fold(f64::NEG_INFINITY, |acc, &val: &f64| {
            f64::max(acc, val.abs())
        });
        let avg_pressure = pressure.iter().sum::<f64>() / pressure.len() as f64;
        writeln!(file, "Last Step,{step}")?;
        writeln!(file, "Max Pressure,{max_pressure:.6e}")?;
        writeln!(file, "Avg Pressure,{avg_pressure:.6e}")?;

        let max_light = light
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &val| f64::max(acc, val));
        let avg_light = light.iter().sum::<f64>() / light.len() as f64;
        writeln!(file, "Last Light Step,{step}")?;
        writeln!(file, "Max Light Fluence,{max_light:.6e}")?;
        writeln!(file, "Avg Light Fluence,{avg_light:.6e}")?;
    }

    Ok(())
}

/// Save 3D field data to CSV format.
///
/// Writes data as (x, y, z, value) rows for easy import into
/// analysis tools (MATLAB, Python, R, etc.).
pub fn save_data_csv(data: &Array3<f64>, filename: &str) -> io::Result<()> {
    info!("Saving data to CSV: {}", filename);
    let mut file = File::create(filename)?;
    let (nx, ny, nz) = data.dim();

    // Write header
    writeln!(file, "x,y,z,value")?;

    // Write data
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                writeln!(file, "{},{},{},{}", i, j, k, data[[i, j, k]])?;
            }
        }
    }

    info!("Data saved to {}", filename);
    Ok(())
}
