mod artifacts;
mod configuration;
mod experiments;
mod measurement;
mod model;
mod propagation;

use std::fs;
use std::path::PathBuf;

use anyhow::Result;

use self::artifacts::csv::{write_layered_csv, write_sweep_csv};
use self::artifacts::plot::write_plot;
use self::configuration::{
    time_step, ALPHA0_DB, DX, GAMMAS, N, OUT_DIR, SENSOR_FAR, SENSOR_NEAR, STEPS,
};
use self::experiments::homogeneous::homogeneous_sweep;
use self::experiments::layered::{layered_medium, LAYERS};
use self::measurement::reference_spectra;
use self::model::SweepRow;

pub(super) fn run() -> Result<()> {
    let dt = time_step();
    let out_dir = PathBuf::from(OUT_DIR);
    fs::create_dir_all(&out_dir)?;

    println!("Fullwave 2.5 heterogeneous power-law attenuation replication");
    println!(
        "grid {N} x 1 x 1, dx = {DX:e} m, dt = {dt:e} s, {STEPS} steps, \
         sensors {SENSOR_NEAR} -> {SENSOR_FAR}"
    );

    let reference = reference_spectra(dt)?;
    let rows = homogeneous_sweep(dt, &reference)?;
    let sweep_csv = out_dir.join("attenuation_sweep.csv");
    let sweep_png = out_dir.join("attenuation_sweep.png");
    write_sweep_csv(&sweep_csv, &rows)?;
    write_plot(&sweep_png, &rows)?;

    println!("\nhomogeneous sweep — worst relative error per (alpha0, gamma):");
    println!("{:>10} {:>8} {:>14}", "alpha0_db", "gamma", "worst_rel_err");
    let mut worst_overall = 0.0_f64;
    for &alpha0_db in &ALPHA0_DB {
        for &gamma in &GAMMAS {
            let worst = rows
                .iter()
                .filter(|row| {
                    row.alpha0_db == alpha0_db && (row.gamma - gamma).abs() < f64::EPSILON
                })
                .map(SweepRow::relative_error)
                .fold(0.0_f64, f64::max);
            worst_overall = worst_overall.max(worst);
            println!("{alpha0_db:>10} {gamma:>8} {worst:>14.4}");
        }
    }
    println!("worst over the whole envelope: {worst_overall:.4}");

    let layered = layered_medium(dt, &reference)?;
    let layered_csv = out_dir.join("layered_medium.csv");
    write_layered_csv(&layered_csv, &layered)?;

    println!("\nheterogeneous stack along the propagation path:");
    for layer in &LAYERS {
        println!(
            "  {:<7} {:>4} cells   alpha0 = {:.2} dB/cm/MHz^gamma   gamma = {:.2}",
            layer.name, layer.cells, layer.alpha0_db, layer.gamma
        );
    }
    println!("\nrecovered vs the exact path-weighted prediction:");
    println!(
        "{:>12} {:>18} {:>14} {:>12}",
        "freq [MHz]", "path-weighted", "measured", "rel_err"
    );
    for &(frequency_hz, predicted, measured) in &layered {
        println!(
            "{:>12.2} {predicted:>18.3} {measured:>14.3} {:>12.4}",
            frequency_hz / 1.0e6,
            (measured - predicted).abs() / predicted
        );
    }

    println!("\npng: {}", sweep_png.display());
    println!("csv: {}", sweep_csv.display());
    println!("csv: {}", layered_csv.display());
    Ok(())
}
