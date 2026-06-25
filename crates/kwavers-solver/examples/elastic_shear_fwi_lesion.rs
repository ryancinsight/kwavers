//! Elastic shear-wave FWI — stiff-lesion reconstruction (ADR 033, Ch11 fig07).
//!
//! Runs the adjoint-state elastic full-waveform inversion
//! ([`kwavers_solver::inverse::elastography::elastic_fwi`]) on a synthetic
//! stiff-inclusion phantom and writes the true / initial / recovered shear-
//! modulus maps as CSV for the Ch11 figure script
//! `crates/kwavers-python/examples/book/ch10_elastic_fwi_lesion.py`.
//!
//! Physics runs in Rust; the Python script only plots the exported maps
//! (the book's "physics in Rust, plotting in Python" contract).
//!
//! Run: `cargo run -p kwavers-solver --release --example elastic_shear_fwi_lesion`
//!
//! The medium is deliberately compressible (`c_P = √3·c_S`, Poisson ≈ 0.25) so
//! the elastic CFL (set by `c_P`) does not demand the ~10⁵ steps a realistic
//! near-incompressible `c_P/c_S ≈ 1000` would require; the FWI machinery is
//! identical (see ADR 033 / the elastic_fwi tests).

use kwavers_grid::Grid;
use kwavers_medium::homogeneous::HomogeneousMedium;
use kwavers_solver::inverse::elastography::elastic_fwi::{
    reconstruct_lesion_transmission, TransmissionFwiParams,
};
use ndarray::Array3;
use std::io::Write;

const NXY: usize = 36;
const DX: f64 = 1.0e-3;
const RHO: f64 = 1000.0;
const C_S: f64 = 2.0;
const C_P: f64 = 3.4641016; // √3·c_S  ⇒  λ = μ
const MU_BG: f64 = RHO * C_S * C_S; // 4000 Pa

fn dump(dir: &std::path::Path, name: &str, m: &Array3<f64>) {
    let mut file = std::fs::File::create(dir.join(format!("{name}.csv"))).expect("csv");
    for j in 0..NXY {
        let row: Vec<String> = (0..NXY).map(|i| format!("{:.2}", m[[i, j, 0]])).collect();
        writeln!(file, "{}", row.join(",")).expect("row");
    }
}

fn main() {
    let grid = Grid::new(NXY, NXY, 1, DX, DX, DX).expect("grid");
    let medium = HomogeneousMedium::elastic_homogeneous(RHO, C_P, C_S, &grid).expect("medium");
    let mu_incl = 3.0 * MU_BG;

    // True model: stiff disk (radius 5) at the centre; background = medium μ.
    let c = NXY as f64 / 2.0;
    let mu_initial = Array3::from_elem(grid.dimensions(), MU_BG);
    let mut mu_true = mu_initial.clone();
    for i in 0..NXY {
        for j in 0..NXY {
            if (i as f64 - c).hypot(j as f64 - c) <= 5.0 {
                mu_true[[i, j, 0]] = mu_incl;
            }
        }
    }

    let params = TransmissionFwiParams {
        n_steps: 200,
        iterations: 20,
        ..TransmissionFwiParams::default()
    };
    println!(
        "running elastic shear-wave FWI ({NXY}×{NXY}, {} steps)…",
        params.n_steps
    );
    let mu_rec =
        reconstruct_lesion_transmission(&grid, &medium, &mu_true, &params).expect("reconstruct");

    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/book_data/elastic_fwi");
    std::fs::create_dir_all(&dir).expect("mkdir");
    dump(&dir, "mu_true", &mu_true);
    dump(&dir, "mu_initial", &mu_initial);
    dump(&dir, "mu_recovered", &mu_rec);
    std::fs::write(
        dir.join("meta.csv"),
        format!("dx_m,{DX}\nmu_bg_pa,{MU_BG}\nmu_incl_pa,{mu_incl}\nnxy,{NXY}\n"),
    )
    .expect("meta");
    println!("wrote CSVs to {}", dir.display());
}
