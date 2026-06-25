//! Elastic shear-wave FWI — stiff-lesion reconstruction (ADR 033, Ch11 fig07).
//!
//! Runs the adjoint-state elastic full-waveform inversion
//! ([`kwavers_solver::inverse::elastography::elastic_fwi::ElasticFwi`]) on a
//! synthetic stiff-inclusion phantom and writes the true / initial / recovered
//! shear-modulus maps as CSV for the Ch11 figure script
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
use kwavers_solver::forward::elastic::swe::{
    ElasticPointForce, ElasticWaveConfig, ElasticWaveSolver,
};
use kwavers_solver::inverse::elastography::elastic_fwi::{ElasticFwi, ElasticFwiConfig};
use ndarray::Array3;
use std::io::Write;

const NXY: usize = 36;
const DX: f64 = 1.0e-3;
const RHO: f64 = 1000.0;
const C_S: f64 = 2.0;
const C_P: f64 = 3.4641016; // √3·c_S  ⇒  λ = μ
const MU_BG: f64 = RHO * C_S * C_S; // 4000 Pa
const N_STEPS: usize = 200;
const F0: f64 = 300.0;
const AMP: f64 = 1.0e7;

#[derive(Clone, Copy)]
enum Comp {
    X,
    Y,
}

fn ricker(index: (usize, usize, usize), dt: f64, comp: Comp) -> ElasticPointForce {
    let mut f = ElasticPointForce::zeros(index, N_STEPS);
    let t0 = 1.0 / F0;
    let a = std::f64::consts::PI * std::f64::consts::PI * F0 * F0;
    for n in 0..N_STEPS {
        let t = n as f64 * dt - t0;
        let arg = a * t * t;
        let v = AMP * (1.0 - 2.0 * arg) * (-arg).exp();
        match comp {
            Comp::X => f.fx[n] = v,
            Comp::Y => f.fy[n] = v,
        }
    }
    f
}

fn swe_config() -> ElasticWaveConfig {
    ElasticWaveConfig {
        time_step: 0.0,
        save_every: 1,
        pml_thickness: 6,
        ..ElasticWaveConfig::default()
    }
}

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

    // CFL-stable dt for the stiffest model (the inclusion).
    let dt = {
        let mut s = ElasticWaveSolver::new(&grid, &medium, swe_config()).expect("solver");
        s.set_mu(&Array3::from_elem(grid.dimensions(), mu_incl))
            .expect("set_mu");
        s.recommended_timestep(0.3)
    };

    // Crossed four-side transmission illumination.
    let mut source = Vec::new();
    let mut receivers = Vec::new();
    for r in (9..27).step_by(2) {
        source.push(ricker((7, r, 0), dt, Comp::Y)); // left
        source.push(ricker((28, r, 0), dt, Comp::Y)); // right
        source.push(ricker((r, 7, 0), dt, Comp::X)); // top
        source.push(ricker((r, 28, 0), dt, Comp::X)); // bottom
        receivers.push((9, r, 0));
        receivers.push((26, r, 0));
        receivers.push((r, 9, 0));
        receivers.push((r, 26, 0));
    }

    // True model: stiff disk (radius 5) at the centre.
    let c = NXY as f64 / 2.0;
    let mut mu_true = Array3::from_elem(grid.dimensions(), MU_BG);
    for i in 0..NXY {
        for j in 0..NXY {
            if ((i as f64 - c).powi(2) + (j as f64 - c).powi(2)).sqrt() <= 5.0 {
                mu_true[[i, j, 0]] = mu_incl;
            }
        }
    }

    let mut cfg = ElasticFwiConfig::new(N_STEPS, dt, receivers, source);
    cfg.iterations = 20;
    cfg.step_size = MU_BG;
    cfg.mu_min = 0.5 * MU_BG;
    cfg.mu_max = 5.0 * MU_BG;
    cfg.precond_eps = 0.1; // illumination preconditioner (contrast)
    cfg.mute_radius = 4; // suppress the near-acquisition strain imprint

    let observed =
        ElasticFwi::synthesize_observed(&grid, swe_config(), &medium, &mu_true, &cfg).expect("obs");
    let mu_initial = Array3::from_elem(grid.dimensions(), MU_BG);
    let mut fwi = ElasticFwi::new(
        &grid,
        swe_config(),
        &medium,
        mu_initial.clone(),
        observed,
        cfg,
    )
    .expect("fwi");
    println!("running elastic shear-wave FWI ({NXY}×{NXY}, {N_STEPS} steps)…");
    let mu_rec = fwi.run().expect("run");

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
