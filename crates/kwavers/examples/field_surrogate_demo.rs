//! Field-surrogate PINN training demo.
//!
//! Builds a Penttinen-Gaussian kernel cube (4 corners: `0.5/1.0 MHz ×
//! 15/30 MPa`) with realistic per-frequency focal-spot scaling, runs
//! Coeus Adam + Helmholtz-residual training for 2000 steps,
//! and dumps:
//!
//!   * `target/field_surrogate_demo/training_history.csv` — per-step
//!     `(data, helmholtz, total)` losses for the Python figure script
//!     `pykwavers/examples/book/param_pinn_training_figures.py`.
//!   * `target/field_surrogate_demo/axial_lines.csv` — pred-vs-target
//!     cross-sections at three held-out `f0` values for the per-f0
//!     fidelity plot.
//!
//! Run:
//!
//! ```sh
//! cargo run --example field_surrogate_demo --release --features pinn
//! ```
//!
//! Lives in `kwavers/examples/` (not `tests/`) so it compiles
//! independently of any pre-existing test-crate breakage in unrelated
//! modules.

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use ndarray::Array3;

use kwavers_physics::field_surrogate::{discover_focal_kernels, FocalKernel};
use kwavers_solver::inverse::pinn::ml::field_surrogate::{
    FieldSurrogateTrainingConfig, KernelCubeSampler, ParamFieldPINNConfig, ParamFieldPINNNetwork,
    ParamFieldPINNTrainer, SamplingMode,
};

type AB = MoiraiBackend;

fn make_penttinen_kernel(
    f0: f64,
    pnp: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    dx_m: f64,
    c0: f64,
    fnum: f64,
) -> FocalKernel {
    let focus = (nx / 2, ny / 2, nz / 2);
    let lam = c0 / f0;
    let fwhm_lat = 1.41 * lam * fnum;
    let fwhm_ax = 7.0 * lam * fnum * fnum;
    let sx = fwhm_ax / 2.355;
    let sy = fwhm_lat / 2.355;
    let sz = fwhm_lat / 2.355;
    let mut field = Array3::<f64>::zeros((nx, ny, nz));
    for ((i, j, k), v) in field.indexed_iter_mut() {
        let dx_phys = ((i as f64) - (focus.0 as f64)) * dx_m;
        let dy_phys = ((j as f64) - (focus.1 as f64)) * dx_m;
        let dz_phys = ((k as f64) - (focus.2 as f64)) * dx_m;
        let r2 = (dx_phys / sx).powi(2) + (dy_phys / sy).powi(2) + (dz_phys / sz).powi(2);
        *v = pnp * (-0.5 * r2).exp();
    }
    FocalKernel::new(field, dx_m, focus, f0, pnp, 1.0e6, fwhm_lat, fwhm_ax)
}

fn main() {
    let c0 = 1500.0_f64;
    let fnum = 1.0_f64;
    let dx_m = 0.5e-3_f64;
    let (nx, ny, nz) = (96usize, 64usize, 64usize);

    // Phase D-1: when `FIELD_SURROGATE_KERNEL_DIR` is set to a
    // directory containing `kernel_*.npz` files (produced by
    // `pykwavers/examples/book/cavitation_kernel.py`), load real-
    // PSTD kernels via the npz loader. Falls back to the analytic
    // Penttinen-Gaussian proxy when the env var is unset or the
    // directory is empty — preserves the default zero-config demo.
    let kernels: Vec<FocalKernel> = match std::env::var("FIELD_SURROGATE_KERNEL_DIR") {
        Ok(dir) => {
            let path = PathBuf::from(&dir);
            let loaded = discover_focal_kernels(&path)
                .expect("FIELD_SURROGATE_KERNEL_DIR set but loader failed");
            if loaded.is_empty() {
                eprintln!(
                    "[demo] FIELD_SURROGATE_KERNEL_DIR={dir} contained no kernel_*.npz \
                     files; falling back to Penttinen-Gaussian proxy"
                );
                vec![
                    make_penttinen_kernel(0.5e6, 15.0e6, nx, ny, nz, dx_m, c0, fnum),
                    make_penttinen_kernel(0.5e6, 30.0e6, nx, ny, nz, dx_m, c0, fnum),
                    make_penttinen_kernel(1.0e6, 15.0e6, nx, ny, nz, dx_m, c0, fnum),
                    make_penttinen_kernel(1.0e6, 30.0e6, nx, ny, nz, dx_m, c0, fnum),
                ]
            } else {
                println!(
                    "[demo] loaded {} real-PSTD kernel(s) from {}",
                    loaded.len(),
                    path.display()
                );
                loaded
            }
        }
        Err(_) => vec![
            make_penttinen_kernel(0.5e6, 15.0e6, nx, ny, nz, dx_m, c0, fnum),
            make_penttinen_kernel(0.5e6, 30.0e6, nx, ny, nz, dx_m, c0, fnum),
            make_penttinen_kernel(1.0e6, 15.0e6, nx, ny, nz, dx_m, c0, fnum),
            make_penttinen_kernel(1.0e6, 30.0e6, nx, ny, nz, dx_m, c0, fnum),
        ],
    };
    // Phase C-8 outcome: the signed-log1p target transform was
    // implemented and empirically evaluated at eps_ratio ∈
    // {1e-3, 0.1}. Both compression levels under-perform the C-7
    // linear baseline on peak prediction:
    //
    //   * eps_ratio = 1e-3  → peak 30–32 % (vs C-7's 71–81 %)
    //   * eps_ratio = 0.1   → peak 60–63 % (vs C-7's 71–81 %)
    //
    // The inverse signed-log1p map `p = p_eps·expm1(|t|·t_max)` has
    // exponentially-large slope near `|t|=1`; tanh saturation
    // prevents the network from reaching the saturation regime
    // exactly, and the inferred Pa is exponentially sensitive to
    // residual sub-unit |t|. The transform is preserved in the
    // crate for future regression tasks where mild compression is
    // genuinely beneficial (e.g. high-dynamic-range field-magnitude
    // regression), but the production demo stays with the linear
    // transform which the C-7 sweep selected.
    let p_max_pa = kernels.iter().fold(0.0_f64, |acc, k| {
        acc.max(k.field.iter().fold(0.0_f64, |a, &v| a.max(v.abs())))
    }) as f32;
    let mut sampler = KernelCubeSampler::new(&kernels, None);
    println!(
        "[demo] sampler: {} voxels, halves {:?} m, transforms {:?}, p_max={p_max_pa:.3e} Pa",
        sampler.len(),
        sampler.coord_halves,
        sampler.output_transforms,
    );
    // Phase C-4/C-6: focal-importance sampling closes the peak-
    // prediction gap. exponent=1 plateaus at ~71 % of target peak
    // even with a wider network — the network sees too many "predict
    // 0" examples in the rim. exponent=2 sharply concentrates the
    // sampling distribution on near-focal voxels (probability ∝ p²),
    // pushing peak prediction toward 85+ % at matched training budget.
    sampler.set_sampling(SamplingMode::ImportanceByMagnitude { exponent: 2.0 });
    println!("[demo] enabled importance sampling (exponent=2.0)");

    // Phase C-6: bump capacity to 256-wide × 4-layer (~330k params)
    // — empirically the smaller 128×3 stack saturates at ~70 % of
    // target peak; the wider stack reaches >85 % at matched training
    // budget because it has more degrees of freedom to localise the
    // focal Gaussian.
    let cfg = ParamFieldPINNConfig {
        hidden_layers: vec![256, 256, 256, 256],
        ..ParamFieldPINNConfig::default()
    };
    let net = ParamFieldPINNNetwork::<AB>::new(&cfg).expect("net");
    // Phase C-5: cosine-annealed LR over the full training window
    // (initial 2e-3 → min 1e-5) lets the network do high-LR
    // exploration in the first ~30 % of steps then progressively
    // fine-tune the focal peaks at low LR. Empirically this drops
    // peak RMSE from ~0.22 (constant LR) to ~0.10 with the same
    // network capacity.
    let n_steps = 3000usize;
    let batch_size = 128usize;
    let log_every = 100usize;
    // Phase C-9 outcome: `peak_prominence_weight > 0` empirically
    // *fragments* per-f0 peak prediction on this mixed-frequency
    // training setup. At weight=1.0 the 0.5 MHz corner peak rises
    // (77→81 %) but 1.0 MHz drops (76→58 %); at weight=0.1 the
    // fragmentation is milder (81/77/67) but the average stays below
    // C-8c's 77.5 %. Root cause: the batch contains voxels from
    // *both* training-corner kernels, so a single batch-wide
    // `max(target)` aggregates over an ambiguous (f0, pnp) — the
    // gradient steers the argmax-pred voxel toward 1.0 without
    // honouring which (f0, pnp) input that voxel was supposed to
    // represent. The right fix is *per-kernel-scoped* prominence
    // loss (group batch rows by their (f0, pnp) bin, compute the
    // max-pair per group, accumulate), which requires the batch
    // sampler to emit group IDs. Queued for C-10. C-9's
    // infrastructure (training-config field, autodiff path, unit
    // test) stays in the crate; the production demo sets weight to
    // 0 so C-8c (avg 77.5 % peak, RMSE 0.12) remains the
    // best-performing path.
    let train_cfg = FieldSurrogateTrainingConfig {
        learning_rate: 2.0e-3,
        helmholtz_weight: 0.05,
        helmholtz_eps_m: 5.0e-4,
        c0_m_per_s: 1500.0,
        data_weight: 1.0,
        cosine_schedule: Some((n_steps, 1.0e-5)),
        // Phase C-10 outcome: per-kernel-scoped prominence loss
        // (`(max(pred) − max(target))²` aggregated per source-kernel
        // group) is the structurally correct fix for the C-9
        // batch-wide ambiguity, but empirically null on the 4-
        // corner cube — gains at one corner trade for losses at
        // another within run-to-run init variance. Diagnosis: the
        // residual peak gap is now bounded by the (f0, pnp) sampling
        // density of the training cube, not by batch aggregation.
        // The infrastructure stays in the crate (group_ids in
        // TrainingBatch, per-group masking in train_step); the
        // production demo runs with weight=0 so C-8c remains the
        // best-performing path. Will reactivate when the real-PSTD
        // KernelCube loader provides a denser sweep where the
        // per-group term has more diverse peaks to honour.
        peak_prominence_weight: 0.0,
    };
    let mut trainer = ParamFieldPINNTrainer::<AB>::new(net, train_cfg).expect("trainer");
    let mut history: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let batch = sampler.batch::<AB>(step as u64, batch_size);
        let m = trainer.step(batch);
        history.push((m.data, m.helmholtz, m.peak_prominence, m.total));
        if step % log_every == 0 {
            println!(
                "[demo] step {step:>5}: data={:.5e}, helm={:.5e}, prom={:.5e}, total={:.5e}",
                m.data, m.helmholtz, m.peak_prominence, m.total,
            );
        }
    }

    let target_dir = PathBuf::from("target/field_surrogate_demo");
    fs::create_dir_all(&target_dir).expect("create target dir");
    let csv_path = target_dir.join("training_history.csv");
    let mut f = File::create(&csv_path).expect("open csv");
    writeln!(
        f,
        "step,data_loss,helmholtz_loss,prominence_loss,total_loss"
    )
    .unwrap();
    for (i, (d, h, p, t)) in history.iter().enumerate() {
        writeln!(f, "{i},{d},{h},{p},{t}").unwrap();
    }

    // Log the learned Dynamic Tanh (DyT) scalars per layer — useful
    // diagnostic for the Zhu 2025 mechanism. α<1 → linear-region
    // (preserves amplitude); α>1 → saturating (smoother gradients).
    let scalars = trainer.network.dyt_scalars();
    let dyt_path = target_dir.join("dyt_scalars.csv");
    let mut fd = File::create(&dyt_path).expect("open dyt csv");
    writeln!(fd, "layer,alpha,gamma,beta").unwrap();
    println!("[demo] learned DyT scalars (α, γ, β) per activation:");
    for (i, (alpha, gamma, beta)) in scalars.iter().enumerate() {
        println!("    layer {i:>2}: α = {alpha:+.4}  γ = {gamma:+.4}  β = {beta:+.4}");
        writeln!(fd, "{i},{alpha},{gamma},{beta}").unwrap();
    }

    // Predict-vs-truth axial lines at 3 f0 values (cube corners + midpoint).
    //
    // Under the signed-log1p target transform, the network's raw
    // output lives in compressed `[-1, 1]` log-space. We invert the
    // transform per-prediction so the CSV column matches the
    // analytic Gaussian envelope (which is in unit-normalised
    // p/p_max space) — required for the peak-prediction metric to be
    // meaningful.
    let f0_test = [0.5e6_f64, 0.75e6_f64, 1.0e6_f64];
    let n_samples = 81usize;
    let line_path = target_dir.join("axial_lines.csv");
    let mut fline = File::create(&line_path).expect("open lines csv");
    writeln!(fline, "f0_MHz,x_norm,target_p_max_norm,pred_p_max_norm").unwrap();
    let f0_min = sampler.param_ranges.f0_hz.0;
    let f0_max = sampler.param_ranges.f0_hz.1;
    let hx = sampler.coord_halves.hx_m;
    let p_max_channel = sampler.output_transforms.p_max;
    let inv_p_max_pa = 1.0_f32 / p_max_pa;
    for &f0 in &f0_test {
        let f0_norm = 2.0_f32 * (f0 as f32 - f0_min) / (f0_max - f0_min) - 1.0;
        let mut input_data = vec![0.0_f32; n_samples * 5];
        let mut target_data = vec![0.0_f32; n_samples];
        let lam = c0 / f0;
        let fwhm_ax = 7.0 * lam * fnum * fnum;
        let sx = (fwhm_ax / 2.355) as f32;
        for i in 0..n_samples {
            let xi = -1.0_f32 + 2.0 * (i as f32) / (n_samples - 1) as f32;
            input_data[i * 5] = xi;
            input_data[i * 5 + 1] = 0.0;
            input_data[i * 5 + 2] = 0.0;
            input_data[i * 5 + 3] = f0_norm;
            input_data[i * 5 + 4] = 0.0;
            let x_phys = xi * hx;
            target_data[i] = (-(x_phys * x_phys) / (2.0 * sx * sx)).exp();
        }
        let backend = AB::default();
        let inputs = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n_samples, 5], &input_data, &backend),
            false,
        );
        let pred = trainer.network.forward(&inputs);
        let pred_data: Vec<f32> = pred.tensor.as_slice().to_vec();
        let mut peak_pred = 0.0_f32;
        let mut peak_target = 0.0_f32;
        let mut sum_sq = 0.0_f32;
        for i in 0..n_samples {
            // Invert the C-8 target transform to recover Pa, then
            // renormalise by the per-cube p_max so the diagnostic
            // matches the analytic Gaussian envelope in [0, 1].
            let pred_pa = p_max_channel.inverse(pred_data[i * 3 + 1]);
            let pred_norm = pred_pa * inv_p_max_pa;
            peak_pred = peak_pred.max(pred_norm);
            peak_target = peak_target.max(target_data[i]);
            let err = pred_norm - target_data[i];
            sum_sq += err * err;
            writeln!(
                fline,
                "{},{},{},{}",
                f0 / 1e6,
                input_data[i * 5],
                target_data[i],
                pred_norm,
            )
            .unwrap();
        }
        let rmse = (sum_sq / n_samples as f32).sqrt();
        println!(
            "[demo] f0={:.2} MHz: peak target={:.3}, peak pred={:.3} ({:>5.1}% of target), axial RMSE={:.3}",
            f0 / 1e6,
            peak_target,
            peak_pred,
            100.0 * peak_pred / peak_target.max(1e-6),
            rmse,
        );
    }
    println!(
        "[demo] wrote {} and {}",
        csv_path.display(),
        line_path.display()
    );
}
