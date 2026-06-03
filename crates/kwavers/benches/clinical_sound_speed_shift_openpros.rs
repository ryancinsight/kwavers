//! Dense-versus-sparse OpenPros-style speed-shift reconstruction benchmark.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers_diagnostics::reconstruction::sound_speed_shift::{
    openpros_shift_benchmark_case, OpenProsShiftBenchmarkConfig, SoundSpeedShiftPlan,
    SoundSpeedShiftWorkspace,
};

fn clinical_sound_speed_shift_openpros(c: &mut Criterion) {
    let config = OpenProsShiftBenchmarkConfig::default();
    let case = openpros_shift_benchmark_case(&config).unwrap();
    let dense_plan =
        SoundSpeedShiftPlan::new(case.samples.clone(), &case.active_mask, case.dense_config)
            .unwrap();
    let sparse_plan =
        SoundSpeedShiftPlan::new(case.samples.clone(), &case.active_mask, case.sparse_config)
            .unwrap();
    let frame = case.frame_time_shifts_s;
    let mut dense_workspace = SoundSpeedShiftWorkspace::new();
    let mut sparse_workspace = SoundSpeedShiftWorkspace::new();
    let mut group = c.benchmark_group("clinical_sound_speed_shift_openpros");

    group.bench_function("dense_fixed_plan", |b| {
        b.iter(|| {
            let image = dense_plan
                .reconstruct_with_workspace(black_box(&frame), &mut dense_workspace)
                .unwrap();
            black_box(image.rows_used)
        });
    });
    group.bench_function("sparse_fixed_plan", |b| {
        b.iter(|| {
            let image = sparse_plan
                .reconstruct_with_workspace(black_box(&frame), &mut sparse_workspace)
                .unwrap();
            black_box(image.rows_used)
        });
    });
    group.finish();
}

criterion_group!(benches, clinical_sound_speed_shift_openpros);
criterion_main!(benches);
