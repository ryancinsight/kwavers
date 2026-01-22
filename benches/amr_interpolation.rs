use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::solver::utilities::amr::{Octree, ConservativeInterpolator};
use kwavers::domain::grid::Bounds;
use ndarray::Array3;

fn bench_interpolation(c: &mut Criterion) {
    // Setup
    let bounds = Bounds::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let mut octree = Octree::new(bounds, 3).unwrap();
    // Force refinement
    // Using a markers array that triggers refinement.
    // Based on exploration, check_refinement_marker checks global indices against bounds.
    // 16x16x16 markers.
    let markers = Array3::from_elem((16, 16, 16), 1i8);
    let _ = octree.update_refinement(&markers);

    // Field size
    let field = Array3::from_elem((64, 64, 64), 1.0);
    let interpolator = ConservativeInterpolator::new();

    let mut group = c.benchmark_group("AMR Interpolation");

    group.bench_function("interpolate_to_refined", |b| {
        b.iter(|| {
            interpolator.interpolate_to_refined(black_box(&octree), black_box(&field)).unwrap()
        })
    });

    let mut buffer = Array3::zeros(field.dim());
    group.bench_function("interpolate_into (reuse)", |b| {
        b.iter(|| {
            interpolator.interpolate_into(black_box(&octree), black_box(&field), black_box(&mut buffer)).unwrap()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_interpolation);
criterion_main!(benches);
