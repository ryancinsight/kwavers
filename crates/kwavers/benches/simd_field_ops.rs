//! SIMD field-operation benchmarks.
//!
//! Measures the production [`SimdOps`] element-wise kernels over contiguous
//! Leto arrays. ISA selection remains inside the production implementation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kwavers_math::simd_safe::SimdOps;
use leto::Array3;

fn bench_simd_field_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_field_operations");

    for size in [1_000, 10_000, 100_000] {
        let lhs = Array3::from_elem((10, 10, size), 1.0_f64);
        let rhs = Array3::from_elem((10, 10, size), 2.0_f64);

        group.bench_with_input(
            BenchmarkId::new("add", size),
            &(&lhs, &rhs),
            |b, &(lhs, rhs)| {
                b.iter(|| black_box(SimdOps::add_fields(lhs, rhs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("multiply", size),
            &(&lhs, &rhs),
            |b, &(lhs, rhs)| {
                b.iter(|| black_box(SimdOps::multiply_fields(lhs, rhs)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_simd_field_operations);
criterion_main!(benches);
