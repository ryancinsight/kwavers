#[cfg(feature = "pinn")]
use coeus_core::MoiraiBackend;
#[cfg(feature = "pinn")]
use coeus_tensor::Tensor;
#[cfg(feature = "pinn")]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(feature = "pinn")]
use std::time::Duration;

#[cfg(feature = "pinn")]
type Backend = MoiraiBackend;

#[cfg(feature = "pinn")]
fn benchmark_split_coordinates(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_sampling_split");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let sizes = vec![1000, 10000];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Create random points tensor [size, 3]
        let backend = Backend::default();
        let points_data: Vec<f32> = (0..size * 3).map(|_| rand::random::<f32>()).collect();
        let points = Tensor::<f32, Backend>::from_slice_on(vec![size, 3], &points_data, &backend);

        group.bench_with_input(BenchmarkId::new("slow_cpu", size), &size, |b, _| {
            b.iter(|| {
                let (x, y, t) = split_coordinates_slow(black_box(&points), &backend);
                black_box((x, y, t));
            });
        });

        group.bench_with_input(BenchmarkId::new("fast_gpu", size), &size, |b, _| {
            b.iter(|| {
                let (x, y, t) = split_coordinates_fast(black_box(&points));
                black_box((x, y, t));
            });
        });
    }

    group.finish();
}

#[cfg(feature = "pinn")]
fn split_coordinates_slow(
    points: &Tensor<f32, Backend>,
    backend: &Backend,
) -> (
    Tensor<f32, Backend>,
    Tensor<f32, Backend>,
    Tensor<f32, Backend>,
) {
    let values = points.to_contiguous();
    let values = values.as_slice();
    let total_points = points.shape()[0];
    let mut x_coords = Vec::with_capacity(total_points);
    let mut y_coords = Vec::with_capacity(total_points);
    let mut t_coords = Vec::with_capacity(total_points);

    for chunk in values.chunks(3) {
        if chunk.len() == 3 {
            x_coords.push(chunk[0]);
            y_coords.push(chunk[1]);
            t_coords.push(chunk[2]);
        }
    }

    let x = Tensor::from_slice_on(vec![total_points], &x_coords, backend);
    let y = Tensor::from_slice_on(vec![total_points], &y_coords, backend);
    let t = Tensor::from_slice_on(vec![total_points], &t_coords, backend);

    (x, y, t)
}

#[cfg(feature = "pinn")]
fn split_coordinates_fast(
    points: &Tensor<f32, Backend>,
) -> (
    Tensor<f32, Backend>,
    Tensor<f32, Backend>,
    Tensor<f32, Backend>,
) {
    let total_points = points.shape()[0];
    let x = points
        .slice(&[(0, total_points), (0, 1)])
        .to_contiguous()
        .reshape(vec![total_points]);
    let y = points
        .slice(&[(0, total_points), (1, 2)])
        .to_contiguous()
        .reshape(vec![total_points]);
    let t = points
        .slice(&[(0, total_points), (2, 3)])
        .to_contiguous()
        .reshape(vec![total_points]);
    (x, y, t)
}

#[cfg(feature = "pinn")]
criterion_group!(benches, benchmark_split_coordinates);
#[cfg(feature = "pinn")]
criterion_main!(benches);
