#[cfg(feature = "pinn")]
use burn::backend::{Autodiff, NdArray};
#[cfg(feature = "pinn")]
use burn::tensor::Tensor;
#[cfg(feature = "pinn")]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(feature = "pinn")]
use kwavers::solver::inverse::pinn::ml::AdaptiveCollocationSampler;
#[cfg(feature = "pinn")]
use std::time::Duration;

#[cfg(feature = "pinn")]
type Backend = Autodiff<NdArray<f32>>;

#[cfg(feature = "pinn")]
fn benchmark_split_coordinates(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_sampling_split");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let sizes = vec![1000, 10000];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Create random points tensor [size, 3]
        let device = Default::default();
        let points_data: Vec<f32> = (0..size * 3).map(|_| rand::random::<f32>()).collect();
        let points = Tensor::<Backend, 1>::from_floats(points_data.as_slice(), &device)
            .reshape([size, 3]);

        group.bench_with_input(
            BenchmarkId::new("slow_cpu", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let (x, y, t) = split_coordinates_slow(black_box(&points));
                    black_box((x, y, t));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fast_gpu", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let (x, y, t) = split_coordinates_fast(black_box(&points));
                    black_box((x, y, t));
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "pinn")]
fn split_coordinates_slow(
    points: &Tensor<Backend, 2>,
) -> (Tensor<Backend, 1>, Tensor<Backend, 1>, Tensor<Backend, 1>) {
    let points_data = points.clone().to_data();
    let values = points_data.to_vec::<f32>().unwrap_or_default();
    let total_points = points.shape().dims[0];
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

    let device = points.device();
    let x = Tensor::<Backend, 1>::from_floats(x_coords.as_slice(), &device);
    let y = Tensor::<Backend, 1>::from_floats(y_coords.as_slice(), &device);
    let t = Tensor::<Backend, 1>::from_floats(t_coords.as_slice(), &device);

    (x, y, t)
}

#[cfg(feature = "pinn")]
fn split_coordinates_fast(
    points: &Tensor<Backend, 2>,
) -> (Tensor<Backend, 1>, Tensor<Backend, 1>, Tensor<Backend, 1>) {
    let total_points = points.shape().dims[0];
    let x = points.clone().slice([0..total_points, 0..1]).flatten(0, 1);
    let y = points.clone().slice([0..total_points, 1..2]).flatten(0, 1);
    let t = points.clone().slice([0..total_points, 2..3]).flatten(0, 1);
    (x, y, t)
}

#[cfg(feature = "pinn")]
criterion_group!(benches, benchmark_split_coordinates);
#[cfg(feature = "pinn")]
criterion_main!(benches);

#[cfg(not(feature = "pinn"))]
fn main() {
    println!("PINN feature not enabled");
}
