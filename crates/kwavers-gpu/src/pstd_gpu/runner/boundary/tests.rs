use super::stage;
use kwavers_boundary::cpml::{CPMLConfig, CPMLProfiles};
use kwavers_grid::Grid;
use tracing_subscriber::fmt::format::FmtSpan;

fn trace_test() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    )
}

#[test]
fn graded_fields_match_the_axis_profiles_bitwise() {
    let _trace = trace_test();
    let shape = [11, 9, 7];
    let [nx, ny, nz] = shape;
    let total = nx * ny * nz;
    let dt = 4.0e-8;
    let profiles = tracing::info_span!("arrange").in_scope(|| {
        let grid = Grid::new(nx, ny, nz, 1.0e-3, 1.5e-3, 2.0e-3).expect("valid grid");
        let config = CPMLConfig::with_per_dimension_thickness(3, 2, 2);
        CPMLProfiles::new(&config, &grid, 1500.0, dt).expect("valid boundary profiles")
    });

    let staged = tracing::info_span!("act").in_scope(|| stage(shape, total, dt, Some(&profiles)));
    tracing::info_span!("assert").in_scope(|| {
        let fields = staged.as_arrays();
        let collocated = [fields.x, fields.y, fields.z];
        let staggered = [fields.sgx, fields.sgy, fields.sgz];
        let [dx, dy, dz] = [&profiles.sigma_x, &profiles.sigma_y, &profiles.sigma_z];
        let [vx, vy, vz] = [
            &profiles.sigma_x_sgx,
            &profiles.sigma_y_sgy,
            &profiles.sigma_z_sgz,
        ];
        for (axis, (den, vel)) in [dx, dy, dz].into_iter().zip([vx, vy, vz]).enumerate() {
            assert_eq!(collocated[axis].len(), total);
            assert_eq!(staggered[axis].len(), total);
            for flat in 0..total {
                let coordinate = [flat / (ny * nz), (flat / nz) % ny, flat % nz][axis];
                let attenuation = |sigma: f64| ((-sigma * dt * 0.5).exp() as f32).to_bits();
                assert_eq!(
                    collocated[axis][flat].to_bits(),
                    attenuation(den[coordinate])
                );
                assert_eq!(
                    staggered[axis][flat].to_bits(),
                    attenuation(vel[coordinate])
                );
            }
            assert!(collocated[axis].iter().any(|&value| value < 1.0));
        }
    });
}

#[test]
fn identity_fields_borrow_one_volume() {
    let _trace = trace_test();
    let shape = tracing::info_span!("arrange").in_scope(|| [7, 5, 3]);
    let total = shape.iter().product();
    let staged = tracing::info_span!("act").in_scope(|| stage(shape, total, 1.0e-7, None));
    tracing::info_span!("assert").in_scope(|| {
        let fields = staged.as_arrays();
        assert_eq!(fields.x.len(), total);
        assert!(fields.x.iter().all(|&value| value == 1.0));
        for field in [fields.y, fields.z, fields.sgx, fields.sgy, fields.sgz] {
            assert!(std::ptr::eq(fields.x, field));
        }
    });
}
