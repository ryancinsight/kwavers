//! The collocation draw must be reproducible from its seed.
//!
//! The PDE residual is evaluated at points drawn from the domain, so the loss
//! is a function of the draw as well as of the parameters. Those points came
//! from `rand::random()` -- the unseeded global generator -- so no run could be
//! replayed, no two runs of one configuration could be compared, and a training
//! failure could be described but not handed to anyone
//! (KW-PINN-UNSEEDED-RNG).
//!
//! The draw is rejection-sampled against the geometry, so an unseeded run
//! varied in the *number* of points as well as their positions.
//!
//! These tests assert on the points rather than on the losses. A first version
//! compared per-epoch losses bitwise and failed: two runs at one seed gave
//! 35.74074 and 35.74508. The draw was identical -- verified directly, and
//! asserted below -- so the difference is the parallel backend's reduction
//! order, which floating-point addition makes order-dependent and which no
//! seed controls. Bitwise agreement there needs a deterministic reduction
//! mode, filed separately as KW-PINN-NONDETERMINISTIC-REDUCTION. Asserting it
//! here would have been a test that fails for a reason it does not name.

use super::super::*;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::KwaversResult;

type TestBackend = coeus_core::MoiraiBackend;

/// Every collocation coordinate a solver at `seed` draws, and the seed its
/// metrics report.
fn draw_for(seed: u64) -> KwaversResult<(Vec<f32>, u64)> {
    let config = PinnConfig3D {
        hidden_layers: vec![8],
        num_collocation_points: 64,
        collocation_seed: seed,
        ..Default::default()
    };
    let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;
    let mut solver = PinnWave3D::<TestBackend>::new(config, geometry, wave_speed)?;

    let owned = solver.config.clone();
    let (x, y, z, t) = solver.generate_collocation_points(&owned);
    let mut points = Vec::new();
    for tensor in [&x, &y, &z, &t] {
        points.extend_from_slice(tensor.tensor.as_slice());
    }

    // One epoch is enough to see what the result carries; the losses are not
    // compared, for the reason in the module documentation.
    let metrics = solver.train(&[0.5], &[0.5], &[0.5], &[0.1], &[0.0], None, 1)?;
    Ok((points, metrics.collocation_seed))
}

/// The same seed must draw the same points, and the run must report that seed.
#[test]
fn one_seed_draws_the_same_points() -> KwaversResult<()> {
    let (first, first_seed) = draw_for(20_260_826)?;
    let (second, second_seed) = draw_for(20_260_826)?;

    assert!(!first.is_empty(), "the sampler produced no points");
    assert_eq!(
        first.len(),
        second.len(),
        "the same seed drew {} coordinates and then {}; the draw is \
         rejection-sampled against the geometry, so its length varies with it",
        first.len(),
        second.len()
    );
    assert_eq!(first, second, "the same seed drew different points");
    assert_eq!(
        (first_seed, second_seed),
        (20_260_826, 20_260_826),
        "the metrics must carry the seed that produced the run"
    );
    Ok(())
}

/// A different seed must draw different points.
///
/// Without this the test above would pass against a sampler that ignored the
/// seed entirely -- a fixed grid, say, which is reproducible and also not what
/// this field claims to control.
#[test]
fn a_different_seed_draws_different_points() -> KwaversResult<()> {
    let (first, _) = draw_for(1)?;
    let (second, _) = draw_for(2)?;

    assert_ne!(
        first, second,
        "two seeds drew identical points, so the seed does not reach the draw"
    );
    Ok(())
}
