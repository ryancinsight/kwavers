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
//! The draw and the losses are asserted separately, because they failed
//! separately. A first version compared per-epoch losses bitwise and failed:
//! two runs at one seed gave 35.74074 and 35.74508 while the draw was
//! identical, so something downstream of the draw was order-dependent
//! (KW-PINN-NONDETERMINISTIC-REDUCTION). It no longer is, and
//! `losses_are_reproducible_at_one_seed` below holds the property that
//! regression would break.

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

/// One seed must produce one loss sequence, bitwise.
///
/// This is the property KW-PINN-NONDETERMINISTIC-REDUCTION was filed against:
/// with the draw pinned, two runs still disagreed in the fourth significant
/// figure and diverged further each epoch. Floating-point addition is not
/// associative, so a reduction whose split varied between runs would produce
/// exactly that, and a seed cannot control it.
///
/// It holds now. The configuration below is chosen to keep it meaningful
/// rather than to make it cheap: 10,000 collocation points is the default and
/// is wide enough that a reduction over them is split, which is where an
/// order-dependent sum would show. The network is small only because its width
/// costs time without widening the reduction under test.
///
/// A failure here is a real loss of reproducibility, not a flaky test: a seeded
/// run that does not replay cannot bisect a training regression, and every
/// two-run comparison inherits the disagreement as a noise floor -- about 1e-4
/// relative when this was last measured, larger than many changes worth
/// measuring.
#[test]
fn losses_are_reproducible_at_one_seed() -> KwaversResult<()> {
    /// Every per-epoch loss component of one short training run.
    fn losses() -> KwaversResult<Vec<[f64; 5]>> {
        let config = PinnConfig3D {
            hidden_layers: vec![16, 16],
            num_collocation_points: 10_000,
            collocation_seed: 20_260_826,
            boundary_seed: 20_260_826,
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;
        let mut solver = PinnWave3D::<TestBackend>::new(config, geometry, wave_speed)?;

        let n = 64;
        let xs: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 1e-3).collect();
        let ts: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-3).collect();
        let us: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 1e-3).collect();

        let m = solver.train(&xs, &xs, &xs, &ts, &us, None, 2)?;
        Ok((0..m.total_loss.len())
            .map(|e| {
                [
                    m.total_loss[e],
                    m.data_loss[e],
                    m.pde_loss[e],
                    m.bc_loss[e],
                    m.ic_loss[e],
                ]
            })
            .collect())
    }

    const COMPONENTS: [&str; 5] = ["total", "data", "pde", "bc", "ic"];

    let reference = losses()?;
    assert!(!reference.is_empty(), "the run reported no epochs");

    for run in 1..3 {
        let repeat = losses()?;
        assert_eq!(
            repeat.len(),
            reference.len(),
            "run {run} completed {} epochs against the reference run's {}",
            repeat.len(),
            reference.len()
        );
        for (epoch, (got, want)) in repeat.iter().zip(&reference).enumerate() {
            for (i, name) in COMPONENTS.iter().enumerate() {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "epoch {epoch} {name} loss differs between two runs at one \
                     seed: {:.9e} then {:.9e}. The draw is asserted identical \
                     above, so this is order dependence downstream of it \
                     (KW-PINN-NONDETERMINISTIC-REDUCTION)",
                    want[i],
                    got[i],
                );
            }
        }
    }
    Ok(())
}
