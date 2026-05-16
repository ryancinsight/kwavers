use super::super::{AnatomyKind, Point3};
use super::adjoint::{gradient, GradientInput};
use super::encoding::SourceEncoding;
use super::forward::{
    forward_dense_history_for_test, forward_with_schedule, replay_history_segment, ForwardInput,
    ReplayInput, TimeSchedule,
};
use super::stencil::sponge;
use super::types::{GridIndex, Nonlinear3dAperture, Nonlinear3dConfig};

#[test]
fn checkpoint_replay_matches_dense_forward_history_bitwise() {
    let fixture = fixture(3);
    let checkpointed = forward_with_schedule(fixture.forward_input(true));
    let history = checkpointed
        .history
        .as_ref()
        .expect("checkpointed forward must retain history");
    let dense = forward_dense_history_for_test(fixture.forward_input(false));
    assert_eq!(history.interval(), 3);
    assert_eq!(history.checkpoint_count(), 3);

    for step in 0..=fixture.schedule.time_steps {
        let replay_step = step.min(fixture.schedule.time_steps - 1);
        let segment = replay_history_segment(fixture.replay_input(history, replay_step));
        let expected = dense_state(&dense, step, fixture.cells);
        let actual = segment.state(step);
        for (cell, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "state mismatch at step={step}, cell={cell}: actual={a}, expected={e}"
            );
        }
    }
}

#[test]
fn checkpointed_gradient_is_interval_invariant() {
    let left = fixture(2);
    let right = fixture(5);
    let left_forward = forward_with_schedule(left.forward_input(true));
    let right_forward = forward_with_schedule(right.forward_input(true));
    let residual = left_forward
        .traces
        .iter()
        .enumerate()
        .map(|(idx, value)| 0.07 * value + ((idx + 1) as f64 * 0.11).sin())
        .collect::<Vec<_>>();
    let observed_energy = residual
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .max(1.0e-24);
    let left_gradient = gradient(left.gradient_input(
        left_forward.history.as_ref().expect("left history"),
        &residual,
        observed_energy,
    ));
    let right_gradient = gradient(right.gradient_input(
        right_forward.history.as_ref().expect("right history"),
        &residual,
        observed_energy,
    ));

    assert_eq!(
        left_gradient.sound_speed.len(),
        right_gradient.sound_speed.len()
    );
    assert_eq!(left_gradient.beta.len(), right_gradient.beta.len());
    for (cell, (a, e)) in left_gradient
        .sound_speed
        .iter()
        .zip(right_gradient.sound_speed.iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "sound-speed gradient changed with checkpoint interval at cell={cell}: {a} != {e}"
        );
    }
    for (cell, (a, e)) in left_gradient
        .beta
        .iter()
        .zip(right_gradient.beta.iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "beta gradient changed with checkpoint interval at cell={cell}: {a} != {e}"
        );
    }
}

struct Fixture {
    speed: Vec<f64>,
    density: Vec<f64>,
    beta: Vec<f64>,
    attenuation_alpha0: Vec<f64>,
    attenuation_y: Vec<f64>,
    body: Vec<bool>,
    sponge: Vec<f64>,
    aperture: Nonlinear3dAperture,
    config: Nonlinear3dConfig,
    schedule: TimeSchedule,
    n: usize,
    cells: usize,
    spacing_m: f64,
    encoding: SourceEncoding,
}

impl Fixture {
    fn forward_input(&self, retain_history: bool) -> ForwardInput<'_> {
        ForwardInput {
            speed: &self.speed,
            density: &self.density,
            beta: &self.beta,
            attenuation_np_per_m_mhz: Some(&self.attenuation_alpha0),
            attenuation_power_law_y: Some(&self.attenuation_y),
            n: self.n,
            spacing_m: self.spacing_m,
            aperture: &self.aperture,
            config: &self.config,
            schedule: self.schedule,
            encoding: self.encoding,
            source_scale: 1.0,
            retain_history,
        }
    }

    fn replay_input<'a>(
        &'a self,
        history: &'a super::checkpoint::ForwardHistory,
        step: usize,
    ) -> ReplayInput<'a> {
        ReplayInput {
            history,
            speed: &self.speed,
            density: &self.density,
            beta: &self.beta,
            attenuation_np_per_m_mhz: Some(&self.attenuation_alpha0),
            attenuation_power_law_y: Some(&self.attenuation_y),
            n: self.n,
            spacing_m: self.spacing_m,
            aperture: &self.aperture,
            config: &self.config,
            schedule: self.schedule,
            encoding: self.encoding,
            source_scale: 1.0,
            sponge: &self.sponge,
            step,
        }
    }

    fn gradient_input<'a>(
        &'a self,
        history: &'a super::checkpoint::ForwardHistory,
        residual: &'a [f64],
        observed_energy: f64,
    ) -> GradientInput<'a> {
        GradientInput {
            history,
            cells: self.cells,
            residual,
            speed: &self.speed,
            density: &self.density,
            beta: &self.beta,
            attenuation_np_per_m_mhz: Some(&self.attenuation_alpha0),
            attenuation_power_law_y: Some(&self.attenuation_y),
            body: &self.body,
            n: self.n,
            spacing_m: self.spacing_m,
            aperture: &self.aperture,
            config: &self.config,
            schedule: self.schedule,
            encoding: self.encoding,
            source_scale: 1.0,
            dt: self.schedule.dt_s,
            observed_energy,
        }
    }
}

fn fixture(checkpoint_interval_steps: usize) -> Fixture {
    let n = 6;
    let cells = n * n * n;
    let speed = (0..cells)
        .map(|i| 1470.0 + 2.0 * (i % 11) as f64)
        .collect::<Vec<_>>();
    let density = (0..cells)
        .map(|i| 995.0 + (i % 7) as f64)
        .collect::<Vec<_>>();
    let beta = (0..cells)
        .map(|i| 3.4 + 0.05 * (i % 5) as f64)
        .collect::<Vec<_>>();
    // Mild but nonzero soft-tissue absorption so the absorption operator is
    // exercised in the checkpoint replay/transpose-adjoint paths and the
    // bit-exact gradient invariance still holds.
    let attenuation_alpha0 = (0..cells)
        .map(|i| 4.0 + 0.5 * (i % 3) as f64)
        .collect::<Vec<_>>();
    let attenuation_y = vec![1.05_f64; cells];
    let body = (0..cells)
        .map(|i| {
            let z = i % n;
            let y = (i / n) % n;
            let x = i / (n * n);
            x > 0 && y > 0 && z > 0 && x + 1 < n && y + 1 < n && z + 1 < n
        })
        .collect::<Vec<_>>();
    let aperture = aperture();
    let mut config = Nonlinear3dConfig::new(AnatomyKind::Liver);
    config.frequency_hz = 500_000.0;
    config.source_pressure_pa = 2.0e5;
    config.cycles = 2.5;
    config.cfl = 0.4;
    config.checkpoint_interval_steps = checkpoint_interval_steps;
    Fixture {
        speed,
        density,
        beta,
        attenuation_alpha0,
        attenuation_y,
        body,
        sponge: sponge(n),
        aperture,
        config,
        schedule: TimeSchedule {
            dt_s: 1.0e-7,
            time_steps: 9,
        },
        n,
        cells,
        spacing_m: 8.0e-4,
        encoding: SourceEncoding { index: 0, count: 1 },
    }
}

fn aperture() -> Nonlinear3dAperture {
    Nonlinear3dAperture {
        sources: vec![
            GridIndex { x: 1, y: 2, z: 2 },
            GridIndex { x: 1, y: 3, z: 3 },
        ],
        receivers: vec![
            GridIndex { x: 4, y: 2, z: 2 },
            GridIndex { x: 4, y: 3, z: 3 },
        ],
        therapy_points_m: vec![point(GridIndex { x: 1, y: 2, z: 2 })],
        receiver_points_m: vec![point(GridIndex { x: 4, y: 2, z: 2 })],
        model_name: "checkpoint_test_same_aperture".to_owned(),
        focus: GridIndex { x: 3, y: 3, z: 3 },
    }
}

fn point(index: GridIndex) -> Point3 {
    Point3 {
        x_m: index.x as f64,
        y_m: index.y as f64,
        z_m: index.z as f64,
    }
}

fn dense_state(history: &[f64], step: usize, cells: usize) -> &[f64] {
    let offset = step * cells;
    &history[offset..offset + cells]
}
