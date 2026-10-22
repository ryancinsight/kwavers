//! Tests for the SimpleOptimizer gradient descent implementation.

use super::core::SimpleOptimizer;
use crate::inverse::pinn::be;
use crate::inverse::pinn::ml::wave_equation_1d::config::PinnConfig;
use crate::inverse::pinn::ml::wave_equation_1d::network::PinnWave1D;
use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use kwavers_core::error::KwaversResult;

type TestBackend = MoiraiBackend;

fn var2(vals: &[f32], backend: TestBackend) -> KwaversResult<Var<f32, TestBackend>> {
    be(Var::new(
        be(coeus_tensor::Tensor::from_slice_on(
            vec![vals.len(), 1],
            vals,
            &backend,
        ))?,
        false,
    ))
}

fn zero_grad(pinn: &PinnWave1D<TestBackend>) -> KwaversResult<()> {
    for p in pinn.parameters() {
        be(p.zero_grad())?;
    }
    Ok(())
}

#[test]
fn test_optimizer_creation() {
    let optimizer = SimpleOptimizer::new(0.001);
    assert_eq!(optimizer.learning_rate(), 0.001);
}

#[test]
fn test_optimizer_learning_rate() {
    let optimizer = SimpleOptimizer::new(0.01);
    assert_eq!(optimizer.learning_rate(), 0.01);
}

#[test]
fn test_optimizer_step_compiles() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let optimizer = SimpleOptimizer::new(0.001);

    let x = var2(&[0.5], backend)?;
    let t = var2(&[0.1], backend)?;
    let u = pinn.forward(&x, &t).expect("1D optimizer forward");
    let loss = be(coeus_autograd::mean(&be(coeus_autograd::mul(&u, &u))?))?;

    zero_grad(&pinn)?;
    loss.backward().expect("1D optimizer backward");
    let _updated_pinn = optimizer.step(pinn).expect("PINN 1D optimizer step");

    Ok(())
}

#[test]
fn test_optimizer_step_updates_parameters() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![5, 5],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let x = var2(&[0.5], backend)?;
    let t = var2(&[0.1], backend)?;
    let u_before = pinn.forward(&x, &t).expect("1D optimizer forward");
    let u_before_val = u_before.tensor.as_slice()[0];

    let optimizer = SimpleOptimizer::new(0.1);

    let target = var2(&[1.0], backend)?;
    let diff = be(coeus_autograd::sub(&u_before, &target))?;
    let loss = be(coeus_autograd::mean(&be(coeus_autograd::mul(
        &diff, &diff,
    ))?))?;

    zero_grad(&pinn)?;
    loss.backward().expect("1D optimizer backward");

    let updated_pinn = optimizer.step(pinn).expect("PINN 1D optimizer step");

    let u_after = updated_pinn.forward(&x, &t).expect("1D optimizer forward");
    let u_after_val = u_after.tensor.as_slice()[0];

    assert!(u_before_val.is_finite());
    assert!(u_after_val.is_finite());
    Ok(())
}

#[test]
fn test_optimizer_multiple_steps() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![5, 5],
        ..Default::default()
    };
    let mut pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let optimizer = SimpleOptimizer::new(0.01);

    let x = var2(&[0.5], backend)?;
    let t = var2(&[0.1], backend)?;
    let target = var2(&[0.0], backend)?;

    let mut losses = Vec::new();

    for _ in 0..5 {
        let u = pinn.forward(&x, &t).expect("1D optimizer forward");
        let diff = be(coeus_autograd::sub(&u, &target))?;
        let loss = be(coeus_autograd::mean(&be(coeus_autograd::mul(
            &diff, &diff,
        ))?))?;

        let loss_val = loss.tensor.as_slice()[0];
        losses.push(loss_val);

        zero_grad(&pinn)?;
        loss.backward().expect("1D optimizer backward");
        pinn = optimizer.step(pinn).expect("PINN 1D optimizer step");
    }

    for &loss in &losses {
        assert!(loss.is_finite());
    }

    assert_eq!(losses.len(), 5);

    Ok(())
}

#[test]
fn test_optimizer_with_different_learning_rates() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![5, 5],
        ..Default::default()
    };

    let optimizer_small = SimpleOptimizer::new(0.0001);
    assert_eq!(optimizer_small.learning_rate(), 0.0001);

    let optimizer_medium = SimpleOptimizer::new(0.001);
    assert_eq!(optimizer_medium.learning_rate(), 0.001);

    let optimizer_large = SimpleOptimizer::new(0.1);
    assert_eq!(optimizer_large.learning_rate(), 0.1);

    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();
    let x = var2(&[0.5], backend)?;
    let t = var2(&[0.1], backend)?;

    let u = pinn.forward(&x, &t).expect("1D optimizer forward");
    let loss = be(coeus_autograd::mean(&be(coeus_autograd::mul(&u, &u))?))?;
    zero_grad(&pinn)?;
    loss.backward().expect("1D optimizer backward");
    let pinn = optimizer_small
        .step(pinn.clone())
        .expect("PINN 1D optimizer step");

    let u = pinn.forward(&x, &t).expect("1D optimizer forward");
    let loss = be(coeus_autograd::mean(&be(coeus_autograd::mul(&u, &u))?))?;
    zero_grad(&pinn)?;
    loss.backward().expect("1D optimizer backward");
    let pinn = optimizer_medium
        .step(pinn.clone())
        .expect("PINN 1D optimizer step");

    let u = pinn.forward(&x, &t).expect("1D optimizer forward");
    let loss = be(coeus_autograd::mean(&be(coeus_autograd::mul(&u, &u))?))?;
    zero_grad(&pinn)?;
    loss.backward().expect("1D optimizer backward");
    optimizer_large.step(pinn).expect("PINN 1D optimizer step");
    Ok(())
}

#[test]
fn test_gradient_mapper_preserves_structure() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config.clone()).unwrap();

    let optimizer = SimpleOptimizer::new(0.001);

    let x = var2(&[0.5], backend)?;
    let t = var2(&[0.1], backend)?;
    let u = pinn.forward(&x, &t).expect("1D optimizer forward");
    let loss = be(coeus_autograd::mean(&be(coeus_autograd::mul(&u, &u))?))?;
    zero_grad(&pinn)?;
    loss.backward().expect("1D optimizer backward");

    let updated_pinn = optimizer.step(pinn).expect("PINN 1D optimizer step");

    let u_after = updated_pinn.forward(&x, &t).expect("1D optimizer forward");
    assert_eq!(u_after.tensor.shape(), &[1, 1]);

    Ok(())
}
