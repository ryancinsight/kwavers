//! Does the gradient reach the parameters, is it the right gradient, and does
//! the optimizer apply it?
//!
//! Training does not converge on a target the network represents exactly
//! (KW-PINN-3D-NO-CONVERGENCE): with every target set to `u = 0` the optimum is
//! the zero function, reachable by driving the output weights to zero, and 2000
//! epochs move the loss from 399.9 only to 224.8 while the per-epoch rate falls
//! tenfold. These tests separate the three places that can break, so the next
//! change is aimed rather than guessed:
//!
//! 1. the backward pass never reaches some parameters (gradient is absent),
//! 2. it reaches them with the wrong value (autograd disagrees with the loss),
//! 3. the value is right but the update does not land (optimizer plumbing).
//!
//! Inputs are fixed rather than drawn, because the solver's collocation points
//! come from an unseeded `rand::random()` and a finite difference across two
//! draws measures the draw, not the derivative.

use super::super::*;
use coeus_autograd::Var;
use kwavers_core::error::KwaversResult;

type TestBackend = coeus_core::MoiraiBackend;

/// A column vector `Var` that does not track gradients: it is data, not a
/// parameter.
fn column(backend: &TestBackend, values: &[f32]) -> Var<f32, TestBackend> {
    Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![values.len(), 1], values, backend),
        false,
    )
}

/// A small network and a fixed batch, sized so a finite difference is cheap.
fn fixture() -> KwaversResult<(
    PINN3DNetwork<TestBackend>,
    Vec<Var<f32, TestBackend>>,
    Var<f32, TestBackend>,
)> {
    let backend = TestBackend::default();
    let config = PinnConfig3D {
        hidden_layers: vec![4],
        ..Default::default()
    };
    let network = PINN3DNetwork::<TestBackend>::new(&config)?;
    let inputs = vec![
        column(&backend, &[0.25, 0.75]),
        column(&backend, &[0.5, 0.5]),
        column(&backend, &[0.5, 0.5]),
        column(&backend, &[0.1, 0.2]),
    ];
    let target = column(&backend, &[0.0, 0.0]);
    Ok((network, inputs, target))
}

/// Mean squared error of the network against the target, on the fixed batch.
///
/// This is the data term of the solver's loss. The PDE, boundary and initial
/// terms are deliberately excluded: they add derivative machinery of their own,
/// and the question here is whether a gradient reaches a weight at all, which
/// the simplest term answers without ambiguity. A weight the data term cannot
/// reach cannot be reached by the others either.
fn loss(
    network: &PINN3DNetwork<TestBackend>,
    inputs: &[Var<f32, TestBackend>],
    target: &Var<f32, TestBackend>,
) -> KwaversResult<Var<f32, TestBackend>> {
    let prediction = network.forward(&inputs[0], &inputs[1], &inputs[2], &inputs[3])?;
    let residual = coeus_autograd::sub(&prediction, target);
    Ok(coeus_autograd::mean(&coeus_autograd::mul(
        &residual, &residual,
    )))
}

fn scalar(v: &Var<f32, TestBackend>) -> f32 {
    v.tensor.as_slice()[0]
}

/// Every parameter must receive a finite gradient.
///
/// A parameter whose gradient is absent or zero for a batch that does not
/// already fit is one the optimizer can never move, and a network with such
/// parameters cannot reach an optimum that needs them.
#[test]
fn backward_reaches_every_parameter() -> KwaversResult<()> {
    let (network, inputs, target) = fixture()?;

    for p in network.parameters() {
        p.zero_grad();
    }
    let value = loss(&network, &inputs, &target)?;
    assert!(
        scalar(&value) > 0.0,
        "the fixture already fits the target, so no gradient is expected of it"
    );
    value
        .backward()
        .map_err(|e| kwavers_core::error::KwaversError::InternalError(format!("backward: {e}")))?;

    let mut unreached = Vec::new();
    for (index, p) in network.parameters().iter().enumerate() {
        match p.grad() {
            None => unreached.push(format!("p{index}: no gradient buffer")),
            Some(g) => {
                let slice = g.as_slice();
                if !slice.iter().all(|v| v.is_finite()) {
                    unreached.push(format!("p{index}: non-finite gradient"));
                } else if slice.iter().all(|v| *v == 0.0) {
                    unreached.push(format!("p{index}: gradient is exactly zero"));
                }
            }
        }
    }
    assert!(
        unreached.is_empty(),
        "backward did not deliver a usable gradient to {} of {} parameters: {}",
        unreached.len(),
        network.parameters().len(),
        unreached.join("; ")
    );
    Ok(())
}

/// The reported gradient must be the loss's actual derivative.
///
/// Central difference: `(L(w + h) - L(w - h)) / 2h` approximates `dL/dw` with
/// truncation error `O(h^2)`, against a floating-point cancellation error that
/// grows as `O(eps/h)`. In `f32` (`eps` about `1.2e-7`) the two balance near
/// `h = eps^(1/3)`, so `h = 5e-3` sits close to the optimum and the achievable
/// agreement is a few parts in a thousand -- not the `1e-6` an `f64` check
/// would reach. `2%` is loose enough for that noise and far tighter than any
/// structural error: a missing term, a wrong sign, or a detached graph moves
/// the ratio by tens of percent or more.
#[test]
fn reported_gradient_matches_a_finite_difference() -> KwaversResult<()> {
    const STEP: f32 = 5.0e-3;
    const RELATIVE_TOLERANCE: f32 = 2.0e-2;

    let (network, inputs, target) = fixture()?;

    for p in network.parameters() {
        p.zero_grad();
    }
    loss(&network, &inputs, &target)?
        .backward()
        .map_err(|e| kwavers_core::error::KwaversError::InternalError(format!("backward: {e}")))?;

    // The first weight with a gradient large enough that a finite difference
    // resolves it. Differencing a near-zero derivative measures rounding.
    let parameters = network.parameters();
    let (index, element, analytic) = parameters
        .iter()
        .enumerate()
        .find_map(|(index, p)| {
            let g = p.grad()?;
            let slice = g.as_slice();
            slice
                .iter()
                .enumerate()
                .find(|(_, v)| v.abs() > 1.0e-3)
                .map(|(element, v)| (index, element, *v))
        })
        .expect("at least one parameter carries a resolvable gradient");

    let mut shifted = |delta: f32| -> KwaversResult<f32> {
        let mut params = network.parameters();
        let mut values = params[index].tensor.as_slice().to_vec();
        values[element] += delta;
        let shape = params[index].tensor.shape().to_vec();
        params[index] = Var::new(
            coeus_tensor::Tensor::from_slice_on(shape, &values, &TestBackend::default()),
            true,
        );
        let mut probe = network.clone();
        probe.load_parameters(&params);
        Ok(scalar(&loss(&probe, &inputs, &target)?))
    };

    let numeric = (shifted(STEP)? - shifted(-STEP)?) / (2.0 * STEP);
    let relative = (numeric - analytic).abs() / analytic.abs().max(1.0e-6);

    assert!(
        relative < RELATIVE_TOLERANCE,
        "parameter p{index}[{element}]: autograd reports {analytic:.6e}, the loss's own \
         central difference gives {numeric:.6e} -- {:.1}% apart. The backward pass is \
         not differentiating the function the forward pass computes.",
        relative * 100.0
    );
    Ok(())
}

/// The optimizer must move each parameter by exactly `-lr * grad`.
///
/// `SimpleOptimizer3D` is plain SGD with no momentum, so the step is fully
/// determined. This is the third failure mode: a correct gradient that never
/// lands, which the parameter round-trip through `parameters()` and
/// `load_parameters()` makes possible -- tensor storage is copy-on-write, so a
/// clone detaches on first mutation and an update written to the wrong copy is
/// silently dropped. Nothing tested that round-trip; the existing optimizer
/// test asserts only that the layer count survives the step.
#[test]
fn the_optimizer_applies_the_gradient_it_was_given() -> KwaversResult<()> {
    const LEARNING_RATE: f32 = 0.1;
    // Two f32 roundings (the multiply and the subtract) against parameter
    // values of order one.
    const TOLERANCE: f32 = 1.0e-5;

    let (network, inputs, target) = fixture()?;

    for p in network.parameters() {
        p.zero_grad();
    }
    loss(&network, &inputs, &target)?
        .backward()
        .map_err(|e| kwavers_core::error::KwaversError::InternalError(format!("backward: {e}")))?;

    let before: Vec<Vec<f32>> = network
        .parameters()
        .iter()
        .map(|p| p.tensor.as_slice().to_vec())
        .collect();
    let gradients: Vec<Vec<f32>> = network
        .parameters()
        .iter()
        .map(|p| p.grad().map(|g| g.as_slice().to_vec()).unwrap_or_default())
        .collect();

    let updated = SimpleOptimizer3D::new(LEARNING_RATE).step(network)?;
    let after: Vec<Vec<f32>> = updated
        .parameters()
        .iter()
        .map(|p| p.tensor.as_slice().to_vec())
        .collect();

    let mut moved = 0usize;
    for (index, ((old, new), grad)) in before.iter().zip(&after).zip(&gradients).enumerate() {
        assert_eq!(
            old.len(),
            new.len(),
            "p{index} changed length across a step"
        );
        for (element, ((o, n), g)) in old.iter().zip(new).zip(grad).enumerate() {
            let expected = o - LEARNING_RATE * g;
            assert!(
                (n - expected).abs() <= TOLERANCE,
                "p{index}[{element}]: {o:.6} with gradient {g:.6e} at lr {LEARNING_RATE} \
                 should become {expected:.6}, but the step produced {n:.6}"
            );
            if (n - o).abs() > 0.0 {
                moved += 1;
            }
        }
    }
    assert!(
        moved > 0,
        "no parameter moved: the step is a no-op even though gradients were present"
    );
    Ok(())
}
