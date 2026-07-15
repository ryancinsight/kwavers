use super::lr_schedule::MetaLrSchedule;
use super::meta_optimizer::MetaOptimizer;

#[test]
fn test_meta_optimizer_creation() {
    let optimizer = MetaOptimizer::new(0.001, 10);
    assert_eq!(optimizer.learning_rate(), 0.001);
    assert_eq!(optimizer.iteration_count(), 0);
}

#[test]
#[should_panic(expected = "Learning rate must be positive")]
fn test_meta_optimizer_invalid_lr() {
    let _ = MetaOptimizer::new(-0.001, 10);
}

#[test]
fn test_meta_optimizer_with_momentum() {
    let optimizer = MetaOptimizer::with_momentum(0.001, 10, 0.9);
    assert_eq!(optimizer.learning_rate(), 0.001);
    assert_eq!(optimizer._momentum, Some(0.9));
}

#[test]
fn test_meta_optimizer_with_adam() {
    let optimizer = MetaOptimizer::with_adam(0.001, 10, 0.9, 0.999, 1e-8);
    assert_eq!(optimizer.learning_rate(), 0.001);
    assert_eq!(optimizer._beta1, 0.9);
    assert_eq!(optimizer._beta2, 0.999);
    assert_eq!(optimizer._epsilon, 1e-8);
}

#[test]
fn test_set_learning_rate() {
    let mut optimizer = MetaOptimizer::new(0.001, 10);
    optimizer.set_learning_rate(0.0005);
    assert_eq!(optimizer.learning_rate(), 0.0005);
}

#[test]
fn test_decay_learning_rate() {
    let mut optimizer = MetaOptimizer::new(0.001, 10);
    optimizer.decay_learning_rate(0.5);
    assert!((optimizer.learning_rate() - 0.0005).abs() < 1e-10);
}

#[test]
fn test_reset_optimizer() {
    let mut optimizer = MetaOptimizer::new(0.001, 10);
    optimizer._iteration_count = 100;
    optimizer.reset();
    assert_eq!(optimizer.iteration_count(), 0);
}

#[test]
fn test_lr_schedule_constant() {
    let schedule = MetaLrSchedule::Constant;
    assert_eq!(schedule.get_lr(0, 0.001), 0.001);
    assert_eq!(schedule.get_lr(100, 0.001), 0.001);
}

#[test]
fn test_lr_schedule_step_decay() {
    let schedule = MetaLrSchedule::StepDecay {
        factor: 0.5,
        step_size: 10,
    };
    assert_eq!(schedule.get_lr(0, 0.001), 0.001);
    assert_eq!(schedule.get_lr(10, 0.001), 0.0005);
    assert_eq!(schedule.get_lr(20, 0.001), 0.00025);
}

#[test]
fn test_lr_schedule_exponential() {
    let schedule = MetaLrSchedule::Exponential { decay_rate: 0.01 };
    let lr0 = schedule.get_lr(0, 0.001);
    let lr100 = schedule.get_lr(100, 0.001);
    assert!((lr0 - 0.001).abs() < 1e-10);
    assert!(lr100 < lr0);
}

#[test]
fn test_lr_schedule_cosine_annealing() {
    let schedule = MetaLrSchedule::CosineAnnealing {
        lr_min: 0.0001,
        total_epochs: 1000,
    };
    let lr0 = schedule.get_lr(0, 0.001);
    let lr_mid = schedule.get_lr(500, 0.001);
    let lr_end = schedule.get_lr(1000, 0.001);

    assert!((lr0 - 0.001).abs() < 1e-10);
    assert!(lr_mid > 0.0001 && lr_mid < 0.001);
    assert!((lr_end - 0.0001).abs() < 1e-6);
}

#[test]
fn test_adam_step_updates_parameters() {
    let mut optimizer = MetaOptimizer::with_adam(0.01, 1, 0.9, 0.999, 1e-8);

    let mut params = vec![vec![0.0_f32]];
    let grads = vec![Some(vec![1.0_f32])];

    optimizer.step(&mut params, &grads);
    assert!(params[0][0] < 0.0);
}

#[test]
fn test_rmsprop_step_updates_parameters() {
    let mut optimizer = MetaOptimizer::with_rmsprop(0.01, 1, 0.9, 1e-8);

    let mut params = vec![vec![0.0_f32]];
    let grads = vec![Some(vec![1.0_f32])];

    optimizer.step(&mut params, &grads);
    assert!(params[0][0] < 0.0);
}
