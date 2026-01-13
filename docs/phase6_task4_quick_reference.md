# Phase 6 Task 4: Quick Reference Guide

**Quick Start**: How to run Phase 6 integration tests and benchmarks

---

## 🚀 Fast Track

### Run Everything (45-60 minutes)

```bash
# Integration tests
cargo test --features pinn --test pinn_elastic_validation -- --ignored --nocapture

# Benchmarks
cargo bench --features pinn --bench phase6_persistent_adam_benchmarks
```

---

## 📋 Integration Tests

### All Task 4 Tests

```bash
cargo test --features pinn --test pinn_elastic_validation -- --ignored --nocapture test_persistent_adam_convergence_improvement test_checkpoint_resume_continuity test_performance_benchmarks test_multi_checkpoint_session
```

### Individual Tests

#### 1. Convergence Comparison (~5-10 min)

```bash
cargo test --features pinn --test pinn_elastic_validation test_persistent_adam_convergence_improvement -- --ignored --nocapture
```

**What it tests**: Persistent Adam vs stateless Adam convergence rate

**Expected output**:
```
=== Persistent Adam Training ===
Persistent Adam: final_loss=X.XXe-XX, epochs_to_target=XX/100
✓ Persistent Adam convergence validated
```

#### 2. Checkpoint Resume (~10-15 min)

```bash
cargo test --features pinn --test pinn_elastic_validation test_checkpoint_resume_continuity -- --ignored --nocapture
```

**What it tests**: Training interruption and resumption fidelity

**Expected output**:
```
=== Checkpoint Resume Test ===
Phase 1: Training first 50 epochs...
Saving checkpoint at epoch 50...
Phase 2: Loading checkpoint and resuming...
✓ Checkpoint save/load mechanics validated
```

#### 3. Performance Benchmarks (~3-5 min)

```bash
cargo test --features pinn --test pinn_elastic_validation test_performance_benchmarks -- --ignored --nocapture
```

**What it tests**: Checkpoint I/O performance and training throughput

**Expected output**:
```
=== Performance Benchmarks ===
Training throughput:
  Avg epoch time: X.XX ms
  Samples/sec: XXX.X
Checkpoint save performance:
  Avg time: X.XX ms
Checkpoint load performance:
  Avg time: X.XX ms
✓ All performance benchmarks passed
```

#### 4. Multi-Checkpoint Session (~3-5 min)

```bash
cargo test --features pinn --test pinn_elastic_validation test_multi_checkpoint_session -- --ignored --nocapture
```

**What it tests**: Multiple checkpoints and arbitrary epoch resumption

**Expected output**:
```
=== Multi-Checkpoint Session Test ===
Training to epoch 10...
  ✓ Checkpoint saved at epoch 10
...
✓ Multi-checkpoint session test passed
```

---

## 📊 Performance Benchmarks

### All Benchmarks

```bash
cargo bench --features pinn --bench phase6_persistent_adam_benchmarks
```

### Individual Benchmark Groups

#### 1. Adam Step Overhead

```bash
cargo bench --features pinn --bench phase6_persistent_adam_benchmarks -- adam_step_overhead
```

**Measures**: Persistent vs stateless Adam step time  
**Target**: < 5% overhead

#### 2. Checkpoint Save Performance

```bash
cargo bench --features pinn --bench phase6_persistent_adam_benchmarks -- checkpoint_save
```

**Measures**: Model serialization time  
**Target**: < 500ms (50k-200k params)

#### 3. Checkpoint Load Performance

```bash
cargo bench --features pinn --bench phase6_persistent_adam_benchmarks -- checkpoint_load
```

**Measures**: Model deserialization + reconstruction time  
**Target**: < 1s

#### 4. Training Epoch with Checkpoint

```bash
cargo bench --features pinn --bench phase6_persistent_adam_benchmarks -- training_epoch
```

**Measures**: Full epoch time with/without checkpoint  
**Target**: < 10% overhead

#### 5. Memory Overhead

```bash
cargo bench --features pinn --bench phase6_persistent_adam_benchmarks -- memory_overhead
```

**Measures**: Memory allocation (params + moments)  
**Target**: 3× model size

#### 6. Convergence Rate Comparison

```bash
cargo bench --features pinn --bench phase6_persistent_adam_benchmarks -- convergence_rate
```

**Measures**: Epochs to reach target loss  
**Target**: 20-40% improvement (persistent vs stateless)

---

## 🎯 Benchmark Options

### Save Baseline

```bash
cargo bench --features pinn --bench phase6_persistent_adam_benchmarks -- --save-baseline phase6_v1
```

### Compare Against Baseline

```bash
cargo bench --features pinn --bench phase6_persistent_adam_benchmarks -- --baseline phase6_v1
```

### Generate HTML Report

```bash
# Run benchmarks (generates report automatically)
cargo bench --features pinn --bench phase6_persistent_adam_benchmarks

# View report
# Open target/criterion/report/index.html in browser
```

---

## 📝 Validation Report

### After Running Tests

1. Open report template:
   ```bash
   # Windows
   notepad docs\phase6_task4_validation_report.md
   
   # Linux/macOS
   vim docs/phase6_task4_validation_report.md
   ```

2. Fill in "Actual" columns with test results

3. Complete analysis sections with observations

4. Save and commit completed report

---

## 🔧 Troubleshooting

### Tests Not Found

**Problem**: `cargo test` doesn't find Phase 6 tests

**Solution**: Ensure `pinn` feature is enabled:
```bash
cargo test --features pinn --test pinn_elastic_validation
```

### Tests Fail to Compile

**Problem**: Compilation errors in test file

**Solution**: Check Burn 0.19 is installed:
```bash
cargo tree | grep burn
# Should show burn = "0.19"
```

### Benchmarks Too Slow

**Problem**: Benchmarks take too long

**Solution**: Reduce sample size or measurement time:
```bash
# Edit benches/phase6_persistent_adam_benchmarks.rs
# Change: .sample_size(50) → .sample_size(10)
# Change: .measurement_time(Duration::from_secs(15)) → .measurement_time(Duration::from_secs(5))
```

### Checkpoint Directory Full

**Problem**: Test checkpoints accumulate in temp directories

**Solution**: Tests use `TempDir` which auto-cleans. If manual cleanup needed:
```bash
# Windows
rmdir /s /q %TEMP%\rust_*

# Linux/macOS
rm -rf /tmp/rust_*
```

### Out of Memory

**Problem**: Tests fail with OOM error

**Solution**: Run tests sequentially:
```bash
cargo test --features pinn --test pinn_elastic_validation -- --ignored --nocapture --test-threads=1
```

---

## 📈 Expected Results

### Integration Tests

| Test | Duration | Status | Key Metrics |
|------|----------|--------|-------------|
| Convergence comparison | 5-10 min | ✅ | Epochs to 1e-4: ~60-80 |
| Checkpoint resume | 10-15 min | ✅ | Loss continuity verified |
| Performance benchmarks | 3-5 min | ✅ | Save < 500ms, Load < 1s |
| Multi-checkpoint session | 3-5 min | ✅ | All checkpoints loadable |

### Benchmarks

| Benchmark | Target | Expected Range |
|-----------|--------|----------------|
| Adam step overhead | < 5% | 2-4% |
| Checkpoint save | < 500ms | 100-300ms (50k-200k params) |
| Checkpoint load | < 1s | 200-500ms (50k-200k params) |
| Memory overhead | 3× model | 3.0-3.2× (metadata included) |
| Convergence improvement | 20-40% | 25-35% typical |

---

## 🎓 Understanding Output

### Loss Values

- `1e-4` = Target loss (0.0001)
- `1e-6` = Excellent convergence
- `1e-2` = Acceptable for early training
- `> 1` = Poor convergence (investigate)

### Epoch Times

- `< 10 ms` = Small model (10k params)
- `10-50 ms` = Medium model (50k params)
- `50-200 ms` = Large model (200k params)
- `> 200 ms` = Very large model (500k+ params) or slow backend

### Checkpoint Sizes

- Small model (10k params): ~40 KB (model) + ~1 KB (config/metrics)
- Medium model (50k params): ~200 KB (model)
- Large model (200k params): ~800 KB (model)
- XLarge model (500k params): ~2 MB (model)

---

## 📚 Related Documentation

- **Task 4 Summary**: `docs/phase6_task4_summary.md`
- **Validation Report Template**: `docs/phase6_task4_validation_report.md`
- **Phase 6 Checklist**: `docs/phase6_checklist.md`
- **Checkpoint Implementation**: `docs/phase6_task2_checkpoint_implementation.md`
- **Build Fixes**: `docs/phase6_task3_build_fixes_summary.md`

---

## 🚦 Quick Status Check

### Is Task 4 Complete?

✅ **YES** if:
- All integration tests pass
- All benchmarks run successfully
- Results meet acceptance criteria
- Validation report filled out

⚠️ **PARTIAL** if:
- Tests pass but some metrics outside target range
- Repository-wide build issues prevent full test suite

❌ **NO** if:
- Tests fail to compile
- Tests fail at runtime
- Critical acceptance criteria not met

### Is Phase 6 Ready for Production?

✅ **YES** if:
- Task 4 complete
- All acceptance criteria met
- Known limitations documented
- Validation report approved

⚠️ **WITH RESERVATIONS** if:
- Optimizer state serialization still deferred
- Repository-wide build issues unresolved
- Performance metrics at lower end of targets

---

## 💡 Tips

1. **Run tests during low-activity periods** (computationally expensive)
2. **Use `--nocapture` flag** to see detailed progress
3. **Check disk space** before running (checkpoints accumulate)
4. **Save baseline** before making changes (for comparison)
5. **Run benchmarks multiple times** for statistical significance
6. **Document any anomalies** in validation report notes section

---

**Last Updated**: 2025-01-XX  
**Phase**: 6 - Persistent Adam Optimizer & Full Checkpointing  
**Task**: 4 - Integration & Validation Tests