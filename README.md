# Kwavers: Acoustic Wave Simulation Library

[![Lines](https://img.shields.io/badge/lines-93k-red.svg)](./src)
[![Tests](https://img.shields.io/badge/tests-16-red.svg)](./tests)
[![Test Coverage](https://img.shields.io/badge/coverage-0.02%25-red.svg)](./tests)
[![Warnings](https://img.shields.io/badge/warnings-431-orange.svg)](./src)
[![Grade](https://img.shields.io/badge/grade-C--minus-yellow.svg)](./PRD.md)

## 93,000 Lines of Under-Tested Code

A massive acoustic wave simulation library that technically works but represents a significant maintenance burden. With 93k lines and only 16 tests, this is a textbook example of unchecked growth.

### The Numbers Don't Lie
- **93,062 lines** of Rust code
- **16 tests** total (0.02% coverage by line count)
- **337 source files** (0.05 tests per file)
- **431 warnings** (121 items never used)
- **20+ modules >700 lines** (largest: 1097)
- **457 potential panic points** (unwrap/expect)

## Brutal Assessment

### What Actually Works
- Core FDTD/PSTD solvers function
- Examples run (some timeout)
- Physics calculations appear correct
- No runtime panics in happy path

### The Real Problems
1. **Untested**: 0.02% test coverage is negligent
2. **Bloated**: 93k lines for what should be 20-30k
3. **Unmaintainable**: Files with 1000+ lines
4. **Dead Code**: 121 unused items
5. **No Integration Tests**: Only unit tests exist

## Architecture Analysis

### Largest Offenders (lines)
1. `flexible_transducer.rs` - 1097
2. `kwave_utils.rs` - 976  
3. `hybrid/validation.rs` - 960
4. `transducer_design.rs` - 957
5. `spectral_dg/dg_solver.rs` - 943

These files are doing too much and violate every principle of modular design.

### Code Smell Metrics
- **Functions per file**: ~15-20 (should be <10)
- **Panic points**: 457 (should be <50)
- **Result types**: 1146 (good error handling at least)
- **Unused code**: 121 items (13% waste)

## Testing Reality

```
Tests:     16
Files:     337
Coverage:  0.02%
```

This is not "limited coverage" - this is essentially **untested code**. Any claim of "validated physics" is based on faith, not evidence.

## Risk Assessment

### Critical Risks
- **Correctness**: Unverified beyond happy path
- **Stability**: 457 panic points waiting to explode
- **Performance**: Unknown, unmeasured, unoptimized
- **Security**: Unaudited 93k lines
- **Maintenance**: Nightmare scenario

### Use At Your Own Risk
- Research: Maybe, with extensive validation
- Production: Absolutely not
- Mission-critical: Never
- Commercial: Legal liability

## The Hard Truth

This codebase is the result of:
1. **No code reviews** - How else do you get 1000+ line files?
2. **No refactoring** - Technical debt never paid
3. **No testing culture** - 16 tests for 93k lines
4. **Feature creep** - Everything added, nothing removed
5. **Academic coding** - Works once, ships forever

## What Should Be Done

### Option 1: Salvage Operation
1. Delete 50% of unused code
2. Split every file >500 lines
3. Add 500+ tests minimum
4. Profile and optimize
5. Document everything

**Time estimate**: 6-12 months

### Option 2: Strategic Rewrite
1. Extract core algorithms (10-15k lines)
2. Rewrite with TDD
3. Proper architecture from start
4. Maintain feature parity
5. Deprecate this version

**Time estimate**: 3-6 months

### Option 3: Abandon
1. Mark as unmaintained
2. Extract useful algorithms
3. Start fresh with lessons learned
4. Don't repeat mistakes

**Time estimate**: 0 months

## For Potential Users

**DO NOT** use this in production. This is research code that grew without supervision. It may produce correct results in tested scenarios, but with 0.02% test coverage, most code paths are unverified.

### If You Must Use It
1. Write your own tests for your use case
2. Profile everything
3. Have a backup plan
4. Don't trust the results without validation
5. Consider alternatives

## For Contributors

Before contributing:
1. **Don't add features** - Fix what exists
2. **Add tests** - Every PR must increase coverage
3. **Delete code** - Remove more than you add
4. **Split files** - Nothing over 500 lines
5. **Document why** - Not what, but why

## Engineering Verdict

**Grade: C-** (Generous)

This is 93,000 lines of barely-tested, poorly-structured code that happens to work sometimes. It's a liability, not an asset. The physics might be correct, but without tests, that's just a hypothesis.

The honest recommendation: **Extract the core algorithms and start over.**

---

*"The most dangerous code is code that appears to work."* - Every senior engineer ever
