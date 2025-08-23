# Kwavers: Acoustic Wave Simulation Library

[![Rust](https://img.shields.io/badge/rust-1.89%2B-blue.svg)](https://www.rust-lang.org)
[![Build](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/kwavers/kwavers)
[![Tests](https://img.shields.io/badge/tests-16%2F337_files-red.svg)](./tests)
[![Warnings](https://img.shields.io/badge/warnings-431-red.svg)](./src)
[![Grade](https://img.shields.io/badge/grade-D-red.svg)](./PRD.md)

## Acoustic Wave Simulation Library - Major Refactoring Required

A comprehensive acoustic wave simulation library that works but has significant architectural and quality issues that must be addressed before production use.

### Current Status (v2.15.0)
- **Build**: ✅ Compiles (with 431 warnings)
- **Tests**: ❌ Only 16 tests for 337 source files (0.05 tests/file)
- **Examples**: ✅ 7 examples work
- **Warnings**: ❌ 431 (unacceptable)
- **Code Quality**: Grade D - Poor architecture
- **Production Ready**: ❌ No - requires major refactoring

## Critical Issues

### Architecture Violations ❌
- **20+ modules exceed 700 lines** (many near 1000)
- **Massive SRP violations** - modules doing too many things
- **Poor separation of concerns**
- **Over-engineered plugin system**
- **Insufficient modularity**

### Code Quality Problems ❌
- **431 warnings** - mostly unused code indicating poor API design
- **0.05 tests per file** - essentially untested
- **Large monolithic modules** - hard to maintain
- **Excessive complexity** - violates KISS principle
- **Poor naming** - many generic names

### Testing Disaster ❌
- **16 tests for 337 files** - 95% of code untested
- **No integration tests**
- **No performance tests**
- **No stress tests**
- **Minimal validation**

## What Works (Barely)

### Core Features
- FDTD solver (functional but bloated)
- PSTD solver (works but untested)
- Basic physics (correct but unvalidated)
- Examples run (but don't prove much)

## Engineering Assessment

### Violations of Best Practices
- **SOLID**: ❌ Single Responsibility violated everywhere
- **DRY**: ❌ Code duplication throughout
- **KISS**: ❌ Over-engineered complexity
- **YAGNI**: ❌ Tons of unused code
- **Clean Code**: ❌ 431 warnings, huge modules

### Technical Debt Score: 9/10 (Critical)
- Immediate refactoring required
- Not suitable for production
- High maintenance burden
- Poor testability
- Excessive complexity

## Required Actions

### Immediate (Before ANY Production Use)
1. **Split all modules >500 lines** (20+ violations)
2. **Fix all 431 warnings** - no excuses
3. **Add proper tests** - minimum 1 test per public function
4. **Remove dead code** - tons of unused functionality
5. **Simplify architecture** - plugin system is over-engineered

### Short Term (1-2 weeks)
1. **Achieve 80% test coverage**
2. **Reduce warnings to <50**
3. **Document all public APIs**
4. **Profile and optimize**
5. **Add integration tests**

### Long Term (1 month)
1. **Complete architectural refactor**
2. **Implement proper error handling**
3. **Add benchmarks**
4. **Create comprehensive examples**
5. **Achieve 90% test coverage**

## Usage Warning ⚠️

**DO NOT USE IN PRODUCTION**

This library is not production-ready. It has:
- Insufficient testing
- Poor architecture
- High technical debt
- Excessive warnings
- Unvalidated physics

Use only for:
- Research prototypes (with caution)
- Educational purposes (as example of what not to do)
- Development (if you're willing to refactor)

## Honest Metrics

| Metric | Value | Acceptable | Status |
|--------|-------|------------|--------|
| **Warnings** | 431 | <50 | ❌ FAIL |
| **Tests/File** | 0.05 | >1 | ❌ FAIL |
| **Module Size** | 20+ >700 lines | <500 | ❌ FAIL |
| **Test Coverage** | ~5% | >80% | ❌ FAIL |
| **Code Quality** | D | B+ | ❌ FAIL |

## Contributing

This project needs major help. Priority contributions:
1. **Testing** - We need hundreds of tests
2. **Refactoring** - Split large modules
3. **Warning fixes** - Clean up the 431 warnings
4. **Documentation** - Most code is undocumented
5. **Architecture** - Simplify over-engineered design

## License

MIT License (but please don't use this in production yet)

## Brutal Truth

**Grade: D** - This codebase is a mess. It technically works but violates nearly every software engineering principle. The architecture is poor, testing is essentially non-existent, and the code quality is unacceptable with 431 warnings.

### The Good
- It compiles
- Basic functionality works
- Physics appears correct

### The Bad
- 431 warnings
- 20+ massive modules
- Only 16 tests for 337 files
- Over-engineered design
- Poor separation of concerns

### The Ugly
- 0.05 tests per file
- Modules with 1000+ lines
- Untested critical paths
- Technical debt everywhere

**Recommendation**: This library requires a complete refactor before any production use. The current state is unacceptable for any serious application.
