# Phase 1: Critical Fixes Implementation Guide
**Priority**: 🔴 CRITICAL - Must complete before other work  
**Timeline**: 1-2 days  
**Expected Result**: Clean compilation, zero blockers

---

## Pre-Implementation Checklist

- [ ] Read this entire document
- [ ] Run `cargo build --all-targets 2>&1` to see current errors
- [ ] Create a backup branch: `git branch backup/pre-phase1`
- [ ] Ready to commit after each fix
- [ ] All work on `main` branch (as required)

---

## Step 1: Identify All Compilation Errors

**Command**:
```bash
cargo build --all-targets 2>&1 | grep "^error\["
```

**Expected Output** (example):
```
error[E0432]: unresolved import `kwavers::ml::pinn::*`
error[E0433]: cannot find `AIIntegration` in module `clinical`
error[E0308]: mismatched types
... (more errors)
```

**Record**:
- Write down exact error messages
- Note file paths and line numbers
- Group by module/category

---

## Step 2: Fix PINN Import Paths

**Issue**: Tests/benchmarks import from `kwavers::ml::pinn::*` but module is at `kwavers::solver::inverse::pinn::ml`

**Root Cause**: Module reorganization in Phase 2 refactor

**Search for all occurrences**:
```bash
grep -r "use kwavers::ml::pinn" src/ tests/ examples/ benches/
grep -r "use kwavers::inverse::pinn" src/ tests/ examples/ benches/
```

**Fix Pattern**:
```rust
// WRONG (old path)
use kwavers::ml::pinn::*;

// CORRECT (new path)
use kwavers::solver::inverse::pinn::ml::*;
```

**Files to Check** (likely):
- `benches/pinn_*.rs`
- `tests/pinn_*.rs`
- `examples/pinn_*.rs`

**Implementation**:
```bash
# Find all PINN-related files with imports
find . -name "*.rs" -exec grep -l "pinn" {} \; | head -20

# For each file, check if imports are correct
# Pattern: kwavers::solver::inverse::pinn::ml::* (correct)
#         NOT: kwavers::ml::pinn::* (wrong)
#         NOT: kwavers::inverse::pinn::* (wrong)
```

**Action Items**:
- [ ] Search for all PINN import statements
- [ ] Verify correct path: `kwavers::solver::inverse::pinn::ml::`
- [ ] Update each file with wrong imports
- [ ] Verify compiles after each fix
- [ ] Commit: `fix(pinn): Correct import paths to solver::inverse::pinn::ml`

---

## Step 3: Fix Module Reference Errors

**Issue**: Code references modules that don't exist

**Common Cases**:
1. `kwavers::ml::` - doesn't exist, should be `kwavers::solver::inverse::`
2. `kwavers::ai_integration::` - doesn't exist, remove or move
3. `kwavers::adaptive::legacy::` - doesn't exist, check if needed

**Search**:
```bash
grep -r "use kwavers::ml" src/ tests/ examples/ benches/
grep -r "use kwavers::ai_integration" src/ tests/ examples/ benches/
grep -r "use kwavers::adaptive" src/ tests/ examples/ benches/
```

**For each found**:
1. Verify module actually exists
   ```bash
   ls -la src/solver/inverse/pinn/ml.rs
   ls -la src/clinical/ai_integration/mod.rs
   ```

2. If module doesn't exist:
   - Check if code is test/example only → DELETE or DISABLE test
   - Check if feature-gated → Add `#[cfg(feature = "...")]`
   - Check if unfinished feature → Move to TODO file

3. If module exists but path wrong:
   - Correct the import path
   - Verify with `cargo check`

**Action Items**:
- [ ] List all module reference errors
- [ ] Classify each (delete/disable/fix path)
- [ ] Apply fixes by category
- [ ] Verify compilation
- [ ] Commit: `fix: Correct module import paths and remove broken references`

---

## Step 4: Fix Type and Signature Mismatches

**Issue**: Function calls with wrong number/type of arguments

**Example**:
```rust
// WRONG - function expects (x: i32, y: i32) but got (x: i32)
my_function(5);

// CORRECT
my_function(5, 10);
```

**Find All**:
```bash
cargo build --all-targets 2>&1 | grep "^error\[E0308\]"
cargo build --all-targets 2>&1 | grep "^error\[E0061\]"
```

**For each error**:
1. Read the exact error message
2. Find the function definition
3. Compare expected vs. actual arguments
4. Fix either:
   - Call site (add missing args)
   - Function signature (change parameter list)
   - Type conversion (cast or convert value)

**Action Items**:
- [ ] List all type mismatch errors
- [ ] Verify function signatures in source
- [ ] Fix call sites
- [ ] Add type annotations if needed
- [ ] Commit: `fix: Correct function signatures and type mismatches`

---

## Step 5: Fix Syntax Errors

**Issue**: Code with syntax problems (extra braces, semicolons, etc)

**Common**:
```rust
// WRONG - extra closing brace
fn my_func() {
    println!("hello");
}}  // <-- extra brace

// WRONG - missing semicolon
let x = 5

// WRONG - missing type annotation
let x = compute_value();  // if compute_value() return type is ambiguous
```

**Find All**:
```bash
cargo build --all-targets 2>&1 | grep "^error\[E0308\]"
cargo build --all-targets 2>&1 | grep "expected"
```

**For each**:
1. Go to exact line in file
2. Read the error message
3. Fix syntax error
4. Verify with `cargo check`

**Known Issues** (from audit):
- [ ] Fix `slsc/mod.rs:716` - extra closing brace
- [ ] Fix `validation_suite.rs:167` - type issue
- [ ] Fix any other syntax errors found

**Action Items**:
- [ ] Fix all syntax errors
- [ ] Test each fix
- [ ] Commit: `fix: Correct syntax errors`

---

## Step 6: Fix Disabled/Stub Tests

**Issue**: Tests that reference non-existent modules or unimplemented features

**Options**:
1. **Delete if not needed**: Remove the entire test
2. **Disable if WIP**: Add `#[ignore]` or `#[disabled]`
3. **Fix if implementable**: Implement the feature or mock the module

**Examples**:
```rust
// Option 1: Delete (if not testing anything important)
// [Remove entire test function]

// Option 2: Disable with #[ignore]
#[test]
#[ignore]  // TODO: Implement feature X
fn test_unimplemented_feature() {
    // test code
}

// Option 3: Fix (if implementable)
#[test]
fn test_valid_feature() {
    // Fixed test code using available APIs
}
```

**Files to Check**:
- `tests/ai_integration_test.rs` - likely has issues
- `tests/pinn_*.rs` - check for module references
- Any test file with non-matching imports

**Action Items**:
- [ ] Review each failing test
- [ ] Decide: delete/disable/fix
- [ ] Apply changes
- [ ] Commit: `fix(tests): Remove broken tests, disable WIP features`

---

## Step 7: Verify Clean Compilation

**Command** (after all fixes):
```bash
cargo clean
cargo build --all-targets
cargo clippy --all-features
cargo test --lib --no-fail-fast 2>&1 | tail -20
```

**Expected Result**:
```
   Compiling kwavers v...
    Finished `dev` profile
✅ All targets compile successfully
```

**If Still Errors**:
1. Re-run `cargo build --all-targets 2>&1` to see remaining errors
2. Go back to appropriate step above
3. Fix remaining issues

---

## Step 8: Run Full Test Suite

**Command**:
```bash
cargo test --lib 2>&1 | tail -30
```

**Expected Result**:
```
test result: ok. XXXX passed; 0 failed; Y ignored
```

**If Tests Fail**:
1. Check if failures are pre-existing physics tests (acceptable)
2. Check if failures are compilation-related (fix above)
3. If logic failures: Review test code and fix logic

---

## Step 9: Commit and Verify

**Final Commit**:
```bash
git status  # Review all changes
git diff --stat  # See file count and changes
git add -A
git commit -m "Phase 1: Fix critical compilation errors

- Fix PINN import paths (ml module moved to solver::inverse::pinn::ml)
- Remove broken module references (ai_integration, etc)
- Fix type and signature mismatches
- Fix syntax errors
- Remove/disable broken tests
- Result: Clean compilation, zero blockers"
```

**Verification**:
```bash
# Verify on main branch
git log --oneline -1  # See latest commit
cargo build --release  # Full clean build
cargo test --lib 2>&1 | grep "test result"
```

---

## Troubleshooting

### Error: "Cannot find module X"
**Cause**: Module doesn't exist  
**Fix**:
1. Check if it should be in a different path
2. Check if it's feature-gated
3. If not implementable yet: Remove the test or code using it

### Error: "Type mismatch"
**Cause**: Function argument types don't match  
**Fix**:
1. Check function definition
2. Cast or convert the argument
3. Or change function signature if correct signature is known

### Error: "Unresolved import"
**Cause**: Import path is wrong  
**Fix**:
1. Verify correct module path exists
2. Update import to correct path
3. Test with `cargo check`

### Test Fails but Compiles
**Cause**: Logic error in test or code  
**Fix**:
1. If physics test: Mark as `#[ignore]` (pre-existing issue)
2. If logic test: Review test expectations and code
3. If integration test: Verify modules it depends on exist

---

## Checklist for Completion

Phase 1 is complete when:

- [ ] `cargo build --all-targets` succeeds
- [ ] `cargo clippy --all-features` shows no `error:` lines
- [ ] `cargo test --lib` shows 95%+ pass rate
- [ ] All changes committed to `main` branch
- [ ] `git log` shows Phase 1 commit
- [ ] Baseline metrics recorded

---

## Expected Time Breakdown

| Task | Time |
|------|------|
| Step 1-2: Identify & fix PINN imports | 30 min |
| Step 3: Fix module references | 30 min |
| Step 4: Fix type mismatches | 30 min |
| Step 5: Fix syntax errors | 20 min |
| Step 6: Fix/disable tests | 30 min |
| Step 7-8: Verify & test | 30 min |
| Step 9: Commit & document | 15 min |
| **TOTAL** | **~3-4 hours** |

---

## Next: After Phase 1 Complete

Once Phase 1 complete:
1. **Immediately**: Review baseline metrics
2. **Next Session**: Start Phase 2 (Architecture verification)
3. **Following**: Phase 3 (Dead code elimination)

See `STRATEGIC_ENHANCEMENT_PLAN.md` for Phase 2-8 details.

---

**Status**: Ready to execute  
**Branch**: main (all commits here)  
**Approval**: Go ahead with Phase 1 fixes
