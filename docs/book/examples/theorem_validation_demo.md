# Example: Theorem Validation Demo

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example theorem_validation_demo`  
**Source**: [`crates/kwavers/examples/theorem_validation_demo.rs`](../../../crates/kwavers/examples/theorem_validation_demo.rs)

## What This Example Demonstrates

This example runs the theorem-validation machinery over the mathematical claims encoded in kwavers. It generates both a compact pass-rate summary and a more detailed report suitable for inspection or debugging.

| Component | API | Value |
|---|---|---|
| Validator | `TheoremValidator` | Owns the validation suite and report-generation helpers |
| Result type | `TheoremValidation` | Captures theorem-by-theorem success state and quantitative bounds |
| Reporting | `generate_validation_report` | Builds a human-readable validation report after the suite runs |

## Key Code Snippet

```rust
let validations = validator.run_comprehensive_validation();

println!("Validated {} mathematical theorems", validations.len());

// Display results
display_validation_results(&validations);

// Generate detailed report
let report = validator.generate_validation_report(&validations);
println!("\n{}", report);
```

## Expected Output (if applicable)

The output includes a theorem-count summary, a passed/failed breakdown, detailed per-theorem lines, and a generated report.

## Book Chapter

[← Validation and Benchmarking](../validation_and_benchmarking.md)
