# Example: Comprehensive Clinical Workflow

**Crate**: `kwavers`  
**Run**: `cargo run -p kwavers --example comprehensive_clinical_workflow --features gpu`  
**Source**: [`crates/kwavers/examples/comprehensive_clinical_workflow.rs`](../../../crates/kwavers/examples/comprehensive_clinical_workflow.rs)

## What This Example Demonstrates

This example combines multiple simulation capabilities into a single liver-assessment workflow. It sequences B-mode imaging, shear-wave elastography, CEUS perfusion analysis, uncertainty quantification, treatment planning, and safety/clinical validation into one report-oriented pipeline.

| Component | API | Value |
|---|---|---|
| Workflow object | `LiverAssessmentWorkflow::new` | Initializes a patient-specific liver study over a 120×80×60 mm³ volume |
| Clinical imaging | `ContrastEnhancedUltrasound::new` | Creates a 5 MHz CEUS system coupled to a heterogeneous liver model |
| Acceleration | `--features gpu` + `UnifiedMemoryManager` | Enables the full GPU-backed workflow path described by the example |

## Key Code Snippet

```rust
let mut workflow = LiverAssessmentWorkflow::new(
    "LIVER_PATIENT_001",
    (120.0, 80.0, 60.0), // 120x80x60 mm³ liver volume
)?;

// Execute complete assessment protocol
let report = workflow.execute_assessment()?;

// Display comprehensive results
println!("\n=== LIVER ASSESSMENT REPORT ===");
```

## Expected Output (if applicable)

With GPU features enabled, the program prints a multi-section liver assessment report covering imaging, perfusion, safety, uncertainty, and planning results.

## Book Chapter

[← Therapeutic Ultrasound](../therapy.md)
