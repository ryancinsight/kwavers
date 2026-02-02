# Plugin Architecture Analysis
**Date**: 2026-01-29  
**Finding**: Three complementary plugin systems with clear separation of concerns

---

## Overview

The kwavers codebase has **THREE plugin systems**, not overlapping but complementary:

```
Domain Layer (Layer 2)
└── domain/plugin/          - Plugin SPECIFICATIONS & CONTRACTS

Physics Layer (Layer 3)
└── physics/plugin/         - Physics-specific IMPLEMENTATIONS

Solver Layer (Layer 4)
└── solver/plugin/          - Plugin ORCHESTRATION & MANAGEMENT
```

**Assessment**: This is **CORRECT ARCHITECTURE** ✅ (not a duplication issue)

---

## Detailed Comparison

### 1. domain/plugin/ - Plugin Specifications (SSOT for contracts)

**Purpose**: Define what a plugin is and what contracts it must fulfill

**Exports**:
- `Plugin` trait - Core plugin interface
- `PluginState` enum - Plugin lifecycle states
- `PluginPriority` enum - Execution ordering
- `PluginContext` struct - Runtime context passed to plugins
- `PluginMetadata` struct - Plugin identification
- `PluginFields` struct - Extra fields for communication

**Key Methods** (in Plugin trait):
- `metadata()` - Plugin identity
- `state()` / `set_state()` - Lifecycle management
- `required_fields()` / `provided_fields()` - Field dependencies
- `update()` - Core plugin execution
- `initialize()` / `finalize()` - Setup/teardown
- `stability_constraints()` - Physics constraints
- `priority()` - Execution order
- `is_compatible_with()` - Compatibility checking
- `as_any()` / `as_any_mut()` - Downcasting

**Layer**: Domain (specifications)

**Responsibility**: "What is a plugin? What contract must it implement?"

**Example Use**:
```rust
pub trait Plugin: Debug + Send + Sync {
    fn metadata(&self) -> &PluginMetadata;
    fn required_fields(&self) -> Vec<UnifiedFieldType>;
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PluginContext<'_>,
    ) -> KwaversResult<()>;
    // ... other methods
}
```

---

### 2. physics/plugin/ - Physics Plugin Implementations

**Purpose**: Provide physics-specific field accessors and implementations

**Exports**:
- `FieldAccessor` - Physics-specific field access

**Contents**:
- `field_access.rs` - Accessor implementations for physical fields

**Layer**: Physics (implementations using domain specifications)

**Responsibility**: "How do physics plugins access and manipulate fields?"

**Key Difference**:
- Lightweight module
- Focuses on **physics-specific implementations** of the domain Plugin trait
- Provides helper functions for physics calculations
- Implements domain::plugin::Plugin for physics plugins

**Example Architecture**:
```rust
// domain/plugin/mod.rs defines:
pub trait Plugin { ... }

// physics/plugin/field_access.rs implements:
pub struct FieldAccessor { ... }  // Helps physics plugins use fields
impl Plugin for PhysicsPlugin { ... }  // Implements domain contract
```

---

### 3. solver/plugin/ - Plugin Orchestration & Management

**Purpose**: Execute and manage plugins within the solver loop

**Exports**:
- `PluginManager` - Manages registered plugins
- `PluginExecutor` - Executes plugins
- `ExecutionStrategy` trait - Different execution orders
- `SequentialStrategy` - Sequential execution implementation

**Key Differences**:
- **Management**: Register, unregister, retrieve plugins
- **Execution**: Run plugins in correct order with correct timing
- **Orchestration**: Coordinate plugin dependencies and priorities
- **Scheduling**: Respect PluginPriority for execution order

**Layer**: Solver (orchestration)

**Responsibility**: "How do we manage and run plugins in the solver loop?"

**Example Architecture**:
```rust
// Domain defines Plugin trait
// Physics implements Plugin for specific physics
// Solver uses PluginManager to run them:

let mut manager = PluginManager::new();
manager.register("my_physics", physics_plugin)?;
executor.execute_plugins(&mut manager, &mut fields, dt)?;
```

---

## Architecture Diagram

```
Domain Layer (Specifications)
┌─────────────────────────────────┐
│ domain/plugin/                  │
│ - Plugin trait (contract)        │
│ - PluginState enum              │
│ - PluginPriority enum           │
│ - PluginContext struct          │
│ - PluginMetadata struct         │
└──────────────▲──────────────────┘
               │ implements
               │
Physics Layer (Implementations)
┌─────────────────────────────────┐
│ physics/plugin/                 │
│ - FieldAccessor (physics-specific) │
│ - Physics Plugin implementations │
└──────────────▲──────────────────┘
               │ uses/manages
               │
Solver Layer (Orchestration)
┌─────────────────────────────────┐
│ solver/plugin/                  │
│ - PluginManager (registration)  │
│ - PluginExecutor (execution)    │
│ - ExecutionStrategy (scheduling)│
└─────────────────────────────────┘
```

---

## Responsibility Distribution

| Aspect | Domain | Physics | Solver |
|--------|--------|---------|--------|
| **What is a plugin?** | ✅ Defines trait | - | - |
| **Interface contract** | ✅ Specifies methods | - | - |
| **Plugin state management** | ✅ Defines states | - | - |
| **Physics-specific access** | - | ✅ Implements accessors | - |
| **Physics calculations** | - | ✅ Custom logic | - |
| **Plugin registration** | - | - | ✅ Manages registry |
| **Plugin execution** | - | - | ✅ Runs in order |
| **Dependency ordering** | - | - | ✅ Uses priority |
| **Stability constraints** | ✅ Trait method | ✅ Implements | ✅ Enforces |

---

## Comparison Table

| Feature | domain/plugin | physics/plugin | solver/plugin |
|---------|---------------|----------------|---------------|
| **Type** | Specifications | Implementations | Orchestration |
| **Layer** | Domain (2) | Physics (3) | Solver (4) |
| **Dependencies** | Core, Error | Domain | Domain, Physics |
| **Exports** | Plugin trait | FieldAccessor | Manager, Executor |
| **What It Defines** | Contracts | Field access | Execution logic |
| **What It Implements** | Interfaces | Physics logic | Management loop |
| **Size** | Medium (~200 lines) | Small (~100 lines) | Medium (~200 lines) |
| **SSOT?** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Overlap?** | ❌ No | ❌ No | ❌ No |

---

## Is This Duplication? ❌ NO

**Why it's NOT duplication:**

1. **Different Layers, Different Concerns**
   - Domain: "What must plugins implement?"
   - Physics: "How do physics plugins work?"
   - Solver: "How do we run plugins?"

2. **Clear Dependency Flow**
   ```
   Domain (specifications) ← Physics (uses domain)
   Domain + Physics ← Solver (uses both)
   ```

3. **Single Responsibility Principle**
   - Each module has ONE reason to change
   - Each module is testable independently
   - Each module is reusable

4. **Proper Layer Separation**
   - No upward dependencies (solver doesn't define plugins)
   - No downward coupling (physics doesn't depend on solver)
   - Clean interfaces between layers

---

## Architecture Compliance ✅

| Criterion | Status | Notes |
|-----------|--------|-------|
| **Layering** | ✅ PASS | Domain → Physics → Solver |
| **SSOT** | ✅ PASS | Each module is SSOT for its concern |
| **Separation of Concerns** | ✅ PASS | No mixed responsibilities |
| **Circular Dependencies** | ✅ PASS | None detected |
| **Duplication** | ✅ PASS | Each serves different purpose |
| **Reusability** | ✅ PASS | Can use domain without physics, etc. |

---

## How They Work Together

### Example: Physics Plugin Lifecycle

```rust
// 1. Domain defines the contract
pub trait Plugin { ... }

// 2. Physics implements the contract
pub struct MyPhysicsPlugin { ... }
impl Plugin for MyPhysicsPlugin { ... }

// 3. Solver orchestrates execution
let mut manager = PluginManager::new();
manager.register("physics", MyPhysicsPlugin::new())?;

// In solver loop:
executor.execute_all(&mut manager, ...)?;  // Respects priorities
```

---

## When to Use Each

### domain/plugin/
- **When**: Defining a new plugin interface
- **What**: Traits and specifications
- **Who**: Library developers, architects

### physics/plugin/
- **When**: Implementing physics-specific plugins
- **What**: Physics calculations, field access helpers
- **Who**: Physics domain specialists

### solver/plugin/
- **When**: Integrating plugins into solver
- **What**: Execution order, lifecycle management
- **Who**: Solver developers, integrators

---

## Recommendation ✅

**KEEP THIS ARCHITECTURE** - It's correct!

The three plugin systems are **not duplication** but **intentional layering**:

1. ✅ **domain/plugin/** - SSOT for plugin specifications
2. ✅ **physics/plugin/** - Physics-specific implementations
3. ✅ **solver/plugin/** - Plugin orchestration engine

Each layer serves a distinct purpose with clean dependencies and no overlap.

---

## Future Considerations

### If Plugin System Grows

Consider these optional improvements (not urgent):

1. **Documentation**: Add examples of how to implement custom physics plugins
2. **Registry**: Centralized plugin discovery mechanism
3. **Versioning**: Plugin compatibility checking
4. **Testing**: Plugin test harness framework

### What NOT to Do

❌ Don't consolidate into single module (loses separation of concerns)  
❌ Don't move domain/plugin to physics (violates layering)  
❌ Don't move solver/plugin to domain (domain shouldn't know about execution)

---

## Summary

The three plugin systems are **correctly separated** and form a **clean layered architecture**:

- **Domain** = "What is a plugin?" (specification)
- **Physics** = "How do physics plugins work?" (implementation)  
- **Solver** = "How do we run plugins?" (orchestration)

**Verdict**: ✅ **NO ACTION NEEDED** - This is good architecture!

---

**Analysis Date**: 2026-01-29  
**Conclusion**: Plugin architecture is architecturally sound with proper separation of concerns
