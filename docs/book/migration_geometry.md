# Chapter 37 — Geometry: Leto for Point, Vector, Isometry

Replace `nalgebra` geometric primitives with `leto::geometry::*`.

## Type Mapping

| nalgebra | leto::geometry |
|---|---|
| `Vector2<f64>` | `Vector2<f64>` |
| `Vector3<f64>` | `Vector3<f64>` |
| `Point2<f64>` | `Point2<f64>` |
| `Point3<f64>` | `Point3<f64>` |
| `Isometry3<f64>` | `Isometry3<f64>` |
| `Quaternion<f64>` | `Quaternion<f64>` |
| `UnitQuaternion<f64>` | `UnitQuaternion<f64>` |
| `Translation3<f64>` | `Translation3<f64>` |
| `Unit<T>` | `Unit<T>` |

All types are re-exported from `leto` root: `use leto::{Vector3, Point3, …}`.

## Key API Differences

```rust
// nalgebra
use nalgebra::{Vector3, Point3, UnitQuaternion};
let v = Vector3::new(1.0, 0.0, 0.0);
let q = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(v), 0.5);

// leto
use leto::{Vector3, Point3, UnitQuaternion};
let v = Vector3::new(1.0_f64, 0.0, 0.0);
let unit = leto::Unit::new_normalize(v);
let q = UnitQuaternion::from_axis_angle(&unit, 0.5);
```

## Isometry (rigid transforms)

```rust
use leto::{Isometry3, Translation3, UnitQuaternion, Vector3};

let t = Translation3::new(1.0_f64, 2.0, 3.0);
let r = UnitQuaternion::identity();
let iso = Isometry3::from_parts(t, r);
let pt = iso * leto::Point3::new(0.0, 0.0, 0.0);
// pt = (1.0, 2.0, 3.0)
```

## kwavers usage

kwavers uses `leto::geometry` in `kwavers-grid`, `kwavers-transducer`,
and `kwavers-solver` for source/receiver positions, array element coordinates,
and simulation domain geometry.
