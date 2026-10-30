//! Constructors and element-addition methods for [`KWaveArray`].

use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::KwaversResult;

use aequitas::systems::si::quantities::{Angle, Length};
use aequitas::systems::si::units::{Degree, Meter};

use super::{ArrayTransform, DiscSourceProfile, ElementShape, KWaveArray, KWaveElement};
use crate::curvilinear::ConvexArrayGeometry;
use crate::transducers::physics::PlanarApertureGeometry;
use kwavers_core::error::{ConfigError, KwaversError};

/// Build the `Config::InvalidValue` error this module's constructors return
/// for rejected inputs, naming the offending parameter.
fn invalid_value(parameter: &str, value: String, constraint: &str) -> KwaversError {
    KwaversError::Config(ConfigError::InvalidValue {
        parameter: parameter.to_owned(),
        value,
        constraint: constraint.to_owned(),
    })
}

/// Reject a NaN or infinite element position with a typed error.
fn check_finite_position(parameter: &str, position: (f64, f64, f64)) -> KwaversResult<()> {
    if position.0.is_finite() && position.1.is_finite() && position.2.is_finite() {
        Ok(())
    } else {
        Err(invalid_value(
            parameter,
            format!("({}, {}, {})", position.0, position.1, position.2),
            "all coordinates must be finite",
        ))
    }
}

impl KWaveArray {
    /// Create a new empty `KWaveArray`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            frequency: MHZ_TO_HZ,
            sound_speed: SOUND_SPEED_TISSUE,
            _element_width: 0.5e-3,
            array_transform: None,
        }
    }

    /// Create a new array with specified frequency and sound speed.
    #[must_use]
    pub fn with_params(frequency: f64, sound_speed: f64) -> Self {
        Self {
            elements: Vec::new(),
            frequency,
            sound_speed,
            _element_width: 0.5e-3,
            array_transform: None,
        }
    }

    /// Set the element width (for arc discretization).
    #[must_use]
    pub fn with_element_width(mut self, width: f64) -> Self {
        self._element_width = width;
        self
    }

    /// Get the operating frequency (Hz).
    #[must_use]
    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Update the operating frequency while preserving existing elements.
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = frequency;
    }

    /// Update the sound speed while preserving existing elements.
    pub fn set_sound_speed(&mut self, sound_speed: f64) {
        self.sound_speed = sound_speed;
    }

    /// Install a global translation + intrinsic X-Y-Z Euler rotation (degrees)
    /// applied to every element at rasterization time. Mirrors
    /// k-wave-python's `kWaveArray.set_array_position(translation, rotation)`.
    /// Passing zero for both is equivalent to the identity transform.
    pub fn set_array_position(
        &mut self,
        translation: (f64, f64, f64),
        euler_xyz_deg: (f64, f64, f64),
    ) {
        self.array_transform = Some(ArrayTransform {
            translation,
            euler_xyz_deg,
        });
    }

    /// Remove the global array transform if one was previously installed.
    pub fn clear_array_position(&mut self) {
        self.array_transform = None;
    }

    /// Add an arc-shaped element with default ±45° angular span.
    ///
    /// # Arguments
    /// * `position` - Arc center `[x, y, z]` in metres
    /// * `radius`   - Arc radius in metres
    /// * `diameter` - Element diameter in metres
    pub fn add_arc_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
    ) -> &mut Self {
        self.elements.push(
            ElementShape::Arc {
                position,
                radius,
                diameter,
                start_angle: -45.0,
                end_angle: 45.0,
            }
            .into(),
        );
        self
    }

    /// Add an arc-shaped element with custom angular span.
    ///
    /// # Arguments
    /// * `position`    - Arc center `[x, y, z]` in metres
    /// * `radius`      - Arc radius in metres
    /// * `diameter`    - Element diameter in metres
    /// * `start_angle` - Start angle in degrees
    /// * `end_angle`   - End angle in degrees
    pub fn add_arc_element_with_angles(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
        start_angle: f64,
        end_angle: f64,
    ) -> &mut Self {
        self.elements.push(
            ElementShape::Arc {
                position,
                radius,
                diameter,
                start_angle,
                end_angle,
            }
            .into(),
        );
        self
    }

    /// Add an axis-aligned rectangular element (ADR 136).
    ///
    /// Builds a `PlanarApertureShape::Rectangle` under
    /// `SurfaceConvention::KWave`: the rect-canonical lattice deposited
    /// through the raw k-Wave BLI stencil, bit-for-bit the retired
    /// `ElementShape::Rect` path. The legacy `length` parameter is gone
    /// (nothing consumed it); this is a breaking change, see below.
    ///
    /// # Arguments
    /// * `position` - Center position `[x, y, z]` in metres
    /// * `width`    - Width in x-direction (m)
    /// * `height`   - Height in y-direction (m)
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` when the dimensions are not positive
    /// and finite, or the position is not finite.
    ///
    /// # Breaking change
    ///
    /// The `length` parameter of the previous signature is removed; update
    /// call sites to the 3-argument form.
    pub fn add_rect_element(
        &mut self,
        position: (f64, f64, f64),
        width: f64,
        height: f64,
    ) -> KwaversResult<&mut Self> {
        self.add_rect_rot_element(position, width, height, (0.0, 0.0, 0.0))
    }

    /// Add a rectangular element rotated about its center by intrinsic X-Y-Z
    /// Euler angles (degrees). Matches the upstream k-wave-python
    /// `KWaveArray.add_rect_element` rotation contract used by the linear
    /// array transducer example.
    ///
    /// Builds a `PlanarApertureShape::Rectangle` under
    /// `SurfaceConvention::KWave` (ADR 136): the element frame
    /// `(R·e_x, R·e_z)` of the Euler rotation becomes the aperture
    /// `(first_axis, normal)`, so the rasterized values match the retired
    /// `ElementShape::Rect` path at the k-Wave reference tolerance.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` when the dimensions are not positive
    /// and finite, or the position is not finite.
    ///
    /// # Breaking change
    ///
    /// The `length` parameter of the previous signature is removed; update
    /// call sites to the 4-argument form.
    pub fn add_rect_rot_element(
        &mut self,
        position: (f64, f64, f64),
        width: f64,
        height: f64,
        euler_xyz_deg: (f64, f64, f64),
    ) -> KwaversResult<&mut Self> {
        use super::math::{apply_matrix, euler_xyz_rotation_matrix};
        use crate::transducers::physics::{PlanarApertureShape, SurfaceConvention};

        // Validate inputs up front so every rejection below is a `Config`
        // error naming the parameter; the `?` on the aperture build then
        // only guards unreachable residuals (a finite Euler rotation always
        // yields a unit frame, and the bounds were just checked).
        check_finite_position("position", position)?;
        if !(width.is_finite() && width > 0.0) {
            return Err(invalid_value(
                "width_m",
                width.to_string(),
                "must be finite and positive",
            ));
        }
        if !(height.is_finite() && height > 0.0) {
            return Err(invalid_value(
                "height_m",
                height.to_string(),
                "must be finite and positive",
            ));
        }
        for (axis, angle) in [
            ("euler_x_deg", euler_xyz_deg.0),
            ("euler_y_deg", euler_xyz_deg.1),
            ("euler_z_deg", euler_xyz_deg.2),
        ] {
            if !angle.is_finite() {
                return Err(invalid_value(axis, angle.to_string(), "must be finite"));
            }
        }
        let rotation = euler_xyz_rotation_matrix(euler_xyz_deg);
        let (first_x, first_y, first_z) = apply_matrix(&rotation, (1.0, 0.0, 0.0));
        let (normal_x, normal_y, normal_z) = apply_matrix(&rotation, (0.0, 0.0, 1.0));
        let geometry = PlanarApertureGeometry::oriented_with_convention(
            [position.0, position.1, position.2],
            [normal_x, normal_y, normal_z],
            [first_x, first_y, first_z],
            PlanarApertureShape::Rectangle {
                width_m: width,
                height_m: height,
            },
            SurfaceConvention::KWave,
        )?;
        self.elements.push(KWaveElement::PlanarAperture(geometry));
        Ok(self)
    }

    /// Add every element of a convex (curvilinear) array.
    ///
    /// Each element becomes one rotated rectangle: the layout supplies the
    /// centre, and the element's outward normal is produced by rotating the
    /// rectangle about `y` by its own subtended angle.
    ///
    /// # Why the rotated rect and not the arc (ADR 112, ADR 136)
    /// The rect lattice is built at `(lx, ly, 0)` before applying the Euler
    /// rotation, so a rectangle's local normal is `+z`. The planar aperture
    /// stores that rotated frame as `(first_axis, normal)`.
    /// `euler_xyz_rotation_matrix` composes `Rz·Ry·Rx`, whose `y` block sends
    /// `+z` to `[sin β, 0, cos β]` — exactly
    /// [`ConvexArrayGeometry::element_normal`] at `β = θᵢ`, with no sign flip
    /// and no axis swap.
    ///
    /// The arc primitive cannot serve here: `rasterize_arc_points` writes
    /// `center.2` unchanged into every sample, so it only expresses arcs in the
    /// `x`–`y` plane, while the convex layout curves in `x`–`z`.
    ///
    /// # Arguments
    /// * `geometry`       - Convex array layout
    /// * `element_width`  - Element extent along the array (m)
    /// * `element_height` - Element extent across the array, elevation (m)
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` when the element dimensions are not
    /// positive and finite (propagated from `add_rect_rot_element`).
    pub fn add_convex_array(
        &mut self,
        geometry: &ConvexArrayGeometry,
        element_width: Length<f64>,
        element_height: Length<f64>,
    ) -> KwaversResult<&mut Self> {
        let width = element_width.in_unit::<Meter>();
        let height = element_height.in_unit::<Meter>();
        for i in 0..geometry.num_elements() {
            let position = geometry.element_position(i);
            // The rasterizer's surface is f64 degrees; reading the typed angle
            // in degrees is the whole conversion. No hand-rolled to_degrees(),
            // so a unit slip here is a type error rather than a wrong mask.
            let theta: Angle<f64> = geometry.element_angle_quantity(i);
            let theta_deg = theta.in_unit::<Degree>();
            self.add_rect_rot_element(
                (position[0], position[1], position[2]),
                width,
                height,
                (0.0, theta_deg, 0.0),
            )?;
        }
        Ok(self)
    }

    /// Add a disc-shaped element.
    ///
    /// # Arguments
    /// * `position`       - Center position `[x, y, z]` in metres
    /// * `diameter`       - Disc diameter in metres
    /// * `focus_position` - Optional focus point defining the disc normal
    /// Add a disc-shaped element (ADR 136 follow-up for discs).
    ///
    /// Builds a `PlanarApertureShape::Disk` under `SurfaceConvention::KWave`:
    /// the disc normal derives from `focus_position` exactly as the retired
    /// `ElementShape::Disc` path (`(focus − center)/‖…‖`, or `e_z`
    /// unfocused), and the in-plane frame is the legacy disc-basis tangent,
    /// so the rasterized spiral coincides with the retired path. Unlike the
    /// rectangle fold this changes no signature: there was no dead
    /// parameter to remove.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` when the diameter is not positive and
    /// finite, the position is not finite, or a focus coincides with the
    /// disc centre.
    pub fn add_disc_element(
        &mut self,
        position: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) -> KwaversResult<&mut Self> {
        use super::math::DISC_AXIS_EPSILON;
        use crate::transducers::physics::{PlanarApertureShape, SurfaceConvention};

        check_finite_position("position", position)?;
        if !(diameter.is_finite() && diameter > 0.0) {
            return Err(invalid_value(
                "diameter",
                diameter.to_string(),
                "must be finite and positive",
            ));
        }
        // Reject a coincident focus up front with a typed error, using the
        // same comparison `disc_basis` asserts on, so its assert is
        // unreachable from this constructor.
        if let Some((fx, fy, fz)) = focus_position {
            let nx = fx - position.0;
            let ny = fy - position.1;
            let nz = fz - position.2;
            let norm = nz.mul_add(nz, nx.mul_add(nx, ny * ny)).sqrt();
            if norm <= DISC_AXIS_EPSILON {
                return Err(invalid_value(
                    "focus_position",
                    format!("focus=({fx}, {fy}, {fz})"),
                    "focus must differ from the disc centre",
                ));
            }
        }
        let (first, _, normal) = Self::disc_basis(position, focus_position);
        let geometry = PlanarApertureGeometry::oriented_with_convention(
            [position.0, position.1, position.2],
            normal,
            first,
            PlanarApertureShape::Disk {
                radius_m: diameter / 2.0,
            },
            SurfaceConvention::KWave,
        )?;
        self.elements.push(KWaveElement::PlanarAperture(geometry));
        Ok(self)
    }

    /// Add a disc-shaped element with a finite-source surface profile.
    ///
    /// # Arguments
    /// * `position`       - Center position `[x, y, z]` in metres
    /// * `diameter`       - Disc diameter in metres
    /// * `focus_position` - Optional focus point defining the disc normal
    /// * `profile`        - Normalized source-pressure profile over the disc
    pub fn add_profiled_disc_element(
        &mut self,
        position: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
        profile: DiscSourceProfile,
    ) -> &mut Self {
        self.elements.push(
            ElementShape::ProfiledDisc {
                position,
                diameter,
                focus_position,
                profile,
            }
            .into(),
        );
        self
    }

    /// Add validated oriented planar geometry: a disk, an annular sector, or a
    /// rectangle.
    pub fn add_planar_aperture_element(&mut self, geometry: PlanarApertureGeometry) -> &mut Self {
        self.elements.push(KWaveElement::PlanarAperture(geometry));
        self
    }

    /// Add a bowl-shaped (focused) element.
    ///
    /// # Arguments
    /// * `position` - Bowl center position `[x, y, z]` in metres
    /// * `radius`   - Radius of curvature (m)
    /// * `diameter` - Bowl aperture diameter (m)
    pub fn add_bowl_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
    ) -> &mut Self {
        self.elements.push(
            ElementShape::Bowl {
                position,
                radius,
                diameter,
            }
            .into(),
        );
        self
    }

    /// Add an annular spherical-cap element — a bowl section bounded by inner
    /// and outer aperture diameters. Mirrors k-wave-python's
    /// `add_annular_element`. Uses the same orientation convention as
    /// `add_bowl_element`.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` when the diameters violate
    /// `0 ≤ inner_diameter < outer_diameter` (NaN fails the comparison and
    /// is rejected the same way the retired `assert!` rejected it).
    ///
    pub fn add_annular_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
    ) -> KwaversResult<&mut Self> {
        check_finite_position("position", position)?;
        if !(outer_diameter > inner_diameter && inner_diameter >= 0.0) {
            return Err(invalid_value(
                "inner_diameter/outer_diameter",
                format!("inner={inner_diameter}, outer={outer_diameter}"),
                "0 ≤ inner_diameter < outer_diameter",
            ));
        }
        self.elements.push(
            ElementShape::Annulus {
                position,
                radius,
                inner_diameter,
                outer_diameter,
            }
            .into(),
        );
        Ok(self)
    }

    /// Add a concentric annular array (sequence of annuli sharing a common
    /// center of curvature). Each `(inner_diameter, outer_diameter)` pair
    /// becomes one `ElementShape::Annulus`, matching k-wave-python's
    /// `add_annular_array`.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::Config` when any pair violates
    /// `0 ≤ inner_diameter < outer_diameter`; no element is added for the
    /// failing pair or any later pair.
    pub fn add_annular_array(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameters: &[(f64, f64)],
    ) -> KwaversResult<&mut Self> {
        for &(inner_d, outer_d) in diameters {
            self.add_annular_element(position, radius, inner_d, outer_d)?;
        }
        Ok(self)
    }
}
