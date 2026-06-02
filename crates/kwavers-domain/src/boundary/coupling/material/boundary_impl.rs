use super::MaterialInterface;
use kwavers_core::error::KwaversResult;
use crate::boundary::coupling::types::BoundaryDirections;
use crate::boundary::traits::BoundaryCondition;
use crate::grid::GridTopology;
use ndarray::ArrayViewMut3;

impl BoundaryCondition for MaterialInterface {
    fn name(&self) -> &str {
        "MaterialInterface"
    }

    fn active_directions(&self) -> BoundaryDirections {
        BoundaryDirections::all()
    }

    fn apply_scalar_spatial(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        let dimensions = grid.dimensions();
        let (nx, ny, nz) = (dimensions[0], dimensions[1], dimensions[2]);

        let r = self.reflection_coefficient();
        let t = self.transmission_coefficient();

        let interface_pos = self.position;
        let normal = self.normal;

        let normal_mag = normal[2]
            .mul_add(
                normal[2],
                normal[0].mul_add(normal[0], normal[1] * normal[1]),
            )
            .sqrt();
        let normal_unit = [
            normal[0] / normal_mag,
            normal[1] / normal_mag,
            normal[2] / normal_mag,
        ];

        let spacing = grid.spacing();
        let dx = spacing[0];
        let smooth_thickness = self.thickness.max(2.0 * dx);

        // PASS 1: Estimate incident wave amplitude from material 1 side near interface.
        let mut incident_amplitude = 0.0;
        let mut sample_count = 0;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let point = grid.indices_to_coordinates([i, j, k]);
                    let rel_pos = [
                        point[0] - interface_pos[0],
                        point[1] - interface_pos[1],
                        point[2] - interface_pos[2],
                    ];
                    let signed_distance = rel_pos[2].mul_add(
                        normal_unit[2],
                        rel_pos[0].mul_add(normal_unit[0], rel_pos[1] * normal_unit[1]),
                    );

                    if signed_distance < 0.0 && signed_distance.abs() < smooth_thickness {
                        incident_amplitude += field[[i, j, k]];
                        sample_count += 1;
                    }
                }
            }
        }

        if sample_count > 0 {
            incident_amplitude /= sample_count as f64;
        }

        // PASS 2: Apply reflection and transmission with smooth blending.
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let point = grid.indices_to_coordinates([i, j, k]);
                    let rel_pos = [
                        point[0] - interface_pos[0],
                        point[1] - interface_pos[1],
                        point[2] - interface_pos[2],
                    ];

                    let signed_distance = rel_pos[2].mul_add(
                        normal_unit[2],
                        rel_pos[0].mul_add(normal_unit[0], rel_pos[1] * normal_unit[1]),
                    );

                    if signed_distance.abs() <= smooth_thickness {
                        let p_current = field[[i, j, k]];
                        let blend = signed_distance / smooth_thickness;

                        if signed_distance <= 0.0 {
                            let p_reflected = r * incident_amplitude;
                            let reflection_weight = 1.0 - signed_distance.abs() / smooth_thickness;

                            if signed_distance.abs() < 1e-10 {
                                let p_transmitted = t * incident_amplitude;
                                field[[i, j, k]] =
                                    0.5f64.mul_add(p_current + p_reflected, 0.5 * p_transmitted);
                            } else {
                                field[[i, j, k]] = p_current + reflection_weight * p_reflected;
                            }
                        } else {
                            let p_transmitted = t * incident_amplitude;
                            field[[i, j, k]] = blend * p_transmitted + (1.0 - blend) * p_current;
                        }
                    } else if signed_distance > smooth_thickness {
                        field[[i, j, k]] = t * incident_amplitude;
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut ndarray::Array3<kwavers_math::fft::Complex64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        Ok(())
    }

    fn reset(&mut self) {}
}
