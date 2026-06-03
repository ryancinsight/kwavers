//! Shader compilation helpers for PipelineManager.

use super::super::types::PipelineType;
use kwavers_core::error::KwaversResult;
use std::collections::HashMap;
use wgpu;

impl super::PipelineManager {
    /// Compile elementwise pipeline.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compile_elementwise_pipeline(
        device: &wgpu::Device,
        pipelines: &mut HashMap<PipelineType, wgpu::ComputePipeline>,
        layouts: &mut HashMap<PipelineType, wgpu::BindGroupLayout>,
    ) -> KwaversResult<()> {
        let shader_source = include_str!("../../shaders/operators.wgsl");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("operators-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("elementwise-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("elementwise-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("elementwise-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("elementwise_multiply"),
            compilation_options: Default::default(),
            cache: None,
        });

        layouts.insert(PipelineType::ElementWiseMultiply, layout);
        pipelines.insert(PipelineType::ElementWiseMultiply, pipeline);

        Ok(())
    }
    /// Compile derivative pipeline.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compile_derivative_pipeline(
        device: &wgpu::Device,
        pipelines: &mut HashMap<PipelineType, wgpu::ComputePipeline>,
        layouts: &mut HashMap<PipelineType, wgpu::BindGroupLayout>,
    ) -> KwaversResult<()> {
        let shader_source = include_str!("../../shaders/operators.wgsl");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("derivative-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("derivative-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("derivative-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("derivative-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("spatial_derivative"),
            compilation_options: Default::default(),
            cache: None,
        });

        layouts.insert(PipelineType::SpatialDerivative, layout);
        pipelines.insert(PipelineType::SpatialDerivative, pipeline);

        Ok(())
    }
}
