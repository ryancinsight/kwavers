// Volume Rendering Shader - Phase 11 Visualization
// GPU volume rendering with ray casting and transparency

struct VolumeUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    volume_size: vec3<f32>,
    color_scale: vec4<f32>,
    transparency: f32,
    iso_value: f32,
    step_size: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) ray_direction: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: VolumeUniforms;
@group(0) @binding(1) var volume_texture: texture_3d<f32>;
@group(0) @binding(2) var color_lut: texture_1d<f32>;
@group(0) @binding(3) var volume_sampler: sampler;

// Vertex shader - creates a full-screen quad for ray casting
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Create full-screen quad
    let x = f32((vertex_index << 1u) & 2u) - 1.0;
    let y = f32(vertex_index & 2u) - 1.0;
    
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    
    // Calculate ray direction in world space
    let clip_pos = vec4<f32>(x, y, 1.0, 1.0);
    let world_pos = uniforms.view_matrix * clip_pos;
    out.ray_direction = normalize(world_pos.xyz);
    
    return out;
}

// Fragment shader - performs volume ray casting
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_origin = vec3<f32>(0.0, 0.0, -2.0); // Camera position
    let ray_dir = normalize(in.ray_direction);
    
    // Ray-volume intersection
    let inv_ray_dir = 1.0 / ray_dir;
    let t_min_vec = (vec3<f32>(0.0) - ray_origin) * inv_ray_dir;
    let t_max_vec = (uniforms.volume_size - ray_origin) * inv_ray_dir;
    
    let t_min_all = min(t_min_vec, t_max_vec);
    let t_max_all = max(t_min_vec, t_max_vec);
    
    let t_near = max(max(t_min_all.x, t_min_all.y), t_min_all.z);
    let t_far = min(min(t_max_all.x, t_max_all.y), t_max_all.z);
    
    if (t_near > t_far || t_far < 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0); // Miss
    }
    
    // Volume ray marching
    let start_pos = ray_origin + ray_dir * max(t_near, 0.0);
    let end_pos = ray_origin + ray_dir * t_far;
    let ray_length = distance(end_pos, start_pos);
    
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;
    var current_pos = start_pos;
    
    let num_steps = i32(ray_length / uniforms.step_size);
    let step_vec = ray_dir * uniforms.step_size;
    
    for (var i = 0; i < num_steps; i++) {
        if (accumulated_alpha >= 0.99) {
            break; // Early termination for full opacity
        }
        
        // Sample volume texture
        let tex_coord = current_pos / uniforms.volume_size;
        if (all(tex_coord >= vec3<f32>(0.0)) && all(tex_coord <= vec3<f32>(1.0))) {
            let density = textureSample(volume_texture, volume_sampler, tex_coord).r;
            
            if (density > 0.01) { // Skip empty space
                // Map density to color using lookup table
                let color_sample = textureSample(color_lut, volume_sampler, density);
                let sample_color = color_sample.rgb * uniforms.color_scale.rgb;
                let sample_alpha = density * uniforms.transparency * uniforms.step_size;
                
                // Front-to-back alpha blending
                let alpha_factor = sample_alpha * (1.0 - accumulated_alpha);
                accumulated_color += sample_color * alpha_factor;
                accumulated_alpha += alpha_factor;
            }
        }
        
        current_pos += step_vec;
    }
    
    return vec4<f32>(accumulated_color, accumulated_alpha);
}

// Isosurface extraction functions
fn compute_gradient(pos: vec3<f32>) -> vec3<f32> {
    let epsilon = 1.0 / 256.0; // Gradient sampling offset
    
    let grad_x = textureSample(volume_texture, volume_sampler, pos + vec3<f32>(epsilon, 0.0, 0.0)).r -
                 textureSample(volume_texture, volume_sampler, pos - vec3<f32>(epsilon, 0.0, 0.0)).r;
    let grad_y = textureSample(volume_texture, volume_sampler, pos + vec3<f32>(0.0, epsilon, 0.0)).r -
                 textureSample(volume_texture, volume_sampler, pos - vec3<f32>(0.0, epsilon, 0.0)).r;
    let grad_z = textureSample(volume_texture, volume_sampler, pos + vec3<f32>(0.0, 0.0, epsilon)).r -
                 textureSample(volume_texture, volume_sampler, pos - vec3<f32>(0.0, 0.0, epsilon)).r;
    
    return normalize(vec3<f32>(grad_x, grad_y, grad_z) / (2.0 * epsilon));
}

fn phong_lighting(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let ambient = 0.2;
    let diffuse = max(dot(normal, light_dir), 0.0) * 0.6;
    let reflect_dir = reflect(-light_dir, normal);
    let specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * 0.4;
    
    return ambient + diffuse + specular;
}