struct Uniforms {
    camera: mat4x4<f32>;
    camera_inverse: mat4x4<f32>;
    dimensions: vec4<f32>;
    cube_size: u32;
};

[[group(0), binding(0)]]
var<uniform> u: Uniforms;

struct Data {
    data: [[stride(4)]] array<u32>;
};

[[group(0), binding(1)]]
var<storage, read_write> d: Data;

[[stage(vertex)]]
fn vs_main([[builtin(vertex_index)]] in_vertex_index: u32) -> [[builtin(position)]] vec4<f32> {
    var x = 0.0;
    var y = 0.0;

    if (in_vertex_index == 0u) {
        x = -1.0;
        y = -1.0;
    } else if (in_vertex_index == 1u) {
        x = 1.0;
        y = -1.0;
    } else if (in_vertex_index == 2u) {
        x = -1.0;
        y = 1.0;
    } else if (in_vertex_index == 3u) {
        x = 1.0;
        y = 1.0;
    }

    return vec4<f32>(x, y, 0.0, 1.0);
}

fn get_clip_space(frag_pos: vec4<f32>, dimensions: vec2<f32>) -> vec2<f32> {
    var clip_space = frag_pos.xy / dimensions * 2.0;
    clip_space = clip_space - 1.0;
    clip_space = clip_space * vec2<f32>(1.0, -1.0);
    return clip_space;
}

struct Ray {
    pos: vec3<f32>;
    dir: vec3<f32>;
};

fn ray_box_dist(r: Ray, vmin: vec3<f32>, vmax: vec3<f32>) -> f32 {
    let v1 = (vmin.x - r.pos.x) / r.dir.x;
    let v2 = (vmax.x - r.pos.x) / r.dir.x;
    let v3 = (vmin.y - r.pos.y) / r.dir.y;
    let v4 = (vmax.y - r.pos.y) / r.dir.y;
    let v5 = (vmin.z - r.pos.z) / r.dir.z;
    let v6 = (vmax.z - r.pos.z) / r.dir.z;
    let v7 = max(max(min(v1, v2), min(v3, v4)), min(v5, v6));
    let v8 = min(min(max(v1, v2), max(v3, v4)), max(v5, v6));
    if (v8 < 0.0 || v7 > v8) {
        return 0.0;
    }
    
    return v7;
}

struct FSIn {
    [[builtin(position)]] frag_pos: vec4<f32>;
};

fn look_up_pos(pos: vec3<f32>) -> u32 {
    let pos = vec3<u32>((pos * 0.5 + 0.5) * f32(u.cube_size));

    let index = pos.x * u.cube_size * u.cube_size + pos.y * u.cube_size + pos.z;
    return d.data[index];
}

fn unpack_u8(p: u32) -> vec4<u32> {
    return vec4<u32>(
        (p >> u32(24)) & u32(255),
        (p >> u32(16)) & u32(255),
        (p >> u32(8)) & u32(255),
        p & u32(255)
    );
}

fn in_bounds(v: vec3<f32>) -> bool {
    let s = step(vec3<f32>(-1.0), v) - step(vec3<f32>(1.0), v);
    return (s.x * s.y * s.z) > 0.5; 
}

[[stage(fragment)]]
fn fs_main(in: FSIn) -> [[location(0)]] vec4<f32> {
    var output_colour = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let clip_space = get_clip_space(in.frag_pos, u.dimensions.xy);

    let pos = u.camera_inverse * vec4<f32>(clip_space.x, clip_space.y, 0.0, 1.0);
    let dir = u.camera_inverse * vec4<f32>(clip_space.x, clip_space.y, 1.0, 1.0);
    let pos = pos.xyz / pos.w;
    let dir = normalize(dir.xyz / dir.w - pos);
    var ray = Ray(pos.xyz, dir.xyz);

    let dist = ray_box_dist(ray, vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(1.0, 1.0, 1.0));
    if (dist == 0.0) {
        return vec4<f32>(0.1);
    }

    ray.pos = ray.pos + ray.dir * (dist + 0.0001);

    let r_sign = sign(ray.dir);
    let scale = f32(u.cube_size) / 2.0;
    let voxel_size = 2.0 / f32(u.cube_size);
    let t_step = voxel_size / (ray.dir * r_sign);
    let step = voxel_size * r_sign;
    let start_voxel = ceil(ray.pos * scale * r_sign) / scale * r_sign;

    var t_max = (start_voxel - ray.pos) / ray.dir;
    var pos = ray.pos;
    var count = 0;
    var normal = trunc(ray.pos * 1.0001);
    loop {
        let voxel_data = unpack_u8(look_up_pos(pos));
        if (voxel_data.w != u32(0)) {
            break;
        }

        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                t_max.x = t_max.x + t_step.x;
                pos.x = pos.x + step.x;
                normal = vec3<f32>(-r_sign.x, 0.0, 0.0);
            } else {
                t_max.z = t_max.z + t_step.z;
                pos.z = pos.z + step.z;
                normal = vec3<f32>(0.0, 0.0, -r_sign.z);
            }
        } else {
            if (t_max.y < t_max.z) {
                t_max.y = t_max.y + t_step.y;
                pos.y = pos.y + step.y;
                normal = vec3<f32>(0.0, -r_sign.y, 0.0);
            } else {
                t_max.z = t_max.z + t_step.z;
                pos.z = pos.z + step.z;
                normal = vec3<f32>(0.0, 0.0, -r_sign.z);
            }
        }

        if (!in_bounds(pos)) {
            return vec4<f32>(0.1);
        }

        count = count + 1;

        if (count > 500) {
            break;
        }
    }

    let voxel_data = unpack_u8(look_up_pos(pos));
    let voxel_colour = vec3<f32>(
        f32(voxel_data.x),
        f32(voxel_data.y),
        f32(voxel_data.z)
    ) / 255.0;

    let shade = max(clamp(dot(normal, normalize(vec3<f32>(0.6, 1.0, -0.4))), 0.0, 1.0), 0.3);
    let shaded = clamp(shade, 0.0, 1.0) * voxel_colour;
    
    output_colour = vec4<f32>(shaded, 1.0);
    return pow(clamp(output_colour, vec4<f32>(0.0), vec4<f32>(1.0)), vec4<f32>(2.2, 2.2, 2.2, 1.0));
}

// let cube_pos = floor(ray.pos * side_length) / side_length;
// (cube_pos - ray.pos + r_sign * 2.0 / f32(u.cube_size));