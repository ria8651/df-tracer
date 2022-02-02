struct Uniforms {
    camera: mat4x4<f32>;
    camera_inverse: mat4x4<f32>;
    dimensions: vec4<f32>;
    sun_dir: vec4<f32>;
    cube_size: u32;
    soft_shadows: bool;
    ao: bool;
    steps: bool;
};

[[group(0), binding(0)]]
var<uniform> u: Uniforms;

struct Data {
    data: [[stride(4)]] array<u32>;
};

[[group(0), binding(1)]]
var df_texture: texture_3d<f32>;
[[group(0), binding(2)]]
var nearest_sampler: sampler;

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

fn look_up_pos(pos: vec3<f32>) -> vec4<f32> {
    let pos = vec3<f32>(pos * 0.5 + 0.5);
    return textureSample(df_texture, nearest_sampler, pos);
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

struct Hit {
    hit: bool;
    pos: vec3<f32>;
    colour: vec3<f32>;
    normal: vec3<f32>;
    steps: u32;
    closest_ratio: f32;
};

fn sdf_ray(ray: Ray) -> Hit {
    var dist = 0.0;
    if (!in_bounds(ray.pos)) {
        dist = ray_box_dist(ray, vec3<f32>(-1.0, -1.0, -1.0), vec3<f32>(1.0, 1.0, 1.0));
        if (dist == 0.0) {
            return Hit(false, vec3<f32>(0.0), vec3<f32>(1.0), vec3<f32>(0.0), u32(0), 0.0);
        }
    }

    let pos = ray.pos + ray.dir * (dist + 0.0001);

    let r_sign = sign(ray.dir);
    let scale = f32(u.cube_size) / 2.0;
    let voxel_size = 2.0 / f32(u.cube_size);
    let t_step = voxel_size / (ray.dir * r_sign);
    let step = voxel_size * r_sign;
    let start_voxel = ceil(pos * scale * r_sign) / scale * r_sign;

    var t_max = (start_voxel - pos) / ray.dir;
    var t_current = 0.0;
    var count = 0;
    var normal = trunc(pos * 1.0001);
    var voxel_pos = start_voxel - step / 2.0;
    var closest_ratio = 1000.0;
    loop {
        let voxel_data = look_up_pos(voxel_pos);
        if (voxel_data.w == 0.0) {
            break;
        }

        let sdf_distance = u32(voxel_data.w * 255.0);
        if (sdf_distance > u32(3)) {
            // https://www.desmos.com/calculator/rl7xhy6cj9
            voxel_pos = voxel_pos + voxel_size * (0.54 * f32(sdf_distance) - 1.73) * ray.dir;
        } else {
            let start_voxel = ceil(voxel_pos * scale * r_sign) / scale * r_sign;
            voxel_pos = start_voxel - step / 2.0;

            t_max = (start_voxel - pos) / ray.dir;

            // https://www.shadertoy.com/view/4dX3zl (good old shader toy)
            var mask = vec3<f32>(t_max.xyz <= min(t_max.yzx, t_max.zxy));
            normal = mask * -r_sign;

            let t_current = dot(t_max * mask, vec3<f32>(1.0));
            voxel_pos = pos + ray.dir * t_current - normal * 0.001;
        }

        let new_closest_ratio = f32(sdf_distance) / length(voxel_pos - ray.pos);
        closest_ratio = min(closest_ratio, new_closest_ratio);

        if (!in_bounds(voxel_pos)) {
            return Hit(false, vec3<f32>(0.0), vec3<f32>(0.8), vec3<f32>(0.0), u32(count), closest_ratio);
        }

        count = count + 1;
        // worst case senario for 256x256x256
        if (count > 500) {
            return Hit(false, vec3<f32>(0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0), u32(count), closest_ratio);
        }
    }

    let voxel_colour = look_up_pos(voxel_pos).rgb;
    return Hit(true, voxel_pos, voxel_colour, normal, u32(count), closest_ratio);
}

[[stage(fragment)]]
fn fs_main(in: FSIn) -> [[location(0)]] vec4<f32> {
    var output_colour = vec3<f32>(0.0, 0.0, 0.0);
    let clip_space = get_clip_space(in.frag_pos, u.dimensions.xy);

    let pos = u.camera_inverse * vec4<f32>(clip_space.x, clip_space.y, 0.0, 1.0);
    let dir = u.camera_inverse * vec4<f32>(clip_space.x, clip_space.y, 1.0, 1.0);
    let pos = pos.xyz / pos.w;
    let dir = normalize(dir.xyz / dir.w - pos);
    var ray = Ray(pos.xyz, dir.xyz);

    let hit = sdf_ray(ray);
    if (hit.hit) {
        let sun_dir = normalize(u.sun_dir.xyz);

        let ambient = 0.3;
        var diffuse = max(dot(hit.normal, -sun_dir), 0.0);

        // var v = 0.0;
        // let s = 1;
        // for (var i: i32 = 0; i < s; i = i + 1) {
        //     let shadow_hit = sdf_ray(Ray(hit.pos + hit.normal * 0.0156, -sun_dir));
        //     if (!shadow_hit.hit) {
        //         v = v + 1.0 / f32(s);
        //     }
        // }
        // diffuse = diffuse * v;

        let shadow_hit = sdf_ray(Ray(hit.pos + hit.normal * 0.0156, -sun_dir));
        if (shadow_hit.hit) {
            diffuse = 0.0;
        } else {
            if (u.soft_shadows) {
                diffuse = diffuse * clamp(shadow_hit.closest_ratio / 10.0, 0.0, 1.0);
            }
        }

        output_colour = (ambient + diffuse) * hit.colour;
    } else {
        if (u.steps) {
            output_colour = vec3<f32>(f32(hit.steps) / 64.0);
        } else {
            output_colour =  vec3<f32>(0.2);
        }
    }

    // output_colour = vec3<f32>(textureSample(df_texture, nearest_sampler, vec3<f32>(clip_space * 0.5 + 0.5, 0.5)).rgb);
    
    return vec4<f32>(pow(clamp(output_colour, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2)), 0.5);
}