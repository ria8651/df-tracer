use cgmath::prelude::*;
use cgmath::{perspective, Deg, Matrix4, Point3, Vector3};
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

struct Sdf(u32, Vec<u8>);

fn main() {
    // Defualt file path that only works on the terminal
    let path = std::path::PathBuf::from("vox/defualt.vox");

    let mut sdf = None;
    if let Ok(bytes) = std::fs::read(path) {
        if let Ok(output) = get_voxels(&bytes) {
            sdf = Some(output);
        }
    }

    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = pollster::block_on(State::new(&window, sdf));

    let now = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        state.egui_platform.handle_event(&event);
        state.input(&event);
        match event {
            Event::RedrawRequested(_) => {
                match state.render(&window) {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
                state.update(now.elapsed().as_secs_f64());
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                match event {
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        // new_inner_size is &&mut so we have to dereference it twice
                        state.resize(**new_inner_size);
                    }
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Q),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    _ => {}
                }
            }
            _ => {}
        }
    });
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    storage_buffer: wgpu::Buffer,
    main_bind_group: wgpu::BindGroup,
    input: Input,
    character: Character,
    previous_frame_time: Option<f64>,
    egui_platform: egui_winit_platform::Platform,
    egui_rpass: egui_wgpu_backend::RenderPass,
    error_string: String,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: &Window, sdf: Option<Sdf>) -> Self {
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits {
                        max_storage_buffer_binding_size: 1024000000,
                        ..Default::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        // println!("Info: {:?}", device.limits().max_storage_buffer_binding_size);

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(
                (concat!(include_str!("common.wgsl"), include_str!("shader.wgsl"))).into(),
            ),
        });

        // #region Buffers
        let bytes = include_bytes!("../vox/defualt.vox");
        let mut sdf = sdf.unwrap_or(get_voxels(bytes).unwrap());
        // To make the buffer fit at least a 256x256x256 model.
        sdf.1
            .extend(std::iter::repeat(0).take(67108864 - sdf.1.len()));

        let uniforms = Uniforms::new(sdf.0);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SDF Buffer"),
            contents: bytemuck::cast_slice(&sdf.1),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let main_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("main_bind_group_layout"),
            });

        let main_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &main_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage_buffer.as_entire_binding(),
                },
            ],
            label: Some("uniform_bind_group"),
        });
        // #endregion

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&main_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let input = Input::new();
        let character = Character::new();

        // egui
        let size = window.inner_size();
        let egui_platform =
            egui_winit_platform::Platform::new(egui_winit_platform::PlatformDescriptor {
                physical_width: size.width as u32,
                physical_height: size.height as u32,
                scale_factor: window.scale_factor(),
                font_definitions: egui::FontDefinitions::default(),
                style: Default::default(),
            });

        // We use the egui_wgpu_backend crate as the render backend.
        let egui_rpass = egui_wgpu_backend::RenderPass::new(&device, config.format, 1);

        let previous_frame_time = None;

        let error_string = "".to_string();

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            uniforms,
            uniform_buffer,
            storage_buffer,
            main_bind_group,
            input,
            character,
            previous_frame_time,
            egui_platform,
            egui_rpass,
            error_string,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state,
                            virtual_keycode,
                            ..
                        },
                    ..
                } => match virtual_keycode {
                    Some(VirtualKeyCode::W) => {
                        self.input.forward = *state == ElementState::Pressed;
                    }
                    Some(VirtualKeyCode::S) => {
                        self.input.backward = *state == ElementState::Pressed;
                    }
                    Some(VirtualKeyCode::D) => {
                        self.input.right = *state == ElementState::Pressed;
                    }
                    Some(VirtualKeyCode::A) => {
                        self.input.left = *state == ElementState::Pressed;
                    }
                    Some(VirtualKeyCode::Space) => {
                        self.input.up = *state == ElementState::Pressed;
                    }
                    Some(VirtualKeyCode::LShift) => {
                        self.input.down = *state == ElementState::Pressed;
                    }
                    _ => {}
                },
                _ => {}
            },
            _ => {}
        }
    }

    fn update(&mut self, time: f64) {
        let input = Vector3::new(
            self.input.right as u32 as f32 - self.input.left as u32 as f32,
            self.input.up as u32 as f32 - self.input.down as u32 as f32,
            self.input.forward as u32 as f32 - self.input.backward as u32 as f32,
        ) * 0.01;

        let forward: Vector3<f32> = -self.character.pos.to_vec().normalize();
        let right = forward.cross(Vector3::new(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(forward);

        self.character.pos += forward * input.z + right * input.x + up * input.y;

        let dimensions = [self.size.width as f32, self.size.height as f32];

        let view = Matrix4::<f32>::look_at_rh(
            self.character.pos,
            Point3::new(0.0, 0.0, 0.0),
            Vector3::unit_y(),
        );
        let proj = perspective(Deg(90.0), dimensions[0] / dimensions[1], 0.001, 1.0);
        let camera = proj * view;
        let camera_inverse = camera.invert().unwrap();

        self.uniforms.dimensions = [dimensions[0], dimensions[1], 0.0, 0.0];
        self.uniforms.camera = camera.into();
        self.uniforms.camera_inverse = camera_inverse.into();

        let fps = if let Some(previous_frame_time) = self.previous_frame_time {
            let fps = 1.0 / (time - previous_frame_time);
            self.previous_frame_time = Some(time);
            fps
        } else {
            self.previous_frame_time = Some(time);
            0.0
        };
        egui::Window::new("Info").show(&self.egui_platform.context(), |ui| {
            ui.label(format!("FPS: {:.0}", fps));
            if ui.button("Open File").clicked() {
                let path = native_dialog::FileDialog::new()
                    .set_location("~/Desktop")
                    .add_filter("Magica Voxel VOX File", &["vox"])
                    .show_open_single_file()
                    .unwrap();

                match path {
                    Some(path) => match std::fs::read(path) {
                        Ok(bytes) => match get_voxels(&bytes) {
                            Ok(sdf) => {
                                self.uniforms.cube_size = sdf.0;
                                self.queue.write_buffer(
                                    &self.storage_buffer,
                                    0,
                                    bytemuck::cast_slice(&sdf.1),
                                );

                                self.error_string = "".to_string();
                            }
                            Err(error) => {
                                self.error_string = error;
                            }
                        },
                        Err(error) => {
                            self.error_string = error.to_string();
                        }
                    },
                    None => self.error_string = "No file selected".to_string(),
                }
            }
            ui.colored_label(
                egui::color::Color32::from_rgb(255, 22, 22),
                self.error_string.clone(),
            );
        });

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );

        self.egui_platform.update_time(time);
    }

    fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let size = window.inner_size();

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Draw my app
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.main_bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        // Draw the UI frame.
        self.egui_platform.begin_frame();

        // End the UI frame. We could now handle the output and draw the UI with the backend.
        let (_output, paint_commands) = self.egui_platform.end_frame(Some(&window));
        let paint_jobs = self.egui_platform.context().tessellate(paint_commands);

        // Upload all resources for the GPU.
        let screen_descriptor = egui_wgpu_backend::ScreenDescriptor {
            physical_width: size.width,
            physical_height: size.height,
            scale_factor: window.scale_factor() as f32,
        };
        self.egui_rpass.update_texture(
            &self.device,
            &self.queue,
            &self.egui_platform.context().font_image(),
        );
        self.egui_rpass
            .update_user_textures(&self.device, &self.queue);
        self.egui_rpass
            .update_buffers(&self.device, &self.queue, &paint_jobs, &screen_descriptor);

        // Record all render passes.
        self.egui_rpass
            .execute(&mut encoder, &view, &paint_jobs, &screen_descriptor, None)
            .unwrap();

        // Submit the command buffer.
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

struct Input {
    forward: bool,
    backward: bool,
    right: bool,
    left: bool,
    up: bool,
    down: bool,
}

impl Input {
    fn new() -> Self {
        Self {
            forward: false,
            backward: false,
            right: false,
            left: false,
            up: false,
            down: false,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    camera: [[f32; 4]; 4],
    camera_inverse: [[f32; 4]; 4],
    dimensions: [f32; 4],
    cube_size: u32,
    junk: [u32; 3],
}

impl Uniforms {
    fn new(cube_size: u32) -> Self {
        Self {
            camera: [[0.0; 4]; 4],
            camera_inverse: [[0.0; 4]; 4],
            dimensions: [0.0, 0.0, 0.0, 0.0],
            cube_size,
            junk: [0; 3],
        }
    }
}

struct Character {
    pos: Point3<f32>,
}

impl Character {
    fn new() -> Self {
        Self {
            pos: Point3::new(0.0, 0.0, -1.5),
        }
    }
}

static ORTH: [Vector3<i32>; 6] = [
    Vector3::new(1, 0, 0),
    Vector3::new(-1, 0, 0),
    Vector3::new(0, 1, 0),
    Vector3::new(0, -1, 0),
    Vector3::new(0, 0, 1),
    Vector3::new(0, 0, -1),
];

fn get_voxels(file: &[u8]) -> Result<Sdf, String> {
    let vox_data = dot_vox::load_bytes(file)?;
    let size = vox_data.models[0].size;
    if size.x != size.y || size.x != size.z || size.y != size.z {
        return Err("Voxel model is not a cube!".to_string());
    }

    let size = size.x as i32;

    // Min 1
    let max_sdf_distance = 16;

    let mut voxels = Vec::new();
    for _ in 0..size {
        for _ in 0..size {
            for _ in 0..size {
                voxels.extend([max_sdf_distance, 0, 0, 0]);
            }
        }
    }

    fn get_index(pos: Vector3<i32>, size: i32) -> Option<usize> {
        if pos.x >= size || pos.y >= size || pos.z >= size || pos.x < 0 || pos.y < 0 || pos.z < 0 {
            return None;
        }

        Some(
            4 * pos.x as usize * size as usize * size as usize + // x
            4 * pos.z as usize * size as usize +                 // z
            4 * pos.y as usize, // y
        )
    }

    for voxel in &vox_data.models[0].voxels {
        let colour = vox_data.palette[voxel.i as usize].to_le_bytes();
        // Magica voxel is flipped for some reason idk
        let pos = Vector3::new(size - voxel.x as i32 - 1, voxel.y as i32, voxel.z as i32);
        let index = get_index(pos, size).unwrap();
        voxels[index] = 0;
        voxels[index + 1] = colour[2];
        voxels[index + 2] = colour[1];
        voxels[index + 3] = colour[0];

        let mut stack: Vec<(u8, Vector3<i32>)> = Vec::new();
        stack.push((0, pos));
        while let Some((sdf_distance, pos)) = stack.pop() {
            for dir in ORTH {
                if let Some(index) = get_index(pos + dir, size) {
                    if voxels[index] > sdf_distance + 1 && voxels[index] <= max_sdf_distance {
                        voxels[index] = sdf_distance + 1;
                        stack.push((sdf_distance + 1, pos + dir));
                    }
                }
            }
        }
    }

    Ok(Sdf(size as u32, voxels))
}
