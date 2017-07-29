// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.


extern crate openvr as vr;
#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::ImmutableBuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, CommandBuffer};
use vulkano::device::Device;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::Subpass;
use vulkano::instance::{Instance, InstanceExtensions, RawInstanceExtensions};
use vulkano::instance::debug::{DebugCallback, MessageTypes};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::format::Format;
use vulkano::image::{AttachmentImage, ImageUsage, ImageAccess};
use vulkano::{VulkanObject, SynchronizedVulkanObject};

use std::sync::Arc;

fn main() {
    let openvr = unsafe { vr::init(vr::ApplicationType::Scene).unwrap() };
    let vr = openvr.system().unwrap();
    let compositor = openvr.compositor().unwrap();
    let origin = vr::TrackingUniverseOrigin::Standing;
    compositor.set_tracking_space(origin);

    let instance = {
        let extensions: RawInstanceExtensions = (&InstanceExtensions {
            ext_debug_report: true,
            ..InstanceExtensions::none()
        }).into();
        let extensions = RawInstanceExtensions::new(compositor.vulkan_instance_extensions_required()).union(&extensions);
        Instance::new(None, extensions, &["VK_LAYER_LUNARG_standard_validation"]).expect("failed to create Vulkan instance")
    };

    let _debug_callback = DebugCallback::new(&instance, MessageTypes {
        error: true,
        warning: true,
        performance_warning: true,
        information: true,
        debug: true,
    }, |msg| {
        let ty = if msg.ty.error {
            "error"
        } else if msg.ty.warning {
            "warning"
        } else if msg.ty.performance_warning {
            "performance_warning"
        } else if msg.ty.information {
            "information"
        } else if msg.ty.debug {
            "debug"
        } else {
            panic!("no-impl");
        };
        if msg.ty.error || msg.ty.warning || msg.ty.performance_warning {
            println!("{} {}: {}", msg.layer_prefix, ty, msg.description);
        }
    }).unwrap();

    //let desired_device = vr.vulkan_output_device().unwrap() as usize;
    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
        .next() //.find(|phys| phys.internal_object() == desired_device)
        .expect("no device available");
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());


    let dimensions = {
        let (width, height) = vr.recommended_render_target_size();
        [width, height]
    };

    let queue = physical.queue_families().find(|&q| {
        q.supports_graphics()
    }).expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            .. vulkano::device::DeviceExtensions::none()
        };

        Device::new(physical, physical.supported_features(), &device_ext,
                    [(queue, 0.5)].iter().cloned()).expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    const FORMAT: Format = Format::B8G8R8A8Srgb;
    let left = AttachmentImage::with_usage(device.clone(), dimensions, FORMAT,
                                           ImageUsage {
                                               transfer_source: true,
                                               sampled: true,
                                               ..ImageUsage::none()
                                           }).unwrap();
    let right = AttachmentImage::with_usage(device.clone(), dimensions, FORMAT,
                                            ImageUsage {
                                                transfer_source: true,
                                                sampled: true,
                                                ..ImageUsage::none()
                                            }).unwrap();

    let (vertex_buffer, _) = {
        #[derive(Debug, Clone)]
        struct Vertex { position: [f32; 2] }
        impl_vertex!(Vertex, position);

        ImmutableBuffer::from_iter([
            Vertex { position: [-0.5, -0.25] },
            Vertex { position: [0.0, 0.5] },
            Vertex { position: [0.25, -0.1] }
        ].iter().cloned(),
        BufferUsage::all(), Some(queue.family()), queue.clone()).expect("failed to create buffer")
    };

    mod vs {
        #[derive(VulkanoShader)]
        #[ty = "vertex"]
        #[src = "
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"]
        struct Dummy;
    }

    mod fs {
        #[derive(VulkanoShader)]
        #[ty = "fragment"]
        #[src = "
#version 450

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"]
        struct Dummy;
    }

    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");

    let render_pass = Arc::new(single_pass_renderpass!(device.clone(),
        attachments: {
            // `color` is a custom name we give to the first and only attachment.
            color: {
                // `load: Clear` means that we ask the GPU to clear the content of this
                // attachment at the start of the drawing.
                load: Clear,
                // `store: Store` means that we ask the GPU to store the output of the draw
                // in the actual image. We could also ask it to discard the result.
                store: Store,
                // `format: <ty>` indicates the type of the format of the image. This has to
                // be one of the types of the `vulkano::format` module (or alternatively one
                // of your structs that implements the `FormatDesc` trait). Here we use the
                // generic `vulkano::format::Format` enum because we don't know the format in
                // advance.
                format: FORMAT,
                // TODO:
                samples: 1,
            }
        },
        pass: {
            // We use the attachment named `color` as the one and only color attachment.
            color: [color],
            // No depth-stencil attachment is indicated with empty brackets.
            depth_stencil: {}
        }
    ).unwrap());

    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer()
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let left_fb = Arc::new(Framebuffer::start(render_pass.clone())
                           .add(left.clone()).unwrap()
                           .build().unwrap());
    let right_fb = Arc::new(Framebuffer::start(render_pass.clone())
                            .add(right.clone()).unwrap()
                            .build().unwrap());

    'outer: loop {
        compositor.wait_get_poses().unwrap();

        for &fb in &[&left_fb, &right_fb] {
            let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
                .begin_render_pass(fb.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()])
                .unwrap()
                .draw(pipeline.clone(),
                      DynamicState {
                          line_width: None,
                          viewports: Some(vec![Viewport {
                              origin: [0.0, 0.0],
                              dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                              depth_range: 0.0 .. 1.0,
                          }]),
                          scissors: None,
                      },
                      vertex_buffer.clone(), (), ())
                .unwrap()
                .end_render_pass()
                .unwrap()
                .build().unwrap();

            let _ = command_buffer.execute(queue.clone()).unwrap();
        }

        unsafe {
            let queue_handle = queue.internal_object_guard();
            compositor.submit(vr::Eye::Left, &vr::compositor::Texture {
                color_space: vr::compositor::texture::ColorSpace::Gamma,
                handle: vr::compositor::texture::Handle::Vulkan(
                    vr::compositor::texture::vulkan::Texture {
                        image: left.inner().image.internal_object() as u64,
                        device: device.internal_object() as *mut _,
                        physical_device: device.physical_device().internal_object() as *mut _,
                        instance: instance.internal_object() as *mut _,
                        queue: *queue_handle as *mut _,
                        queue_family_index: queue.family().id(),
                        width: dimensions[0],
                        height: dimensions[1],
                        format: FORMAT as u32,
                        sample_count: 1,
                    })
            }, None).expect("failed to submit image to OpenVR");
            compositor.submit(vr::Eye::Right, &vr::compositor::Texture {
                color_space: vr::compositor::texture::ColorSpace::Gamma,
                handle: vr::compositor::texture::Handle::Vulkan(
                    vr::compositor::texture::vulkan::Texture {
                        image: right.inner().image.internal_object() as u64,
                        device: device.internal_object() as *mut _,
                        physical_device: device.physical_device().internal_object() as *mut _,
                        instance: instance.internal_object() as *mut _,
                        queue: *queue_handle as *mut _,
                        queue_family_index: queue.family().id(),
                        width: dimensions[0],
                        height: dimensions[1],
                        format: FORMAT as u32,
                        sample_count: 1,
                    })
            }, None).expect("failed to submit image to OpenVR");
        }
        while let Some((info, _pose)) = vr.poll_next_event_with_pose(origin) {
            use vr::system::Event;
            match info.event {
                Event::Quit(_) => {
                    println!("QUITTING");
                    vr.acknowledge_quit_exiting();
                    break 'outer;
                }
                _ => {}
            }
        }
    }
    queue.wait().unwrap();
    // println!("OPENVR SHUTDOWN");
    // unsafe { openvr.shutdown() };
    // println!("VULKAN SHUTDOWN");
}
