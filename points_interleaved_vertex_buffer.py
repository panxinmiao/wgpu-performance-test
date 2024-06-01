from wgpu.gui.auto import WgpuCanvas, run
import wgpu
import numpy as np
from fps import FPSRecorder

# Create a canvas to render to
canvas = WgpuCanvas(title="wgpu test", size=(640, 480), max_fps=-1, vsync=False)

# Create a wgpu device
adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
print(adapter.request_adapter_info())
device = adapter.request_device()

# Prepare present context
present_context = canvas.get_context()
render_texture_format = present_context.get_preferred_format(device.adapter)
present_context.configure(device=device, format=render_texture_format)

moudle = device.create_shader_module(
    code="""
      struct Vertex {
        @location(0) position: vec2f,
        @location(1) size: f32,
      };

      struct Uniforms {
        resolution: vec2f,
      };

      struct VSOutput {
        @builtin(position) position: vec4f,
        @location(0) texcoord: vec2f,
      };

      @group(0) @binding(0) var<uniform> uni: Uniforms;

      @vertex 
      fn vs(
          vert: Vertex,
          @builtin(vertex_index) idx: u32,
      ) -> VSOutput {
        var points = array(
          vec2f(-1, -1),
          vec2f( 1, -1),
          vec2f(-1,  1),
          vec2f(-1,  1),
          vec2f( 1, -1),
          vec2f( 1,  1),
        );

        var vsOut: VSOutput;
        let pos = points[idx];
        vsOut.position = vec4f(vert.position + pos *  vert.size / uni.resolution, 0, 1);
        vsOut.texcoord = pos * 0.5 + 0.5;
        return vsOut;
      }

      @group(0) @binding(1) var s: sampler;
      @group(0) @binding(2) var t: texture_2d<f32>;

      @fragment 
      fn fs(vsOut: VSOutput) -> @location(0) vec4f {
        // return textureSample(t, s, vsOut.texcoord);
        let a = select(0.0, 1.0, length(vsOut.texcoord - 0.5) < 0.5 );
        return vec4f(1, 1, 0, 1) * a;
      }
"""
)

pipeline = device.create_render_pipeline(
    layout="auto",
    vertex={
        "module": moudle,
        "entry_point": "vs",
        "buffers": [
            {
                "array_stride": (2 + 1) * 4,  # 3 floats, 4 bytes each
                "step_mode": "instance",
                "attributes": [
                    {"shader_location": 0, "offset": 0, "format": "float32x2"}, # position
                    {"shader_location": 1, "offset": 8, "format": "float32"}, # size
                ],
            }
        ],
    },
    fragment={
        "module": moudle,
        "entry_point": "fs",
        "targets": [
            {
                "format": render_texture_format,
                "blend": {
                    "color": {
                        "src_factor": "src-alpha",
                        "dst_factor": "one-minus-src-alpha",
                        "operation": "add",
                    },
                    "alpha": {
                        "src_factor": "one",
                        "dst_factor": "one-minus-src-alpha",
                        "operation": "add",
                    },
                }, # comment out the blend block to disable blending
            }
        ],
    },
)


num_points = 1000 * 1000 * 15


vertex_data = np.random.rand(num_points, 3)
vertex_data = vertex_data*np.array([2, 2, 31]) - np.array([1, 1, -1])  # scale (x, y, size) to [-1, 1], [-1, 1], [1, 32]
vertex_data = vertex_data.astype(np.float32)

vertex_buffer = device.create_buffer(size=vertex_data.nbytes, usage=wgpu.BufferUsage.VERTEX|wgpu.BufferUsage.COPY_DST)
device.queue.write_buffer(vertex_buffer, 0, vertex_data.tobytes())

uniform_data = np.array([640, 480], dtype=np.float32)
uniform_buffer = device.create_buffer(size=uniform_data.nbytes, usage=wgpu.BufferUsage.UNIFORM|wgpu.BufferUsage.COPY_DST)

bind_group = device.create_bind_group(
    layout=pipeline.get_bind_group_layout(0),
    entries=[
        {"binding": 0, "resource": {"buffer": uniform_buffer, "offset": 0, "size": uniform_data.nbytes}},
        # {"binding": 1, "resource": sampler},
        # {"binding": 2, "resource": texture.create_view()},
    ],
)


render_pass_descriptor = {
    "color_attachments": [
        {
            "clear_value": (0.1, 0.1, 0.1, 1),
            "load_op": "clear",
            "store_op": "store",
        }
    ]
}


def render():
    canvas_texture = present_context.get_current_texture()
    render_pass_descriptor["color_attachments"][0]["view"] = canvas_texture.create_view()
    uniform_data[0] = canvas_texture.size[0]
    uniform_data[1] = canvas_texture.size[1]
    device.queue.write_buffer(uniform_buffer, 0, uniform_data.tobytes())

    command_encoder = device.create_command_encoder()
    render_pass = command_encoder.begin_render_pass(**render_pass_descriptor)
    render_pass.set_pipeline(pipeline)
    render_pass.set_vertex_buffer(0, vertex_buffer)
    render_pass.set_bind_group(0, bind_group, [], 0, 99)
    render_pass.draw(6, num_points)
    render_pass.end()

    device.queue.submit([command_encoder.finish()])


fps = FPSRecorder()

def loop():
    render()
    fps.update()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(loop)
    run()
