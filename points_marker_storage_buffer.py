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
      struct Uniforms {
        resolution: vec2f,
      };

      struct VSOutput {
        @builtin(position) position: vec4f,
        @location(0) texcoord: vec2f,
      };

      @group(0) @binding(0) var<uniform> uni: Uniforms;
      @group(0) @binding(1) var<storage, read> positions: array<vec2f>;
      @group(0) @binding(2) var<storage, read> sizes: array<f32>;

      @vertex 
      fn vs(
          @builtin(vertex_index) idx: u32,
      ) -> VSOutput {

        let node_index = idx / 6;
        let vertex_index = idx % 6;

        let position = positions[node_index];
        let size = sizes[node_index];
    
        var points = array(
          vec2f(-1, -1),
          vec2f( 1, -1),
          vec2f(-1,  1),
          vec2f(-1,  1),
          vec2f( 1, -1),
          vec2f( 1,  1),
        );

        var vsOut: VSOutput;
        let pos = points[vertex_index];
        vsOut.position = vec4f(position + pos *  size / uni.resolution, 0, 1);
        vsOut.texcoord = pos * 0.5 + 0.5;
        return vsOut;
      }

      @fragment 
      fn fs(vsOut: VSOutput) -> @location(0) vec4f {
        // return textureSample(t, s, vsOut.texcoord);
        // let a = select(0.0, 1.0, length(vsOut.texcoord - 0.5) < 0.5 );
        let s = get_signed_distance(vsOut.texcoord - 0.5);
        let a = select(0.0, 1.0, s < 0.0 );
        return vec4f(1, 1, 0, 1) * a;
      }


      fn get_signed_distance(coord: vec2f) -> f32 {
            let m_pi = 3.141592653589793;

            let t1 = -m_pi/2.0;
            let c1 = 0.225*vec2<f32>(cos(t1),sin(t1));
            let t2 = t1+2*m_pi/3.0;
            let c2 = 0.225*vec2<f32>(cos(t2),sin(t2));
            let t3 = t2+2*m_pi/3.0;
            let c3 = 0.225*vec2<f32>(cos(t3),sin(t3));
            let r1 = length( coord - c1) - 1/4.25;
            let r2 = length( coord - c2) - 1/4.25;
            let r3 = length( coord - c3) - 1/4.25;
            let r4 = min(min(r1,r2),r3);
            // Root (2 circles and 2 half-planes)
            let c4 = vec2<f32>( 0.65, 0.125);
            let c5 = vec2<f32>(-0.65, 0.125);
            let r5 = length(coord-c4) - 1/1.6;
            let r6 = length(coord-c5) - 1/1.6;
            let r7 = coord.y - 0.5;
            let r8 = 0.2 - coord.y;
            let r9 = max(-min(r5,r6), max(r7,r8));
            return min(r4,r9);

      }
"""
)

pipeline = device.create_render_pipeline(
    layout="auto",
    vertex={
        "module": moudle,
        "entry_point": "vs",
        "buffers": [],
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
                    }, # comment out the blend block to disable blending
                },
            }
        ],
    },
)



num_points = 1000 * 1000 * 15


# vertex_data = np.random.rand(num_points, 4)

# vertex_data = vertex_data*np.array([2, 2, 31, 0]) - np.array([1, 1, -1, 0])  # scale (x, y, size) to [-1, 1], [-1, 1], [1, 32]
# vertex_data = vertex_data.astype(np.float32)

# vertex_buffer = device.create_buffer(size=vertex_data.nbytes, usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_DST)
# device.queue.write_buffer(vertex_buffer, 0, vertex_data.tobytes())

positions_data = np.random.rand(num_points, 4)
positions_data = positions_data*np.array([2, 2, 0, 0]) - np.array([1, 1, 0, 0])  # scale (x, y) to [-1, 1], [-1, 1]
positions_data = positions_data.astype(np.float32)

positions_buffer = device.create_buffer(size=positions_data.nbytes, usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_DST)
device.queue.write_buffer(positions_buffer, 0, positions_data.tobytes())

sizes_data = np.random.rand(num_points, 4)
sizes_data = sizes_data*np.array([31, 0, 0, 0]) + np.array([1, 0, 0, 0])  # scale (size) to [1, 32]
sizes_data = sizes_data.astype(np.float32)
sizes_buffer = device.create_buffer(size=sizes_data.nbytes, usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_DST)
device.queue.write_buffer(sizes_buffer, 0, sizes_data.tobytes())

uniform_data = np.array([640, 480], dtype=np.float32)
uniform_buffer = device.create_buffer(size=uniform_data.nbytes, usage=wgpu.BufferUsage.UNIFORM|wgpu.BufferUsage.COPY_DST)

bind_group = device.create_bind_group(
    layout=pipeline.get_bind_group_layout(0),
    entries=[
        {"binding": 0, "resource": {"buffer": uniform_buffer, "offset": 0, "size": uniform_data.nbytes}},
        {"binding": 1, "resource": {"buffer": positions_buffer, "offset": 0, "size": positions_data.nbytes}},
        {"binding": 2, "resource": {"buffer": sizes_buffer, "offset": 0, "size": sizes_data.nbytes}},
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
    # render_pass.set_vertex_buffer(0, vertex_buffer)
    render_pass.set_bind_group(0, bind_group, [], 0, 99)
    render_pass.draw(6 * num_points, 1)
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
