use super::*;

#[derive(ugli::Vertex, Copy, Clone, Debug)]
pub struct Vertex {
    pub a_pos: Vec3<f32>,
    pub a_border_distance: f32,
}

impl Add for Vertex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            a_pos: self.a_pos + rhs.a_pos,
            a_border_distance: self.a_border_distance + rhs.a_border_distance,
        }
    }
}

impl Div<f32> for Vertex {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Self {
            a_pos: self.a_pos / rhs,
            a_border_distance: self.a_border_distance / rhs,
        }
    }
}

pub struct Renderer {
    box_geometry: ugli::VertexBuffer<Vertex>,
    program: ugli::Program,
}

impl Renderer {
    pub fn new(geng: &Geng) -> Self {
        Self {
            program: geng
                .shader_lib()
                .compile(include_str!("program.glsl"))
                .unwrap(),
            box_geometry: ugli::VertexBuffer::new_static(geng.ugli(), {
                let mut vs = Vec::new();
                let mut add_face = |v: [Vertex; 4]| {
                    let mut center = (v[0] + v[1] + v[2] + v[3]) / 4.0;
                    center.a_border_distance = 1.0;

                    vs.push(center.clone());
                    vs.push(v[0].clone());
                    vs.push(v[1].clone());

                    vs.push(center.clone());
                    vs.push(v[1].clone());
                    vs.push(v[2].clone());

                    vs.push(center.clone());
                    vs.push(v[2].clone());
                    vs.push(v[3].clone());

                    vs.push(center.clone());
                    vs.push(v[3].clone());
                    vs.push(v[0].clone());
                };
                #[derive(Copy, Clone)]
                enum Coord {
                    Fix(f32),
                    Var(usize),
                }
                let mut add = |x: Coord, y: Coord, z: Coord| {
                    let get_coord = |x: Coord, vs: [f32; 2]| match x {
                        Fix(value) => value,
                        Var(idx) => vs[idx],
                    };

                    add_face([0, 1, 2, 3].map(|idx| {
                        let vs = match idx {
                            0 => [-1.0, -1.0],
                            1 => [-1.0, 1.0],
                            2 => [1.0, 1.0],
                            3 => [1.0, -1.0],
                            _ => unreachable!(),
                        };
                        Vertex {
                            a_pos: vec3(get_coord(x, vs), get_coord(y, vs), get_coord(z, vs)),
                            a_border_distance: 0.0,
                        }
                    }));
                };

                use Coord::*;
                add(Fix(-1.0), Var(0), Var(1));
                add(Fix(1.0), Var(0), Var(1));
                add(Var(0), Fix(-1.0), Var(1));
                add(Var(0), Fix(1.0), Var(1));
                add(Var(0), Var(1), Fix(-1.0));
                add(Var(0), Var(1), Fix(1.0));
                vs
            }),
        }
    }

    pub fn draw(
        &self,
        framebuffer: &mut ugli::Framebuffer,
        camera: &impl geng::AbstractCamera3d,
        block_matrix: Mat4<f32>,
        border_color: Color<f32>,
        color: Color<f32>,
    ) {
        let framebuffer_size = framebuffer.size();
        ugli::draw(
            framebuffer,
            &self.program,
            ugli::DrawMode::Triangles,
            &self.box_geometry,
            (
                ugli::uniforms! {
                    u_model_matrix: block_matrix,
                    u_border_color: border_color,
                    u_color: color,
                },
                geng::camera3d_uniforms(camera, framebuffer_size.map(|x| x as f32)),
            ),
            ugli::DrawParameters {
                depth_func: Some(default()),
                ..default()
            },
        )
    }
}
