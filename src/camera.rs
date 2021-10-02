use super::*;

pub struct Camera {
    pub look_at: Vec3<f32>,
    pub rotation: f32,
    pub attack_angle: f32,
    pub distance: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            look_at: Vec3::ZERO,
            rotation: 0.0,
            attack_angle: 0.0,
            distance: 0.0,
        }
    }
}

impl geng::AbstractCamera3d for Camera {
    fn view_matrix(&self) -> Mat4<f32> {
        Mat4::translate(vec3(0.0, 0.0, -self.distance))
            * Mat4::rotate_x(-self.attack_angle - f32::PI / 2.0)
            * Mat4::rotate_z(-self.rotation)
            * Mat4::translate(-self.look_at)
    }
    fn projection_matrix(&self, framebuffer_size: Vec2<f32>) -> Mat4<f32> {
        Mat4::perspective(
            f32::PI / 3.0,
            framebuffer_size.x / framebuffer_size.y,
            0.1,
            1000.0,
        )
    }
}
