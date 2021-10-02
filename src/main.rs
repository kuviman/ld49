use geng::net::simple as simple_net;
use geng::prelude::*;

mod camera;
mod renderer;

use camera::*;
use renderer::*;

type Id = usize;

const EPS: f32 = 1e-5;

const GRAVITY: f32 = 30.0;

#[derive(Debug, Serialize, Deserialize, Diff, Clone, PartialEq)]
struct Player {
    id: Id,
    position: Vec3<f32>,
    jump_speed: f32,
    can_jump: bool,
}

impl HasId for Player {
    type Id = Id;
    fn id(&self) -> &Id {
        &self.id
    }
}

pub struct Intersection {
    pub penetration: f32,
    pub normal: Vec3<f32>,
}

impl Player {
    const RADIUS: f32 = 1.0;
    const HEIGHT: f32 = 6.0;
    const JUMP_INITIAL_SPEED: f32 = 10.0;
    pub fn intersect(&self, block: &Block) -> Option<Intersection> {
        let mut penetration = Self::HEIGHT / 2.0 + 0.5
            - (block.layer as f32 + 0.5 - (self.position.z + Self::HEIGHT / 2.0)).abs();
        let mut normal: Vec3<f32> = vec3(
            0.0,
            0.0,
            if self.position.z + Self::HEIGHT / 2.0 > block.layer as f32 + 0.5 {
                1.0
            } else {
                -1.0
            },
        );
        let delta_pos = Vec2::rotated(self.position.xy() - block.position, -block.rotation);
        let size = block.size + vec2(Self::RADIUS, Self::RADIUS);
        let x_penetration = size.x - delta_pos.x.abs();
        if x_penetration < penetration {
            penetration = x_penetration;
            normal = Vec2::rotated(
                vec2(if delta_pos.x > 0.0 { 1.0 } else { -1.0 }, 0.0),
                block.rotation,
            )
            .extend(0.0);
        }
        let y_penetration = size.y - delta_pos.y.abs();
        if y_penetration < penetration {
            penetration = y_penetration;
            normal = Vec2::rotated(
                vec2(0.0, if delta_pos.y > 0.0 { 1.0 } else { -1.0 }),
                block.rotation,
            )
            .extend(0.0);
        }
        if penetration < 0.0 {
            None
        } else {
            Some(Intersection {
                penetration,
                normal,
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Side {
    pub coord: usize,
    pub positive: bool,
}

#[derive(Debug, Serialize, Deserialize, Diff, Clone, PartialEq)]
pub struct Block {
    pub id: Id,
    pub position: Vec2<f32>,
    pub rotation: f32,
    pub size: Vec2<f32>,
    pub layer: i32,
}

impl HasId for Block {
    type Id = Id;
    fn id(&self) -> &Id {
        &self.id
    }
}

impl Block {
    pub fn matrix(&self) -> Mat4<f32> {
        Mat4::translate(self.position.extend(self.layer as f32 + 0.5))
            * Mat4::rotate_z(self.rotation)
            * Mat4::scale(self.size.extend(0.5))
            * Mat4::scale_uniform(0.99)
    }
    pub fn intersect(&self, mut ray: geng::CameraRay) -> Option<(f32, Side)> {
        let im = self.matrix().inverse();
        ray.from = (im * ray.from.extend(1.0)).xyz();
        ray.dir = (im * ray.dir.extend(0.0)).xyz();
        let mut result = (
            f32::INFINITY,
            Side {
                coord: 0,
                positive: true,
            },
        );
        for coord in 0..3 {
            let from = ray.from[coord];
            let dir = ray.dir[coord];
            if dir.abs() < EPS {
                continue;
            }
            for value in [-1.0, 1.0] {
                // from + dir * t = value
                let t = (value - from) / dir;
                if t > 0.0 {
                    let p = ray.from + ray.dir * t;
                    if p.x.abs() <= 1.0 + EPS && p.y.abs() <= 1.0 + EPS && p.z.abs() <= 1.0 + EPS {
                        if t < result.0 {
                            result = (
                                t,
                                Side {
                                    coord,
                                    positive: value > 0.0,
                                },
                            );
                        }
                    }
                }
            }
        }

        if result.0 == f32::INFINITY {
            None
        } else {
            Some(result)
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Diff, Clone, PartialEq)]
struct Model {
    current_time: f32,
    next_id: Id,
    players: Collection<Player>,
    blocks: Collection<Block>,
}

impl Model {
    fn new() -> Self {
        Self {
            current_time: 0.0,
            next_id: 1,
            players: Collection::new(),
            blocks: Collection::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Message {
    UpdatePosition(Vec3<f32>),
    PlaceBlock(Block),
    DeleteBlock(Id),
}

impl simple_net::Model for Model {
    type PlayerId = Id;
    type Message = Message;
    const TICKS_PER_SECOND: f32 = 20.0;
    fn new_player(&mut self) -> Id {
        let player_id = self.next_id;
        self.next_id += 1;
        self.players.insert(Player {
            id: player_id,
            position: vec3(
                global_rng().gen_range(-5.0..=5.0),
                global_rng().gen_range(-5.0..=5.0),
                0.0,
            ),
            jump_speed: 0.0,
            can_jump: false,
        });
        player_id
    }
    fn drop_player(&mut self, player_id: &Id) {
        self.players.remove(player_id);
    }
    fn handle_message(&mut self, player_id: &Id, message: Self::Message) {
        match message {
            Message::UpdatePosition(position) => {
                self.players.get_mut(player_id).unwrap().position = position;
            }
            Message::PlaceBlock(mut block) => {
                block.id = self.next_id;
                self.next_id += 1;
                self.blocks.insert(block);
            }
            Message::DeleteBlock(id) => {
                self.blocks.remove(&id);
            }
        }
    }
    fn tick(&mut self) {
        self.current_time += 1.0 / Self::TICKS_PER_SECOND;
    }
}

struct Game {
    framebuffer_size: Vec2<usize>,
    geng: Geng,
    traffic_watcher: geng::net::TrafficWatcher,
    next_update: f32,
    player: Player,
    model: simple_net::Remote<Model>,
    current_time: f32,
    renderer: Renderer,
    camera: Camera,
    floor: Block,
}

impl Game {
    fn new(geng: &Geng, player_id: Id, model: simple_net::Remote<Model>) -> Self {
        let current_time = model.get().current_time;
        let player = model.get().players.get(&player_id).unwrap().clone();
        Self {
            framebuffer_size: vec2(1, 1),
            geng: geng.clone(),
            renderer: Renderer::new(geng),
            traffic_watcher: geng::net::TrafficWatcher::new(),
            next_update: 0.0,
            model,
            player,
            current_time,
            camera: Camera::new(),
            floor: Block {
                id: 0,
                position: Vec2::ZERO,
                rotation: 0.0,
                layer: -1,
                size: vec2(100.0, 100.0),
            },
        }
    }
    fn look(&self) -> Option<(Option<Id>, Vec3<f32>, Side)> {
        let ray = self.camera.pixel_ray(vec2(2.0, 2.0), vec2(1.0, 1.0));
        let model = self.model.get();
        let mut closest_t = f32::INFINITY;
        let mut closest_side = Side {
            coord: 0,
            positive: true,
        };
        let mut closest = None;
        for block in &model.blocks {
            if let Some((t, side)) = block.intersect(ray) {
                if t < closest_t {
                    closest_t = t;
                    closest_side = side;
                    closest = Some(block);
                }
            }
        }
        if ray.dir.z.abs() > EPS {
            let t = -ray.from.z / ray.dir.z;
            if t > 0.0 && t < closest_t {
                closest_t = t;
                closest_side = Side {
                    coord: 2,
                    positive: ray.from.z > 0.0,
                };
                closest = None;
            }
        }
        if closest_t == f32::INFINITY {
            None
        } else {
            let block_id = closest.map(|block| block.id);
            Some((block_id, ray.from + ray.dir * closest_t, closest_side))
        }
    }
    fn try_place(&self) -> Option<Block> {
        if let Some((_, pos, side)) = self.look() {
            if side.coord == 2 {
                return Some(Block {
                    id: 0, // Doesn't matter
                    position: pos.xy(),
                    rotation: self.camera.rotation,
                    size: vec2(2.0, 4.0),
                    layer: (pos.z + if side.positive { 0.5 } else { -0.5 }).floor() as i32,
                });
            }
        }
        None
    }
}

impl geng::State for Game {
    fn update(&mut self, delta_time: f64) {
        self.model.update();
        let model = self.model.get();
        self.traffic_watcher.update(&self.model.traffic());
        let delta_time = delta_time as f32;

        self.current_time += delta_time;

        const SPEED: f32 = 10.0;
        let mut direction = Vec2::ZERO;
        if self.geng.window().is_key_pressed(geng::Key::Left)
            || self.geng.window().is_key_pressed(geng::Key::A)
        {
            direction.x -= 1.0;
        }
        if self.geng.window().is_key_pressed(geng::Key::Right)
            || self.geng.window().is_key_pressed(geng::Key::D)
        {
            direction.x += 1.0;
        }
        if self.geng.window().is_key_pressed(geng::Key::Up)
            || self.geng.window().is_key_pressed(geng::Key::W)
        {
            direction.y += 1.0;
        }
        if self.geng.window().is_key_pressed(geng::Key::Down)
            || self.geng.window().is_key_pressed(geng::Key::S)
        {
            direction.y -= 1.0;
        }
        direction = direction.clamp(1.0);
        self.player.position +=
            Vec2::rotated(direction, self.camera.rotation).extend(0.0) * SPEED * delta_time;
        self.player.jump_speed -= GRAVITY * delta_time;
        self.player.position.z += self.player.jump_speed * delta_time;

        self.player.can_jump = false;
        for block in model.blocks.iter().chain(std::iter::once(&self.floor)) {
            if let Some(intersection) = self.player.intersect(block) {
                self.player.position += intersection.normal * intersection.penetration;
                if intersection.normal.z > 0.5 {
                    self.player.can_jump = true;
                }
                if intersection.normal.z.abs() > 0.5 {
                    self.player.jump_speed = 0.0;
                }
            }
        }

        self.next_update -= delta_time;
        if self.next_update < 0.0 {
            while self.next_update < 0.0 {
                self.next_update += 1.0 / <Model as simple_net::Model>::TICKS_PER_SECOND;
            }
            self.model
                .send(Message::UpdatePosition(self.player.position));
        }
    }
    fn draw(&mut self, framebuffer: &mut ugli::Framebuffer) {
        self.framebuffer_size = framebuffer.size();
        self.camera.look_at = self.player.position + vec3(0.0, 0.0, Player::HEIGHT * 0.9);
        ugli::clear(framebuffer, Some(Color::WHITE), Some(1.0));
        let look = self.look();
        let model = self.model.get();
        self.renderer.draw(
            framebuffer,
            &self.camera,
            &self.floor,
            Color::rgb(0.8, 0.8, 0.8),
            Color::rgb(0.8, 0.8, 0.8),
        );
        for block in &model.blocks {
            self.renderer.draw(
                framebuffer,
                &self.camera,
                block,
                Color::BLACK,
                match look {
                    Some((block_id, _, side)) if block_id == Some(block.id) => {
                        if side.coord == 2 {
                            Color::GREEN
                        } else {
                            Color::RED
                        }
                    }
                    _ => Color::WHITE,
                },
            );
        }
        if let Some(block) = self.try_place() {
            self.renderer.draw(
                framebuffer,
                &self.camera,
                &block,
                Color::BLACK,
                Color::TRANSPARENT_BLACK,
            );
        }
        self.geng.draw_2d().circle(
            framebuffer,
            &geng::Camera2d {
                center: Vec2::ZERO,
                rotation: 0.0,
                fov: 200.0,
            },
            Vec2::ZERO,
            1.0,
            Color::BLACK,
        );
        for player in &model.players {
            self.renderer.draw(
                framebuffer,
                &self.camera,
                &Block {
                    id: player.id,
                    position: player.position.xy(),
                    rotation: 0.0,
                    size: vec2(Player::RADIUS, Player::RADIUS),
                    layer: player.position.z as i32,
                },
                Color::BLACK,
                Color::TRANSPARENT_BLACK,
            );
        }
    }
    fn handle_event(&mut self, event: geng::Event) {
        match event {
            geng::Event::MouseMove { delta, .. } => {
                let delta = delta.map(|x| x as f32);
                // if self
                //     .geng
                //     .window()
                //     .is_button_pressed(geng::MouseButton::Left)
                // {
                // let delta = position - self.prev_mouse_pos;
                const SENS: f32 = 0.002;
                self.camera.rotation -= delta.x * SENS;
                self.camera.attack_angle = clamp(
                    self.camera.attack_angle + delta.y * SENS,
                    -f32::PI / 2.0..=f32::PI / 2.0,
                );
                // }
                // self.prev_mouse_pos = position;
            }
            geng::Event::MouseDown {
                button: geng::MouseButton::Left,
                ..
            } => {
                self.geng.window().lock_cursor();
                if let Some(block) = self.try_place() {
                    self.model.send(Message::PlaceBlock(block));
                }
            }
            geng::Event::MouseDown {
                button: geng::MouseButton::Right,
                ..
            } => {
                if let Some((Some(block_id), ..)) = self.look() {
                    self.model.send(Message::DeleteBlock(block_id));
                }
            }
            geng::Event::KeyDown { key } => match key {
                geng::Key::Space => {
                    if dbg!(self.player.can_jump) {
                        self.player.jump_speed = Player::JUMP_INITIAL_SPEED;
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
}

fn main() {
    logger::init().unwrap();

    geng::net::simple::run("Multiplayer", Model::new, Game::new);
}
