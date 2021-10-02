use geng::net::simple as simple_net;
use geng::prelude::*;

mod camera;
mod renderer;

use camera::*;
use renderer::*;

type PlayerId = usize;

#[derive(Debug, Serialize, Deserialize, Diff, Clone, PartialEq)]
struct Player {
    id: PlayerId,
    position: Vec3<f32>,
}

impl HasId for Player {
    type Id = PlayerId;
    fn id(&self) -> &PlayerId {
        &self.id
    }
}

#[derive(Debug, Serialize, Deserialize, Diff, Clone, PartialEq)]
pub struct Block {
    pub position: Vec2<f32>,
    pub rotation: f32,
    pub size: Vec2<f32>,
    pub layer: usize,
}

impl Block {
    pub fn matrix(&self) -> Mat4<f32> {
        Mat4::translate(self.position.extend(self.layer as f32))
            * Mat4::rotate_z(self.rotation)
            * Mat4::scale(self.size.extend(1.0))
    }
}

#[derive(Debug, Serialize, Deserialize, Diff, Clone, PartialEq)]
struct Model {
    current_time: f32,
    next_player_id: PlayerId,
    players: Collection<Player>,
    #[diff = "clone"]
    blocks: Vec<Block>,
}

impl Model {
    fn new() -> Self {
        Self {
            current_time: 0.0,
            next_player_id: 1,
            players: Collection::new(),
            blocks: vec![],
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Message {
    UpdatePosition(Vec3<f32>),
    PlaceBlock(Block),
}

impl simple_net::Model for Model {
    type PlayerId = PlayerId;
    type Message = Message;
    const TICKS_PER_SECOND: f32 = 20.0;
    fn new_player(&mut self) -> Self::PlayerId {
        let player_id = self.next_player_id;
        self.next_player_id += 1;
        self.players.insert(Player {
            id: player_id,
            position: vec3(
                global_rng().gen_range(-5.0..=5.0),
                global_rng().gen_range(-5.0..=5.0),
                0.0,
            ),
        });
        player_id
    }
    fn drop_player(&mut self, player_id: &PlayerId) {
        self.players.remove(player_id);
    }
    fn handle_message(&mut self, player_id: &PlayerId, message: Self::Message) {
        match message {
            Message::UpdatePosition(position) => {
                self.players.get_mut(player_id).unwrap().position = position;
            }
            Message::PlaceBlock(block) => {
                self.blocks.push(block);
            }
        }
    }
    fn tick(&mut self) {
        self.current_time += 1.0 / Self::TICKS_PER_SECOND;
    }
}

struct Game {
    geng: Geng,
    traffic_watcher: geng::net::TrafficWatcher,
    next_update: f32,
    player: Player,
    model: simple_net::Remote<Model>,
    current_time: f32,
    renderer: Renderer,
    camera: Camera,
}

impl Game {
    fn new(geng: &Geng, player_id: PlayerId, model: simple_net::Remote<Model>) -> Self {
        let current_time = model.get().current_time;
        let player = model.get().players.get(&player_id).unwrap().clone();
        Self {
            geng: geng.clone(),
            renderer: Renderer::new(geng),
            traffic_watcher: geng::net::TrafficWatcher::new(),
            next_update: 0.0,
            model,
            player,
            current_time,
            camera: Camera::new(),
        }
    }
}

impl geng::State for Game {
    fn update(&mut self, delta_time: f64) {
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
        self.camera.look_at = self.player.position + vec3(0.0, 0.0, 5.0);
        ugli::clear(framebuffer, Some(Color::BLACK), Some(1.0));
        let model = self.model.get();
        for block in &model.blocks {
            self.renderer.draw(framebuffer, &self.camera, block);
        }
        // for player in &model.players {
        //     self.geng
        //         .draw_2d()
        //         .circle(framebuffer, &camera, player.position, 1.0, Color::GRAY);
        // }
        // self.geng.draw_2d().circle(
        //     framebuffer,
        //     &camera,
        //     self.player.position,
        //     1.0,
        //     Color::WHITE,
        // );
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

                let ray = self.camera.pixel_ray(vec2(2.0, 2.0), vec2(1.0, 1.0));
            }
            geng::Event::KeyDown {
                key: geng::Key::Space,
            } => {
                self.model.send(Message::PlaceBlock(Block {
                    position: self.player.position.xy(),
                    rotation: self.camera.rotation,
                    size: vec2(2.0, 4.0),
                    layer: self.player.position.z as usize,
                }));
            }
            _ => {}
        }
    }
}

fn main() {
    logger::init().unwrap();

    geng::net::simple::run("Multiplayer", Model::new, Game::new);
}
