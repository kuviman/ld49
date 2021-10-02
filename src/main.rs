use geng::net::simple as simple_net;
use geng::{prelude::*, TextAlign};

mod camera;
mod renderer;

use camera::*;
use renderer::*;

type PlayerId = usize;

#[derive(Debug, Serialize, Deserialize, Diff, Clone, PartialEq)]
struct Player {
    id: PlayerId,
    position: Vec2<f32>,
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
        Mat4::rotate_z(self.rotation)
            * Mat4::translate(self.position.extend(self.layer as f32))
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
            blocks: vec![Block {
                position: vec2(0.0, 0.0),
                rotation: 0.0,
                size: vec2(2.0, 4.0),
                layer: 0,
            }],
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Message {
    UpdatePosition(Vec2<f32>),
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
            position: vec2(
                global_rng().gen_range(-5.0..=5.0),
                global_rng().gen_range(-5.0..=5.0),
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
    prev_mouse_pos: Vec2<f32>,
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
            prev_mouse_pos: Vec2::ZERO,
        }
    }
}

impl geng::State for Game {
    fn update(&mut self, delta_time: f64) {
        self.traffic_watcher.update(&self.model.traffic());
        let delta_time = delta_time as f32;

        self.current_time += delta_time;

        const SPEED: f32 = 10.0;
        if self.geng.window().is_key_pressed(geng::Key::Left)
            || self.geng.window().is_key_pressed(geng::Key::A)
        {
            self.player.position.x -= SPEED * delta_time;
        }
        if self.geng.window().is_key_pressed(geng::Key::Right)
            || self.geng.window().is_key_pressed(geng::Key::D)
        {
            self.player.position.x += SPEED * delta_time;
        }
        if self.geng.window().is_key_pressed(geng::Key::Up)
            || self.geng.window().is_key_pressed(geng::Key::W)
        {
            self.player.position.y += SPEED * delta_time;
        }
        if self.geng.window().is_key_pressed(geng::Key::Down)
            || self.geng.window().is_key_pressed(geng::Key::S)
        {
            self.player.position.y -= SPEED * delta_time;
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
            geng::Event::MouseMove { position } => {
                let position = position.map(|x| x as f32);
                if self
                    .geng
                    .window()
                    .is_button_pressed(geng::MouseButton::Left)
                {
                    let delta = position - self.prev_mouse_pos;
                    const SENS: f32 = 0.01;
                    self.camera.rotation += delta.x * SENS;
                    self.camera.attack_angle += delta.y * SENS;
                }
                self.prev_mouse_pos = position;
                self.geng.window().set_cursor_position()
            }
            _ => {}
        }
    }
}

fn main() {
    logger::init().unwrap();

    geng::net::simple::run("Multiplayer", Model::new, Game::new);
}
