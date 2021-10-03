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
    #[diff = "eq"]
    name: String,
    position: Vec3<f32>,
    rotation: f32,
    attack_angle: f32,
    jump_speed: f32,
    can_jump: bool,
}

#[derive(Deref, DerefMut)]
struct InterpolatedPlayer {
    #[deref]
    #[deref_mut]
    player: Player,
    swing_amp: f32,
    t: f32,
}

impl InterpolatedPlayer {
    fn swing(&self) -> f32 {
        (self.t * 10.0).sin() * self.swing_amp
    }
}

impl HasId for InterpolatedPlayer {
    type Id = Id;
    fn id(&self) -> &Id {
        &self.id
    }
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
    const JUMP_INITIAL_SPEED: f32 = 15.0;
    pub fn intersect(&self, block: &Block) -> Option<Intersection> {
        let vertical_penetration = Self::HEIGHT / 2.0 + 0.5
            - (block.layer as f32 + 0.5 - (self.position.z + Self::HEIGHT / 2.0)).abs();
        let vertical_normal: Vec3<f32> = vec3(
            0.0,
            0.0,
            if self.position.z + Self::HEIGHT / 2.0 > block.layer as f32 + 0.5 {
                1.0
            } else {
                -1.0
            },
        );
        let mut penetration = vertical_penetration;
        let mut normal = vertical_normal;
        let delta_pos = Vec2::rotate(self.position.xy() - block.position, -block.rotation);
        let size = block.size + vec2(Self::RADIUS, Self::RADIUS);
        let x_penetration = size.x - delta_pos.x.abs();
        if x_penetration < penetration {
            penetration = x_penetration;
            normal = Vec2::rotate(
                vec2(if delta_pos.x > 0.0 { 1.0 } else { -1.0 }, 0.0),
                block.rotation,
            )
            .extend(0.0);
        }
        let y_penetration = size.y - delta_pos.y.abs();
        if y_penetration < penetration {
            penetration = y_penetration;
            normal = Vec2::rotate(
                vec2(0.0, if delta_pos.y > 0.0 { 1.0 } else { -1.0 }),
                block.rotation,
            )
            .extend(0.0);
        }

        // if vertical_normal.z > 0.0
        //     && vertical_penetration < 1.5
        //     && vertical_penetration > 0.0
        //     && vertical_penetration - 1.5 < penetration
        // {
        //     penetration = vertical_penetration;
        //     normal = vertical_normal;
        // }

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

fn intersect_segments(s1: [Vec2<f32>; 2], s2: [Vec2<f32>; 2]) -> Option<Vec2<f32>> {
    let p1 = s1[0];
    let e1 = s1[1] - s1[0];
    let p2 = s2[0];
    let e2 = s2[1] - s2[0];

    if Vec2::skew(e1, e2).abs() < EPS {
        return None;
    }

    // skew(p1 + e1 * t - p2, e2)
    // t = skew(p2 - p1, e2) / skew(e1, e2)
    let t = Vec2::skew(p2 - p1, e2) / Vec2::skew(e1, e2);
    if t < 0.0 || t > 1.0 {
        return None;
    }
    let p = p1 + e1 * t;
    let t = Vec2::skew(p1 - p2, e1) / Vec2::skew(e2, e1);
    if t < 0.0 || t > 1.0 {
        return None;
    }
    Some(p)
}

fn is_inside(p: Vec2<f32>, vs: &[Vec2<f32>]) -> bool {
    if vs.len() <= 3 {
        return false;
    }
    for i in 0..vs.len() - 1 {
        if Vec2::skew(vs[i + 1] - vs[i], p - vs[i]) < -EPS {
            return false;
        }
    }
    true
}

fn convex_hull(mut a: Vec<Vec2<f32>>) -> Vec<Vec2<f32>> {
    a.sort_by_key(|p| r32(p.x));
    let mut top = Vec::new();
    let mut bottom: Vec<Vec2<f32>> = Vec::new();
    for p in a {
        if let Some(&prev) = bottom.last() {
            if (prev - p).len() < EPS {
                continue;
            }
        }
        while bottom.len() >= 2
            && Vec2::skew(
                bottom[bottom.len() - 1] - bottom[bottom.len() - 2],
                p - bottom[bottom.len() - 1],
            ) < 0.0
        {
            bottom.pop();
        }
        bottom.push(p);
        while top.len() >= 2
            && Vec2::skew(
                top[top.len() - 1] - top[top.len() - 2],
                p - top[top.len() - 1],
            ) > 0.0
        {
            top.pop();
        }
        top.push(p);
    }
    top.pop();
    top.reverse();
    let mut result = bottom;
    result.extend(top);
    result
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
    pub fn points_2d(&self) -> Vec<Vec2<f32>> {
        let e1 = Vec2::rotate(vec2(1.0, 0.0), self.rotation);
        let e2 = Vec2::rotate_90(e1);

        let sx = e1 * self.size.x;
        let sy = e2 * self.size.y;

        let mut result = vec![sx + sy, -sx + sy, -sx - sy, sx - sy];
        for p in &mut result {
            *p += self.position;
        }
        result
    }
    pub fn intersect_2d(&self, other: &Self) -> Vec<Vec2<f32>> {
        let mut p1 = self.points_2d();
        p1.push(p1[0]);
        let mut p2 = other.points_2d();
        p2.push(p2[0]);
        let mut result = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                if let Some(p) = intersect_segments([p1[i], p1[i + 1]], [p2[j], p2[j + 1]]) {
                    result.push(p);
                }
            }
        }
        for &p in &p1[..4] {
            if is_inside(p, &p2) {
                result.push(p);
            }
        }
        for &p in &p2[..4] {
            if is_inside(p, &p1) {
                result.push(p);
            }
        }
        result
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
        let mut result = Self {
            current_time: 0.0,
            next_id: 1,
            players: Collection::new(),
            blocks: Collection::new(),
        };
        for _ in 0..100 {
            let pos = vec2(
                global_rng().gen_range(-100.0..=100.0),
                global_rng().gen_range(-100.0..=100.0),
            );
            let rotation = global_rng().gen_range(0.0..=2.0 * f32::PI);
            let block = Block {
                id: result.next_id,
                position: pos,
                rotation,
                layer: 0,
                size: vec2(1.0, 3.0) * global_rng().gen_range(0.5..=2.0),
            };
            let mut can_place = true;
            for other in &result.blocks {
                if !block.intersect_2d(other).is_empty() {
                    can_place = false;
                    break;
                }
            }
            if can_place {
                result.blocks.insert(block);
            }
            result.next_id += 1;
        }
        result
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Message {
    UpdatePosition(Vec3<f32>, f32, f32),
    PlaceBlock(Block),
    DeleteBlock(Id),
    ChangeName(String),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
enum Event {
    BlockFall(Block, Option<(Vec2<f32>, Vec2<f32>, i32)>),
}

impl simple_net::Model for Model {
    type PlayerId = Id;
    type Message = Message;
    type Event = Event;
    const TICKS_PER_SECOND: f32 = 5.0;
    fn new_player(&mut self) -> Id {
        let player_id = self.next_id;
        self.next_id += 1;
        self.players.insert(Player {
            id: player_id,
            attack_angle: 0.0,
            name: "noname".to_owned(),
            rotation: 0.0,
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
            Message::UpdatePosition(position, rotation, attack_angle) => {
                let player = self.players.get_mut(player_id).unwrap();
                player.position = position;
                player.rotation = rotation;
                player.attack_angle = attack_angle;
            }
            Message::PlaceBlock(mut block) => {
                for other in &self.blocks {
                    if other.layer == block.layer && !block.intersect_2d(other).is_empty() {
                        return;
                    }
                }
                block.id = self.next_id;
                self.next_id += 1;
                self.blocks.insert(block);
            }
            Message::DeleteBlock(id) => {
                self.blocks.remove(&id);
            }
            Message::ChangeName(name) => {
                self.players.get_mut(player_id).unwrap().name = name;
            }
        }
    }
    fn tick(&mut self, events: &mut Vec<Event>) {
        self.current_time += 1.0 / Self::TICKS_PER_SECOND;
        let mut blocks: Vec<Block> = self.blocks.iter().cloned().collect();
        blocks.sort_by_key(|block| -block.layer);

        let mut next: Vec<usize> = (0..blocks.len()).collect();
        let mut top = vec![vec![]; blocks.len()];
        let mut processed = vec![false; blocks.len()];

        for i in 0..blocks.len() {
            if processed[i] {
                continue;
            }

            {
                for j in 0..blocks.len() {
                    let mut t = j;
                    while next[t] != t {
                        t = next[t];
                    }
                    next[j] = t;
                }
                let mut current = Vec::new();
                for j in 0..blocks.len() {
                    if next[j] == next[i] {
                        current.push(j);
                    }
                }
                for window in current.windows(2) {
                    next[window[0]] = window[1];
                }
                let last = *current.last().unwrap();
                next[last] = last;
            }

            let current_layer = blocks[i].layer;

            let mut support_points = Vec::new();
            let mut support_blocks = Vec::new();
            let mut current_blocks = Vec::new();
            {
                fn go_up(
                    i: usize,
                    top: &[Vec<usize>],
                    current_blocks: &mut Vec<usize>,
                    next: &[usize],
                ) {
                    current_blocks.push(i);
                    for &i in &top[i] {
                        go_up(i, top, current_blocks, next);
                        let mut i = i;
                        loop {
                            if next[i] == i {
                                break;
                            }
                            i = next[i];
                        }
                    }
                }
                go_up(i, &top, &mut current_blocks, &next);
            }

            {
                let mut i = i;
                loop {
                    current_blocks.push(i);
                    processed[i] = true;
                    let block = &blocks[i];
                    if block.layer == 0 {
                        support_points.extend(block.points_2d());
                    }
                    for j in 0..blocks.len() {
                        let other = &blocks[j];
                        if other.layer == block.layer - 1 {
                            let intersection = block.intersect_2d(other);
                            if !intersection.is_empty() {
                                support_blocks.push(j);
                                support_points.extend(intersection);
                            }
                        }
                    }
                    if next[i] == i {
                        break;
                    }
                    i = next[i];
                }
            }

            support_blocks.sort();
            support_blocks.dedup();
            for window in support_blocks.windows(2) {
                let mut a = window[0];
                while next[a] != a {
                    a = next[a];
                }
                let mut b = window[1];
                while next[b] != b {
                    b = next[b];
                }
                next[a] = b;
            }
            for &block in &support_blocks {
                top[block].push(i);
            }

            let mut center_of_mass = Vec2::ZERO;
            let mut total_weight = 0.0;
            for &block in &current_blocks {
                let block = &blocks[block];
                center_of_mass += block.position * block.size.x * block.size.y;
                total_weight += block.size.x * block.size.y;
            }
            center_of_mass /= total_weight;

            let support = convex_hull(support_points);
            if !is_inside(center_of_mass, &support) {
                let mut edge = None;
                if support.len() > 3 {
                    let mut min_skew = 0.0;
                    for i in 0..support.len() - 1 {
                        let skew =
                            Vec2::skew(support[i + 1] - support[i], center_of_mass - support[i]);
                        if skew < min_skew {
                            min_skew = skew;
                            edge = Some((support[i], support[i + 1], current_layer));
                        }
                    }
                }
                for block in current_blocks {
                    if let Some(block) = self.blocks.remove(&blocks[block].id) {
                        events.push(Event::BlockFall(block, edge));
                    }
                }
                return;
            }
        }
    }
}

struct FallingBlock {
    block: Block,
    t: f32,
    edge: Option<(Vec2<f32>, Vec2<f32>, i32)>,
}

struct Game {
    framebuffer_size: Vec2<usize>,
    geng: Geng,
    traffic_watcher: geng::net::TrafficWatcher,
    next_update: f32,
    player: Player,
    model: simple_net::Remote<Model>,
    falling_blocks: Vec<FallingBlock>,
    current_time: f32,
    renderer: Renderer,
    camera: Camera,
    floor: Block,
    place_rotation: f32,
    block_size: f32,
    interpolated_players: Collection<InterpolatedPlayer>,
    editing: bool,
    assets: Assets,
    last_blocks: HashSet<Id>,
    show_names: bool,
}
const PLACE_DISTANCE: f32 = 20.0;

impl Game {
    fn play_sound(&self, sound: &geng::Sound, pos: Vec3<f32>) {
        let mut effect = sound.effect();
        let vol = 1.01f32.powf(-(pos - self.camera.look_at).len()) as f64;
        effect.set_volume(vol);
        println!("{:?}", vol);
        effect.play();
    }
    fn new(geng: &Geng, player_id: Id, model: simple_net::Remote<Model>, assets: Assets) -> Self {
        let current_time = model.get().current_time;
        let player = model.get().players.get(&player_id).unwrap().clone();
        let last_blocks = model.get().blocks.iter().map(|block| block.id).collect();
        Self {
            show_names: true,
            assets,
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
                size: vec2(1000.0, 1000.0),
            },
            place_rotation: 0.0,
            falling_blocks: Vec::new(),
            block_size: 0.5,
            interpolated_players: Collection::new(),
            editing: false,
            last_blocks,
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
    fn try_place_at(&self, pos: Vec2<f32>, layer: i32) -> (Block, bool) {
        let block = Block {
            id: 0, // Doesn't matter
            position: pos,
            rotation: self.camera.rotation + self.place_rotation,
            size: vec2(1.0, 3.0) * self.block_size,
            layer,
        };
        for other in &self.model.get().blocks {
            if block.layer == other.layer && !block.intersect_2d(other).is_empty() {
                return (block, false);
            }
        }
        (block, true)
    }
    fn try_place(&self) -> Option<(Block, bool)> {
        let ray = self.camera.pixel_ray(vec2(2.0, 2.0), vec2(1.0, 1.0));
        if ray.dir.z.abs() < EPS {
            return None;
        }
        let mut pos = ray.from;
        while (pos - ray.from).len() < PLACE_DISTANCE {
            {
                let layer = (pos.z + if ray.dir.z > 0.0 { -0.5 } else { 0.5 }).floor() as i32;
                let pos = pos.xy();
                let (block, can_place) = self.try_place_at(pos, layer);
                if can_place {
                    if block.layer == 0 {
                        return Some((block, true));
                    }
                    for other in &self.model.get().blocks {
                        if block.layer == other.layer + 1 && !block.intersect_2d(other).is_empty() {
                            return Some((block, true));
                        }
                    }
                }
            }
            pos += ray.dir / ray.dir.z.abs();
        }
        if let Some((_, pos, side)) = self.look() {
            if side.coord == 2 && (pos - ray.from).len() < PLACE_DISTANCE {
                let layer = (pos.z + if side.positive { 0.5 } else { -0.5 }).floor() as i32;
                let pos = pos.xy();
                for add in 0..1 {
                    let (block, can_place) = self.try_place_at(pos, layer + add);
                    if can_place {
                        return Some((block, true));
                    }
                }
                return Some(self.try_place_at(pos, layer));
            }
        }
        None
    }
}

impl geng::State for Game {
    fn update(&mut self, delta_time: f64) {
        let mut falled_played = false;
        for event in self.model.update() {
            match event {
                Event::BlockFall(block, edge) => {
                    if !falled_played {
                        falled_played = true;
                        self.play_sound(
                            &self.assets.fall,
                            block.position.extend(block.layer as f32),
                        );
                    }
                    self.falling_blocks.push(FallingBlock {
                        block,
                        t: 0.0,
                        edge,
                    });
                }
            }
        }
        let model = self.model.get();
        let blocks: HashSet<Id> = model.blocks.iter().map(|block| block.id).collect();
        for &block in &blocks {
            if !self.last_blocks.contains(&block) {
                let block = model.blocks.get(&block).unwrap();
                self.play_sound(
                    &self.assets.place,
                    block.position.extend(block.layer as f32),
                );
            }
        }
        self.last_blocks = blocks;
        self.traffic_watcher.update(&self.model.traffic());
        let delta_time = delta_time as f32;

        self.current_time += delta_time;

        const SPEED: f32 = 10.0;
        let mut direction = Vec2::ZERO;
        if !self.editing {
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
        }
        direction = direction.clamp(1.0);
        if self.geng.window().is_key_pressed(geng::Key::PageDown) {
            self.block_size -= delta_time;
        }
        if self.geng.window().is_key_pressed(geng::Key::PageUp) {
            self.block_size += delta_time;
        }
        self.block_size = clamp(self.block_size, 0.5..=2.0);
        self.player.position +=
            Vec2::rotate(direction, self.camera.rotation).extend(0.0) * SPEED * delta_time;
        self.player.jump_speed -= GRAVITY * delta_time;
        self.player.position.z += self.player.jump_speed * delta_time;

        self.player.can_jump = false;
        for block in model.blocks.iter() {
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

        if self.player.position.z < 0.0 {
            self.player.position.z = 0.0;
            self.player.jump_speed = 0.0;
            self.player.can_jump = true;
        }

        self.next_update -= delta_time;
        if self.next_update < 0.0 {
            while self.next_update < 0.0 {
                self.next_update += 1.0 / <Model as simple_net::Model>::TICKS_PER_SECOND;
            }
            self.model.send(Message::UpdatePosition(
                self.player.position,
                self.camera.rotation,
                self.camera.attack_angle,
            ));
        }

        for block in &mut self.falling_blocks {
            block.t += delta_time;
        }
        self.falling_blocks.retain(|block| block.t < 10.0);

        self.interpolated_players
            .retain(|player| model.players.get(&player.id).is_some());
        for player in &model.players {
            if self.interpolated_players.get(&player.id).is_none() {
                self.interpolated_players.insert(InterpolatedPlayer {
                    player: player.clone(),
                    t: 0.0,
                    swing_amp: 0.0,
                });
            }
            let interpolated = self.interpolated_players.get_mut(&player.id).unwrap();
            interpolated.swing_amp += ((player.position - interpolated.position).len().min(1.0)
                - interpolated.swing_amp)
                * (delta_time * 10.0).min(1.0);
            interpolated.t += delta_time;
            let interpolated = &mut **interpolated;
            interpolated.position +=
                (player.position - interpolated.position) * (delta_time * 5.0).min(1.0);
            interpolated.rotation +=
                (player.rotation - interpolated.rotation) * (delta_time * 5.0).min(1.0);
            interpolated.attack_angle +=
                (player.attack_angle - interpolated.attack_angle) * (delta_time * 5.0).min(1.0);
        }
    }
    fn draw(&mut self, framebuffer: &mut ugli::Framebuffer) {
        self.framebuffer_size = framebuffer.size();
        self.camera.look_at = self.player.position + vec3(0.0, 0.0, Player::HEIGHT * 0.9);
        ugli::clear(framebuffer, Some(Color::rgb(0.8, 0.8, 1.0)), Some(1.0));
        let look = self.look();
        let model = self.model.get();
        self.floor.position = self.camera.look_at.xy();
        self.renderer.draw(
            framebuffer,
            &self.camera,
            self.floor.matrix(),
            Color::rgb(0.8, 1.0, 0.8),
            Color::rgb(0.8, 1.0, 0.8),
        );
        for block in &model.blocks {
            self.renderer.draw(
                framebuffer,
                &self.camera,
                block.matrix(),
                Color::BLACK,
                match look {
                    Some((block_id, pos, _))
                        if block_id == Some(block.id)
                            && (pos - self.player.position).len() < PLACE_DISTANCE =>
                    {
                        Color::rgb(0.7, 0.7, 0.7)
                    }
                    _ => Color::WHITE,
                },
            );
        }
        for block in &self.falling_blocks {
            let t = block.t * block.t * 10.0;
            match block.edge {
                Some((p1, p2, layer)) => {
                    let p1 = p1.extend(layer as f32);
                    let p2 = p2.extend(layer as f32);
                    self.renderer.draw(
                        framebuffer,
                        &self.camera,
                        Mat4::translate(p1 - vec3(0.0, 0.0, (t - f32::PI / 2.0).max(0.0)))
                            * Mat4::rotate((p2 - p1).normalize(), t.min(f32::PI / 2.0))
                            * Mat4::translate(-p1)
                            * block.block.matrix(),
                        Color::BLACK,
                        Color::rgb(1.0, 0.5, 0.5),
                    );
                }
                None => {
                    self.renderer.draw(
                        framebuffer,
                        &self.camera,
                        Mat4::translate(vec3(0.0, 0.0, -t)) * block.block.matrix(),
                        Color::BLACK,
                        Color::rgb(1.0, 0.5, 0.5),
                    );
                }
            }
        }
        if let Some((block, can_place)) = self.try_place() {
            self.renderer.draw(
                framebuffer,
                &self.camera,
                block.matrix(),
                if can_place {
                    Color::rgb(0.0, 0.7, 0.0)
                } else {
                    Color::RED
                },
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
            if player.id == self.player.id {
                continue;
            }
            let interpolated = match self.interpolated_players.get(&player.id) {
                Some(p) => p,
                None => continue,
            };
            let mat =
                Mat4::translate(interpolated.position) * Mat4::rotate_z(interpolated.rotation);
            let head_mat = mat
                * Mat4::translate(vec3(0.0, 0.0, Player::HEIGHT - Player::RADIUS))
                * Mat4::rotate_x((interpolated.attack_angle / 2.0).max(-f32::PI / 8.0));
            let skin_color = Color::rgb(1.0, 0.8, 0.8);
            self.renderer.draw(
                framebuffer,
                &self.camera,
                mat * Mat4::translate(vec3(0.0, 0.0, Player::HEIGHT / 3.0))
                    * Mat4::scale(vec3(
                        Player::RADIUS / 2.0,
                        Player::RADIUS / 2.0,
                        (Player::HEIGHT - Player::HEIGHT / 3.0 - Player::RADIUS) / 2.0,
                    ))
                    * Mat4::translate(vec3(0.0, 0.0, 1.0)),
                Color::BLACK,
                skin_color,
            );
            self.renderer.draw(
                framebuffer,
                &self.camera,
                head_mat * Mat4::scale_uniform(Player::RADIUS),
                Color::BLACK,
                skin_color,
            );
            let swing = interpolated.swing();
            for side in [-1.0, 1.0] {
                self.renderer.draw(
                    framebuffer,
                    &self.camera,
                    head_mat
                        * Mat4::translate(vec3(
                            Player::RADIUS / 2.0 * side,
                            Player::RADIUS,
                            Player::RADIUS * 0.5,
                        ))
                        * Mat4::scale_uniform(0.1),
                    Color::BLACK,
                    Color::BLACK,
                );
                self.renderer.draw(
                    framebuffer,
                    &self.camera,
                    mat * Mat4::translate(vec3(
                        Player::RADIUS * side,
                        0.0,
                        Player::HEIGHT - Player::RADIUS * 2.0,
                    )) * Mat4::rotate_x(swing * side)
                        * Mat4::scale(vec3(
                            Player::RADIUS / 4.0,
                            Player::RADIUS / 4.0,
                            Player::HEIGHT / 4.0,
                        ))
                        * Mat4::translate(vec3(0.0, 0.0, -1.0)),
                    Color::BLACK,
                    skin_color,
                );
                self.renderer.draw(
                    framebuffer,
                    &self.camera,
                    mat * Mat4::translate(vec3(
                        Player::RADIUS / 2.0 * side,
                        0.0,
                        Player::HEIGHT / 3.0,
                    )) * Mat4::rotate_x(-swing * side)
                        * Mat4::scale(vec3(
                            Player::RADIUS / 4.0,
                            Player::RADIUS / 4.0,
                            Player::HEIGHT / 6.0,
                        ))
                        * Mat4::translate(vec3(0.0, 0.0, -1.0)),
                    Color::BLACK,
                    skin_color,
                );
            }
            self.renderer.draw(
                framebuffer,
                &self.camera,
                head_mat
                    * Mat4::translate(vec3(0.0, Player::RADIUS, -Player::RADIUS * 0.5))
                    * Mat4::scale(vec3(Player::RADIUS * 0.6, 0.1, 0.1)),
                Color::BLACK,
                Color::BLACK,
            );
            if self.show_names {
                if let Some(pos) = self.camera.world_to_screen(
                    self.framebuffer_size.map(|x| x as f32),
                    interpolated.position + vec3(0.0, 0.0, Player::HEIGHT * 1.1),
                ) {
                    self.geng.default_font().draw(
                        framebuffer,
                        &geng::PixelPerfectCamera,
                        &player.name,
                        pos,
                        geng::TextAlign::CENTER,
                        32.0,
                        Color::BLACK,
                    );
                }
            }
        }

        // for a in &model.blocks {
        //     for b in &model.blocks {
        //         if a.layer == b.layer - 1 {
        //             for p in a.intersect_2d(b) {
        //                 if let Some(p) = self.camera.world_to_screen(
        //                     self.framebuffer_size.map(|x| x as f32),
        //                     p.extend(b.layer as f32),
        //                 ) {
        //                     self.geng.draw_2d().circle(
        //                         framebuffer,
        //                         &geng::PixelPerfectCamera,
        //                         p,
        //                         10.0,
        //                         Color::BLUE,
        //                     );
        //                 }
        //             }
        //         }
        //     }
        // }

        if self.editing {
            self.geng.draw_2d().quad(
                framebuffer,
                &geng::PixelPerfectCamera,
                AABB::point(Vec2::ZERO).extend_uniform(5000.0),
                Color::rgba(1.0, 1.0, 1.0, 0.5),
            );
            self.geng.default_font().draw(
                framebuffer,
                &geng::Camera2d {
                    center: Vec2::ZERO,
                    rotation: 0.0,
                    fov: 15.0,
                },
                &format!(
                    "Press Enter to finish\n{}",
                    match self.player.name.as_str() {
                        "" => "Start typing your name",
                        s => s,
                    }
                ),
                vec2(0.0, 0.0),
                geng::TextAlign::CENTER,
                1.0,
                Color::BLACK,
            );
        } else {
            self.geng.default_font().draw(
                framebuffer,
                &geng::Camera2d {
                    center: Vec2::ZERO,
                    rotation: 0.0,
                    fov: 20.0,
                },
                &format!(
                    "You are {:?}\nPress Enter to edit your name",
                    self.player.name
                ),
                vec2(0.0, -9.0),
                geng::TextAlign::CENTER,
                1.0,
                Color::GRAY,
            );
        }
    }
    fn handle_event(&mut self, event: geng::Event) {
        match event {
            geng::Event::MouseMove { delta, .. }
                if !self.editing && self.geng.window().cursor_locked() =>
            {
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
                if let Some((block, true)) = self.try_place() {
                    self.model.send(Message::PlaceBlock(block));
                }
            }
            geng::Event::MouseDown {
                button: geng::MouseButton::Right,
                ..
            } => {
                if let Some((Some(block_id), pos, ..)) = self.look() {
                    if (pos - self.player.position).len() < PLACE_DISTANCE {
                        self.model.send(Message::DeleteBlock(block_id));
                        let model = self.model.get();
                        let block = model.blocks.get(&block_id).unwrap();
                        self.play_sound(
                            &self.assets.del,
                            block.position.extend(block.layer as f32),
                        );
                    }
                }
            }
            geng::Event::KeyDown { key } => {
                match key {
                    geng::Key::Space => {
                        if self.player.can_jump {
                            self.player.jump_speed = Player::JUMP_INITIAL_SPEED;
                        }
                    }
                    geng::Key::PageDown => {
                        // self.block_size = (self.block_size - 0.5).max(0.5);
                    }
                    geng::Key::PageUp => {
                        // self.block_size = (self.block_size + 0.5).min(2.0);
                    }
                    geng::Key::Enter => {
                        self.editing = !self.editing;
                        if self.editing {
                            self.player.name = "".to_owned();
                        } else {
                            self.model
                                .send(Message::ChangeName(self.player.name.clone()));
                        }
                    }
                    geng::Key::Backspace if self.editing => {
                        self.player.name.pop();
                    }
                    geng::Key::T => {
                        self.show_names = !self.show_names;
                    }
                    _ => {}
                }
                let c = format!("{:?}", key);
                if c.len() == 1 && self.editing && self.player.name.len() < 20 {
                    self.player.name.push_str(&c);
                }
            }
            geng::Event::Wheel { delta } => {
                self.place_rotation += delta as f32 * 0.005;
            }
            _ => {}
        }
    }
}

#[derive(geng::Assets)]
struct Assets {
    jump: geng::Sound,
    del: geng::Sound,
    fall: geng::Sound,
    place: geng::Sound,
}

fn main() {
    logger::init().unwrap();

    // Setup working directory
    if let Some(dir) = std::env::var_os("CARGO_MANIFEST_DIR") {
        std::env::set_current_dir(std::path::Path::new(&dir).join("static")).unwrap();
    } else {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(path) = std::env::current_exe().unwrap().parent() {
                std::env::set_current_dir(path).unwrap();
            }
        }
    }

    geng::net::simple::run("Multiplayer", Model::new, |geng, player_id, model| {
        let geng_clone = geng.clone();
        geng::LoadingScreen::new(
            geng,
            geng::EmptyLoadingScreen,
            geng::LoadAsset::load(geng, "."),
            move |assets| Game::new(&geng_clone, player_id, model, assets.unwrap()),
        )
    });
}
