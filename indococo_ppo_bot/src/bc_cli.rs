//! FR5 behavior cloning tools for `indococo-train` (see EXPERIMENTS.md, FR5):
//!
//! ```text
//! indococo-train curr <level> <episodes> <policy> [threads] [data_out_prefix] [noise]
//!     policy: expert | nearest | idle | greedy:<ckpt> | sample:<ckpt>
//! indococo-train normal-obs <ckpt> <setters.toml> <steps> <threads> <out.f32>
//! ```
//!
//! `curr` plays reset curriculum episodes at a fixed level and prints one
//! JSON line: earned success rate (reset after 0.3 s, within 3 s, as in
//! training) and how often the regained flip was used (a dodge touching the
//! ball within 1.5 s after the reset). With a data prefix it also writes the
//! flier's observations and the expert's labels (`.obs.f32`, `.act.u8`,
//! `.phase.u8`); `noise` is the share of steps executed with a random action
//! (labels stay the expert's), so the data covers recoveries. With an actor
//! policy and a data prefix the actor drives and the expert labels (DAgger).
//!
//! `normal-obs` plays the setter mix with the checkpoint (sampled actions,
//! both cars) and writes every other observation, for distillation.

use super::*;
use indococo_bot::curriculum::{Attempt, ResetCurriculum};
use indococo_bot::env::TickResets;
use indococo_bot::expert::{Expert, Phase};
use indococo_bot::setters::SetterKind;
use std::io::Write;

const ENVS: usize = 16;
const USE_STEPS: u32 = 22;

#[derive(Clone)]
enum Pol {
    Expert,
    Nearest,
    Idle,
    Actor(PathBuf, bool),
}

fn parse_pol(s: &str) -> Pol {
    match s {
        "expert" => Pol::Expert,
        "nearest" => Pol::Nearest,
        "idle" => Pol::Idle,
        _ => {
            let (k, p) = s.split_once(':').expect("policy");
            Pol::Actor(PathBuf::from(p), k == "greedy")
        }
    }
}

fn logits(actor: &Net, obs: &[&Vec<f32>], dev: &Device) -> Vec<Vec<f32>> {
    let flat: Vec<f32> = obs.iter().flat_map(|o| o.iter().copied()).collect();
    let x = Tensor::<2>::from_data(TensorData::new(flat, [obs.len(), obs_size()]), dev);
    actor
        .forward(x)
        .into_data()
        .try_to_vec::<f32>()
        .expect("logits")
        .chunks(N_ACTIONS)
        .map(<[f32]>::to_vec)
        .collect()
}

fn pick(row: &[f32], greedy: bool, rng: &mut SmallRng) -> usize {
    if greedy {
        return (0..N_ACTIONS)
            .max_by(|&i, &j| row[i].total_cmp(&row[j]))
            .expect("a");
    }
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f32> = row.iter().map(|l| (l - m).exp()).collect();
    let mut u = rng.random_range(0.0..p.iter().sum::<f32>());
    for (i, x) in p.iter().enumerate() {
        u -= x;
        if u <= 0.0 {
            return i;
        }
    }
    N_ACTIONS - 1
}

struct Slot {
    env: BotEnv,
    state: GameState,
    obs: Vec<Vec<f32>>,
    flier: usize,
    attempt: Attempt,
    phase: Phase,
    use_steps: u32,
    used: bool,
}

#[derive(Default, Clone, Copy, Serialize)]
struct CurrStats {
    episodes: u64,
    successes: u64,
    used: u64,
    steps: u64,
    early_resets: u64,
}

impl Slot {
    fn new(level: usize, c: &ResetCurriculum) -> Self {
        let mix = vec![WeightedSetter {
            weight: 1.0,
            kind: SetterKind::ResetCurriculum(ResetCurriculum {
                fixed_level: Some(level),
                ..c.clone()
            }),
        }];
        let mut env = create_env(&RewardConfig::default(), &mix);
        let (state, obs, _) = env.reset();
        let mut s = Self {
            env,
            state,
            obs,
            flier: 0,
            attempt: Attempt::new(level, 0, c.min_control_s, c.deadline_s),
            phase: Phase::Reset,
            use_steps: 0,
            used: false,
        };
        s.restart(level, c);
        s
    }

    fn restart(&mut self, level: usize, c: &ResetCurriculum) {
        let (state, obs, _) = self.env.reset();
        self.flier = (0..state.cars.len())
            .max_by(|&i, &j| {
                state.cars[i]
                    .1
                    .phys
                    .pos
                    .z
                    .total_cmp(&state.cars[j].1.phys.pos.z)
            })
            .expect("cars");
        self.attempt = Attempt::new(level, self.flier, c.min_control_s, c.deadline_s);
        self.phase = Phase::Reset;
        self.use_steps = 0;
        self.used = false;
        (self.state, self.obs) = (state, obs);
    }

    /// One step; returns true when the episode is over.
    fn step(&mut self, action: usize, idle: usize, st: &mut CurrStats) -> bool {
        let mut actions = vec![idle; self.state.cars.len()];
        actions[self.flier] = action;
        let ball_v0 = self.state.ball.phys.vel;
        let res = self.env.step(&self.state, &actions);
        st.steps += 1;
        let reset = self
            .env
            .get_shared_info()
            .tick_resets()
            .get(self.flier)
            .copied()
            .unwrap_or(false);
        let done_env = res.is_terminal || res.truncated;
        let car = res.state.cars[self.flier].1;
        let touched = (res.state.ball.phys.vel - ball_v0).length() > 150.0;
        (self.state, self.obs) = (res.state, res.obs);
        match self.phase {
            Phase::Reset => match {
                if reset && self.attempt.steps + 1 < self.attempt.first_step {
                    st.early_resets += 1;
                }
                self.attempt.step(reset)
            } {
                Some(true) => {
                    st.successes += 1;
                    self.phase = Phase::Use;
                }
                Some(false) => return true,
                None => {}
            },
            Phase::Use => {
                self.use_steps += 1;
                if std::env::var("FR5_DEBUG").is_ok() {
                    let b = self.state.ball.phys;
                    eprintln!(
                        "use {} a={action} dist={:.0} carz={:.0} ballz={:.0} jumped={} flipped={} flipping={} touched={touched} ground={}",
                        self.use_steps,
                        (b.pos - car.phys.pos).length(),
                        car.phys.pos.z,
                        b.pos.z,
                        car.has_jumped,
                        car.has_flipped,
                        car.is_flipping,
                        car.is_on_ground()
                    );
                }
                if touched && (car.is_flipping || car.has_flipped) {
                    self.used = true;
                    st.used += 1;
                    return true;
                }
                if self.use_steps >= USE_STEPS {
                    return true;
                }
            }
        }
        done_env
    }
}

#[derive(Default)]
struct Data {
    obs: Vec<f32>,
    act: Vec<u8>,
    phase: Vec<u8>,
}

fn curr_worker(
    level: usize,
    episodes: u64,
    pol: &Pol,
    seed: u64,
    record: bool,
    noise: f64,
) -> (CurrStats, Data) {
    let c = ResetCurriculum::default();
    let dev = Device::flex();
    let actor = match pol {
        Pol::Actor(p, _) => Some(load_actor(p, &dev)),
        _ => None,
    };
    let mut expert = Expert::default();
    let idle = expert.idle;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut slots: Vec<Slot> = (0..ENVS).map(|_| Slot::new(level, &c)).collect();
    let mut st = CurrStats::default();
    let mut data = Data::default();
    let mut started = ENVS as u64;
    let mut live: Vec<bool> = vec![true; ENVS];
    while live.iter().any(|&l| l) {
        let acts: Vec<usize> = match (&actor, pol) {
            (Some(a), Pol::Actor(_, greedy)) => {
                let obs: Vec<&Vec<f32>> = slots.iter().map(|s| &s.obs[s.flier]).collect();
                logits(a, &obs, &dev)
                    .iter()
                    .map(|r| pick(r, *greedy, &mut rng))
                    .collect()
            }
            _ => slots
                .iter()
                .map(|s| match pol {
                    Pol::Idle => idle,
                    Pol::Nearest => {
                        if s.phase == Phase::Reset {
                            expert.nearest_align(&s.state, s.flier)
                        } else {
                            idle
                        }
                    }
                    _ => expert.act(&s.state, s.flier, s.phase),
                })
                .collect(),
        };
        for (i, s) in slots.iter_mut().enumerate() {
            if !live[i] {
                continue;
            }
            let mut a = acts[i];
            if record {
                // Labels are always the expert's; with an actor policy the
                // actor drives (DAgger), otherwise the expert plus noise.
                let label = if actor.is_some() {
                    expert.act(&s.state, s.flier, s.phase)
                } else {
                    a
                };
                data.obs.extend_from_slice(&s.obs[s.flier]);
                data.act.push(label as u8);
                data.phase.push(u8::from(s.phase == Phase::Use));
                if rng.random_bool(noise) {
                    a = rng.random_range(0..N_ACTIONS);
                }
            }
            if s.step(a, idle, &mut st) {
                st.episodes += 1;
                if started < episodes {
                    started += 1;
                    s.restart(level, &c);
                } else {
                    live[i] = false;
                }
            }
        }
    }
    (st, data)
}

pub fn curr(args: &[String]) {
    init_from_default(true).expect("load collision meshes");
    let level: usize = args[0].parse().expect("level");
    let episodes: u64 = args[1].parse().expect("episodes");
    let pol = parse_pol(&args[2]);
    let threads: u64 = args.get(3).map_or(16, |s| s.parse().expect("threads"));
    let prefix = args.get(4).filter(|s| *s != "-").cloned();
    let noise: f64 = args.get(5).map_or(0.0, |s| s.parse().expect("noise"));
    let per = episodes.div_ceil(threads).max(ENVS as u64);
    let outs: Vec<(CurrStats, Data)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let pol = pol.clone();
                let rec = prefix.is_some();
                s.spawn(move || curr_worker(level, per, &pol, 1000 * level as u64 + t, rec, noise))
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut tot = CurrStats::default();
    for (st, _) in &outs {
        tot.episodes += st.episodes;
        tot.successes += st.successes;
        tot.used += st.used;
        tot.steps += st.steps;
        tot.early_resets += st.early_resets;
    }
    if let Some(p) = prefix {
        let mut fo = std::fs::File::create(format!("{p}.obs.f32")).expect("obs");
        let mut fa = std::fs::File::create(format!("{p}.act.u8")).expect("act");
        let mut fp = std::fs::File::create(format!("{p}.phase.u8")).expect("phase");
        for (_, d) in &outs {
            let bytes: Vec<u8> = d.obs.iter().flat_map(|x| x.to_le_bytes()).collect();
            fo.write_all(&bytes).expect("w");
            fa.write_all(&d.act).expect("w");
            fp.write_all(&d.phase).expect("w");
        }
    }
    let e = tot.episodes.max(1) as f64;
    println!(
        "{}",
        serde_json::json!({
            "level": level, "policy": args[2], "episodes": tot.episodes,
            "success": tot.successes as f64 / e, "used_per_episode": tot.used as f64 / e,
            "used_per_success": tot.used as f64 / tot.successes.max(1) as f64,
            "steps": tot.steps, "noise": noise, "early_resets_per_episode": tot.early_resets as f64 / e,
        })
    );
}

pub fn normal_obs(args: &[String]) {
    #[derive(Deserialize)]
    struct Mix {
        state_setters: Vec<WeightedSetter>,
    }
    init_from_default(true).expect("load collision meshes");
    let ckpt = PathBuf::from(&args[0]);
    let mix: Mix =
        toml::from_str(&std::fs::read_to_string(&args[1]).expect("read")).expect("parse");
    let steps: usize = args[2].parse().expect("steps");
    let threads: usize = args[3].parse().expect("threads");
    let per = steps.div_ceil(threads);
    let outs: Vec<Vec<f32>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let (ckpt, mix) = (&ckpt, &mix.state_setters);
                s.spawn(move || {
                    let dev = Device::flex();
                    let actor = load_actor(ckpt, &dev);
                    let mut rng = SmallRng::seed_from_u64(77 + t as u64);
                    let mut envs: Vec<(BotEnv, GameState, Vec<Vec<f32>>)> = (0..ENVS)
                        .map(|_| {
                            let mut e = create_env(&RewardConfig::default(), mix);
                            let (s, o, _) = e.reset();
                            (e, s, o)
                        })
                        .collect();
                    let mut out = Vec::new();
                    for k in 0..per.div_ceil(ENVS) {
                        let obs: Vec<&Vec<f32>> = envs.iter().flat_map(|e| e.2.iter()).collect();
                        if k % 2 == 0 {
                            for o in &obs {
                                out.extend_from_slice(o);
                            }
                        }
                        let rows = logits(&actor, &obs, &dev);
                        let mut it = rows.iter();
                        for e in &mut envs {
                            let acts: Vec<usize> = (0..e.2.len())
                                .map(|_| pick(it.next().expect("row"), false, &mut rng))
                                .collect();
                            let r = e.0.step(&e.1, &acts);
                            if r.is_terminal || r.truncated {
                                let (s, o, _) = e.0.reset();
                                (e.1, e.2) = (s, o);
                            } else {
                                (e.1, e.2) = (r.state, r.obs);
                            }
                        }
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    let mut f = std::fs::File::create(&args[4]).expect("out");
    let mut n = 0;
    for o in &outs {
        n += o.len() / obs_size();
        let bytes: Vec<u8> = o.iter().flat_map(|x| x.to_le_bytes()).collect();
        f.write_all(&bytes).expect("w");
    }
    println!("wrote {n} observations to {}", args[4]);
}
