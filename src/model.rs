use std::f64::consts::PI;
use tch::{kind::FLOAT_CPU, nn, nn::OptimizerConfig, Device, Kind::Float, Tensor};

use crate::utils::ReplayBuffer;

pub struct Config {
    pub hidden_one_size: i64,
    pub hidden_two_size: i64,
    pub discount: f64,
    pub tau: f64,
    pub policy_noise: f64,
    pub noise_clip: f64,
    pub actor_learning_rate: f64,
    pub critic_learning_rate: f64,
    pub beta: f64,
    pub num_noise_samples: i64,
    pub with_importance_sampling: bool,
    pub replay_buffer_capacity: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            hidden_one_size: 400,
            hidden_two_size: 300,
            discount: 0.99,
            tau: 0.005,
            policy_noise: 0.2,
            noise_clip: 0.5,
            actor_learning_rate: 1e-3,
            critic_learning_rate: 1e-3,
            beta: 0.001,
            num_noise_samples: 50,
            with_importance_sampling: false,
            replay_buffer_capacity: 1000000,
        }
    }
}

struct Actor {
    var_store: nn::VarStore,
    network: nn::Sequential,
    device: Device,
    state_dim: usize,
    action_dim: usize,
    max_action: f64,
    optimizer: nn::Optimizer<nn::Adam>,
    learning_rate: f64,
    hidden_one_size: i64,
    hidden_two_size: i64,
    path: String,
}

impl Clone for Actor {
    fn clone(&self) -> Self {
        let mut new = Self::new(
            self.state_dim,
            self.action_dim,
            self.max_action,
            self.path.clone(),
            self.learning_rate,
            self.hidden_one_size,
            self.hidden_two_size,
        );
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl Actor {
    fn new(
        state_dim: usize,
        action_dim: usize,
        max_action: f64,
        path: String,
        learning_rate: f64,
        hidden_one_size: i64,
        hidden_two_size: i64,
    ) -> Self {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let optimizer = nn::Adam::default()
            .build(&var_store, learning_rate)
            .unwrap();

        let p = &var_store.root();
        Self {
            network: nn::seq()
                .add(nn::linear(
                    p / format!("{}{}", path, "l1"),
                    state_dim as _,
                    hidden_one_size,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    p / format!("{}{}", path, "l2"),
                    hidden_one_size,
                    hidden_two_size,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    p / format!("{}{}", path, "l3"),
                    hidden_two_size,
                    action_dim as _,
                    Default::default(),
                ))
                .add_fn(move |xs| max_action * xs.tanh()),
            device: p.device(),
            state_dim,
            action_dim,
            var_store,
            max_action,
            optimizer,
            learning_rate,
            hidden_one_size,
            hidden_two_size,
            path,
        }
    }

    fn forward(&self, state: &Tensor) -> Tensor {
        state.to_device(self.device).apply(&self.network)
    }
}

struct Critic {
    var_store: nn::VarStore,
    network: nn::Sequential,
    device: Device,
    state_dim: usize,
    action_dim: usize,
    optimizer: nn::Optimizer<nn::Adam>,
    learning_rate: f64,
    hidden_one_size: i64,
    hidden_two_size: i64,
    path: String,
}

impl Clone for Critic {
    fn clone(&self) -> Self {
        let mut new = Self::new(
            self.state_dim,
            self.action_dim,
            self.path.clone(),
            self.learning_rate,
            self.hidden_one_size,
            self.hidden_two_size,
        );
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl Critic {
    fn new(
        state_dim: usize,
        action_dim: usize,
        path: String,
        learning_rate: f64,
        hidden_one_size: i64,
        hidden_two_size: i64,
    ) -> Self {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let optimizer = nn::Adam::default()
            .build(&var_store, learning_rate)
            .unwrap();
        let p = &var_store.root();
        Self {
            network: nn::seq()
                .add(nn::linear(
                    p / format!("{}{}", path, "l3"),
                    (state_dim + action_dim) as _,
                    hidden_one_size,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    p / format!("{}{}", path, "l3"),
                    hidden_one_size,
                    hidden_two_size,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    p / format!("{}{}", path, "l3"),
                    hidden_two_size,
                    1,
                    Default::default(),
                )),
            device: p.device(),
            var_store,
            state_dim,
            action_dim,
            optimizer,
            learning_rate,
            hidden_one_size,
            hidden_two_size,
            path,
        }
    }

    fn forward(&self, state: &Tensor, action: &Tensor) -> Tensor {
        let mut cat = 1;
        if state.size().len() == 3 {
            cat = 2
        };
        let xs = Tensor::cat(&[state.copy(), action.copy()], cat);
        xs.to_device(self.device).apply(&self.network)
    }
}

pub struct SD3 {
    actor1: Actor,
    actor1_target: Actor,
    actor2: Actor,
    actor2_target: Actor,
    critic1: Critic,
    critic1_target: Critic,
    critic2: Critic,
    critic2_target: Critic,
    replay_buffer: ReplayBuffer,
    state_dim: usize,
    action_dim: usize,
    max_action: f64,
    config: Config,
}

impl SD3 {
    fn new(state_dim: usize, action_dim: usize, max_action: f64, config: Config) -> SD3 {
        let actor1 = Actor::new(
            state_dim,
            action_dim,
            max_action,
            "a1".to_owned(),
            config.actor_learning_rate,
            config.hidden_one_size,
            config.hidden_two_size,
        );
        let actor1_target = actor1.clone();
        let actor2 = Actor::new(
            state_dim,
            action_dim,
            max_action,
            "a2".to_owned(),
            config.actor_learning_rate,
            config.hidden_one_size,
            config.hidden_two_size,
        );
        let actor2_target = actor2.clone();
        let critic1 = Critic::new(
            state_dim,
            action_dim,
            "c1".to_owned(),
            config.critic_learning_rate,
            config.hidden_one_size,
            config.hidden_two_size,
        );
        let critic1_target = critic1.clone();
        let critic2 = Critic::new(
            state_dim,
            action_dim,
            "c2".to_owned(),
            config.critic_learning_rate,
            config.hidden_one_size,
            config.hidden_two_size,
        );
        let critic2_target = critic2.clone();
        let replay_buffer = ReplayBuffer::new(config.replay_buffer_capacity, state_dim, action_dim);
        SD3 {
            actor1,
            actor1_target,
            actor2,
            actor2_target,
            critic1,
            critic1_target,
            critic2,
            critic2_target,
            replay_buffer,
            state_dim,
            action_dim,
            max_action,
            config,
        }
    }

    fn train(&mut self, batch_size: usize, q1: bool) {
        let (states, actions, reward, mut next_state, not_done) =
            match self.replay_buffer.sample(batch_size) {
                Some(v) => v,
                _ => return, // We don't have enough samples for training yet.
            };

        let no_grad = tch::no_grad_guard();
        let mut next_action: Tensor;
        if q1 {
            next_action = self.actor1_target.forward(&next_state);
        } else {
            next_action = self.actor2_target.forward(&next_state);
        }

        let mut noise = Tensor::randn(
            &[
                actions.size()[0],
                self.config.num_noise_samples,
                actions.size()[1],
            ],
            FLOAT_CPU,
        ) * self.config.policy_noise;

        let noise_pdf: Tensor;
        if self.config.with_importance_sampling {
            let pdf_divisor = &noise
                .pow(2.)
                .multiply1(-1.)
                .divide1(2. * self.config.policy_noise.powf(2.))
                .exp()
                .multiply1(self.config.policy_noise * (2. * PI).sqrt());
            let pdfs: Tensor = 1. / pdf_divisor;
            noise_pdf = pdfs.prod1(2, false, Float);
        } else {
            noise_pdf = Tensor::new();
        }

        noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip);

        next_action = next_action.unsqueeze(1);
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action);

        next_state = next_state.unsqueeze(1);
        next_state = next_state.repeat(&[1, self.config.num_noise_samples, 1]);

        let next_q1 = self.critic1_target.forward(&next_state, &next_action);
        let next_q2 = self.critic2_target.forward(&next_state, &next_action);
        let next_q = next_q1.min1(&next_q2).squeeze1(2);

        // Softmax
        let (max_q_vals, _) = next_q.max2(1, true);
        let norm_q_vals = &next_q - max_q_vals;

        let mut denominator = (self.config.beta * norm_q_vals).exp();
        let mut numerator = &next_q * &denominator;

        if self.config.with_importance_sampling {
            numerator = numerator / &noise_pdf;
            denominator = denominator / &noise_pdf;
        }

        let sum_numerator = numerator.sum1(&[1], false, Float);
        let sum_denominator = denominator.sum1(&[1], false, Float);

        let softmax_q_vals = (sum_numerator / sum_denominator).unsqueeze(1);

        let target_q = reward + not_done * self.config.discount * softmax_q_vals;

    }
}
