use tch::{
    kind::{FLOAT_CPU, INT64_CPU},
    Tensor,
};

pub struct ReplayBuffer {
    state: Tensor,
    next_state: Tensor,
    rewards: Tensor,
    actions: Tensor,
    not_done: Tensor,
    capacity: usize,
    size: usize,
    pointer: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, state_dim: usize, action_dim: usize) -> Self {
        Self {
            state: Tensor::zeros(&[capacity as _, state_dim as _], FLOAT_CPU),
            next_state: Tensor::zeros(&[capacity as _, state_dim as _], FLOAT_CPU),
            actions: Tensor::zeros(&[capacity as _, action_dim as _], FLOAT_CPU),
            rewards: Tensor::zeros(&[capacity as _, 1], FLOAT_CPU),
            not_done: Tensor::zeros(&[capacity as _, 1], FLOAT_CPU),
            capacity,
            size: 0,
            pointer: 0,
        }
    }

    pub fn add(&mut self, state: &Tensor, actions: &Tensor, next_state: &Tensor, reward: &Tensor, done: &Tensor) {
        let i = self.pointer % self.capacity;
        self.state.get(i as _).copy_(state);
        self.actions.get(i as _).copy_(actions);
        self.next_state.get(i as _).copy_(next_state);
        self.rewards.get(i as _).copy_(reward);
        self.not_done.get(i as _).copy_(done);
        self.pointer += 1;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    pub fn sample(&self, batch_size: usize) -> Option<(Tensor, Tensor, Tensor, Tensor, Tensor)> {

        let indexes = Tensor::randint((self.size) as _, &[batch_size as _], INT64_CPU);

        let states = self.state.index_select(0, &indexes);
        let actions = self.actions.index_select(0, &indexes);
        let next_states = self.next_state.index_select(0, &indexes);
        let rewards = self.rewards.index_select(0, &indexes);
        let not_done = self.not_done.index_select(0, &indexes);

        Some((states, actions, rewards, next_states, not_done))
    }
}

