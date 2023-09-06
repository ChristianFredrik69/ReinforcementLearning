import Config as confg
import Buffer
import Agent
import QNetwork
import numpy as np
import torch


class DQNAgent(Agent):

    class Config(confg):
        env = "CartPole"
        ac_dim = 2
        ob_dim = 4

        lr = 0.01
        epsilon = 0.2
        gamma = 0.99

        buffer_capacity = 5000
        num_episodes = 100
        eval_freq = 5

    def __init__(self, cfg):
        super().__init__(cfg)

        self.lr = cfg.lr
        self.epsilon = cfg.epsilon
        self.gamma = cfg.gamma

        self.q_values = QNetwork(cfg.ob_dim, self.action_space)
        self.buffer = Buffer.make_default(cfg.buffer_capacity, cfg.ob_dim, cfg.ac_dim)

        self.buffer = Buffer(cfg.buffer_capacity,
                                  fields = [
                                    dict(key = 'ob', shape = [cfg.ob_dim]),
                                    dict(key = 'next_ob', shape = [cfg.ob_dim]),
                                    dict(key = 'ac'),
                                    dict(key = 'reward'),
                                    dict(key = 'done')
                                  ]
                                  )

    def act(self, state):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def _exploration_action(self):
        return np.random.randint(self.cfg.action_space)

    def _greedy_action(self, state):
        return torch.argmax(self.qvalues(state))
    
    def act(self, state):
        if np.random.rand() < self.cfg.epsilon:
            return self._exploration_action()
        else:
            return self._greedy_action(state)

    def save(self, path):
        torch.save(self.q_values.state_dict(), path)
    
    def load(self, path):
        self.q_values = torch.load(path)

    def store_transition(self, ob, ac, rew, next_ob, done):
        self.buffer << {'ob': [ob], 'ac' : [ac], 'rew' : [rew], 'next_ob' : [next_ob], 'done' : [done]}

    def update_q_values(self):
        if self.buffer.capacity < 100:
            return
        else: