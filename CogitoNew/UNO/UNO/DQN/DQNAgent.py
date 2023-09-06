import numpy.random
import torch
import torch.nn as nn
import gymnasium as gym

from DQN.Agent import Agent
from Config import Config as confg
from DQN.QNetwork import QNetwork
from DQN.Buffer import BasicBuffer
import numpy as np
from rlcard.envs import uno

class DQNAgent(Agent):
    
    class Config(confg):
        wandb_name = "DQN-UNO"
        env = "uno"
        ac_dim = 1
        ob_dim = 4*4*15
        action_space = 61
        hidden_dim = 25
     
        lr = 0.01
        epsilon = 0.15
        gamma = 0.99
        batch_size = 64

        min_buffer_size = 100
        buffer_capacity = 10000
        episodes = 500
        eval_freq = 25
        update_target_network_freq = 25

        # Reduce overestimation https://arxiv.org/pdf/1509.06461.pdf
        double_dqn = True

    def __init__(self, cfg):
        super().__init__(cfg)

        self.lr = cfg.lr
        self.epsilon = cfg.epsilon
        self.gamma = cfg.gamma

        # Initialize networks
        self.q_values = QNetwork(cfg.ob_dim, cfg.action_space, cfg.hidden_dim)
        self.target_network = QNetwork(cfg.ob_dim, cfg.action_space, cfg.hidden_dim)
        self.update_target_network()

        # Init replay buffer
        self.buffer = BasicBuffer.make_default(cfg.buffer_capacity, cfg.ob_dim, cfg.ac_dim, wrap=True)

        # Init optimizer stuff
        self.optimizer = torch.optim.Adam(self.q_values.parameters(), cfg.lr)
        self.loss = nn.MSELoss()

    def _exploration_action(self):
        return numpy.random.randint(self.cfg.action_space)

    def _greedy_action(self, state):
        # Make the state a tensor so the network will accept it
        state = torch.tensor(state)
        return torch.argmax(self.q_values(state)).item()

    def act(self, state):
        if np.random.rand() < self.cfg.epsilon:
            return self._exploration_action()
        else:
            return self._greedy_action(state)

    def save(self, path):
        torch.save(self.q_values.state_dict(), path)

    def load(self, path):
        self.q_values = torch.load(path)
        self.update_target_network()

    def store_transition(self, ob, ac, rew, next_ob, done):
        self.buffer << {'ob': [ob],
                        'ac': [ac],
                        'rew': [rew],
                        'next_ob': [next_ob],
                        'done': [done]
                        }

    def update_target_network(self):
        # Update the target network
        self.target_network.load_state_dict(self.q_values.state_dict())

    def update_q_values(self):
        
        if self.buffer.size < self.cfg.min_buffer_size:
            return None
        
        else:
            # Get from buffer
            ob, ac, rew, next_ob, done = self.buffer.sample(self.cfg.batch_size)
            ac = ac.type(torch.long)
            done = done.type(torch.int)

            if not self.cfg.double_dqn:
                # set target
                target_max, _ = self.target_network(next_ob).max(dim=1)
            else:
                # Double DQN
                target_ac = self.q_values(next_ob).argmax(dim=-1)
                target_max = self.target_network(next_ob).gather(1, target_ac.view(-1, 1)).squeeze()

            # * (1-done) er triks for å sette target til kun reward når vi er ferdige
            td_target = rew + self.cfg.gamma * target_max * (1 - done)

            # Predict q-values and index using action
            old_values = self.q_values(ob).gather(1, ac.view(-1, 1)).squeeze()

            loss = self.loss(td_target, old_values)

            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss


if __name__ == '__main__':
    
    agent = DQNAgent(DQNAgent.Config())

    env = gym.make("CartPole-v1")

    obs, info = env.reset()

    for i in range(1000):
        
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = 1

        agent.store_transition(obs, action, reward, next_obs, terminated)

        obs = next_obs

    agent.update_q_values()
