from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import Actor, Critic
from cpprb import ReplayBuffer
from sklearn.mixture import GaussianMixture
import os


class MERTD3Agent:
    def __init__(self, obs_dim, action_dim, action_bounds, env_name, expl_noise=0.1, start_timesteps=25000, buffer_size=200000, actor_lr=1e-3, critic_lr=1e-3, batch_size=256, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.max_action = max(action_bounds["high"])

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.env_name = env_name
        self.expl_noise = expl_noise
        self.start_timesteps = start_timesteps
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise * self.max_action
        self.noise_clip = noise_clip * self.max_action
        self.policy_freq = policy_freq

        self.actor = Actor(obs_dim, action_dim, self.max_action)
        self.actor_target = deepcopy(self.actor)

        self.critic = Critic(obs_dim, action_dim)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.intermediate_rb = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}
        })

        self.rb0 = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}
        })

        self.rb1 = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}
        })
        self.t = 0
        self.model = GaussianMixture(n_components=2)

    def act(self, obs, train_mode=True):
        with torch.no_grad():
            if not train_mode:
                action = self.actor(torch.Tensor(obs)).numpy()
            else:
                if self.t < self.start_timesteps:
                    action = np.random.uniform(low=self.action_bounds['low'], high=self.action_bounds['high'],
                                               size=self.action_dim)
                else:
                    action = (
                        self.actor(torch.Tensor(obs)).numpy()
                        + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                    )

        action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])
        return action

    def step(self, obs, action, reward, next_obs, done):
        self.t += 1
        self.intermediate_rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

        if self.t >= self.start_timesteps:
            if self.t % 1000 == 0:
                self._cluster()

            self._learn()

    def save(self):
        os.makedirs(f"saved_networks/mer/{self.env_name}", exist_ok=True)
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "t": self.t
                    }, f"saved_networks/mer/{self.env_name}/actor_critic.pt")

    def load(self):
        checkpoint = torch.load(f"saved_networks/mer/{self.env_name}/actor_critic.pt")

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target = deepcopy(self.actor)

        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target = deepcopy(self.critic)

        self.t = checkpoint["t"]

    def _cluster(self):
        if self.rb0.get_stored_size() > 0:
            all_rb0_transitions = self.rb0.get_all_transitions()
            self.intermediate_rb.add(obs=all_rb0_transitions['obs'],
                                     action=all_rb0_transitions['action'],
                                     reward=all_rb0_transitions['reward'],
                                     next_obs=all_rb0_transitions['next_obs'],
                                     done=all_rb0_transitions['done'])

        if self.rb1.get_stored_size() > 0:
            all_rb1_transitions = self.rb1.get_all_transitions()
            self.intermediate_rb.add(obs=all_rb1_transitions['obs'],
                                     action=all_rb1_transitions['action'],
                                     reward=all_rb1_transitions['reward'],
                                     next_obs=all_rb1_transitions['next_obs'],
                                     done=all_rb1_transitions['done'])

        self.rb0.clear()
        self.rb1.clear()

        all_transitions = self.intermediate_rb.get_all_transitions()

        preds = self.model.fit_predict(np.concatenate([all_transitions['obs'], all_transitions['action']], axis=1))

        zero_idxs = np.where(preds == 0)
        one_idxs = np.where(preds == 1)

        self.rb0.add(obs=all_transitions['obs'][zero_idxs],
                     action=all_transitions['action'][zero_idxs],
                     reward=all_transitions['reward'][zero_idxs],
                     next_obs=all_transitions['next_obs'][zero_idxs],
                     done=all_transitions['done'][zero_idxs])

        self.rb1.add(obs=all_transitions['obs'][one_idxs],
                     action=all_transitions['action'][one_idxs],
                     reward=all_transitions['reward'][one_idxs],
                     next_obs=all_transitions['next_obs'][one_idxs],
                     done=all_transitions['done'][one_idxs])

        self.intermediate_rb.clear()

    def _sample(self, batch_size):
        intermediate_rb_stored = self.intermediate_rb.get_stored_size()
        rb0_stored = self.rb0.get_stored_size()
        rb1_stored = self.rb1.get_stored_size()

        total = intermediate_rb_stored + rb0_stored + rb1_stored

        samples_intermediate = self.intermediate_rb.sample(batch_size * intermediate_rb_stored // total)
        samples_rb0 = self.rb0.sample(batch_size * rb0_stored // total)
        samples_rb1 = self.rb1.sample(batch_size * rb1_stored // total)

        samples_all = {}

        for key in samples_intermediate:
            samples_all[key] = np.concatenate([samples_intermediate[key], samples_rb0[key], samples_rb1[key]])

        return samples_all

    def _learn(self):
        sample = self._sample(self.batch_size)
        obs = torch.Tensor(sample['obs'])
        action = torch.Tensor(sample['action'])
        reward = torch.Tensor(sample['reward'])
        next_obs = torch.Tensor(sample['next_obs'])
        done = torch.Tensor(sample['done'])

        Q_current1, Q_current2 = self.critic(obs, action)
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (
                self.actor_target(next_obs) + noise
            ).clamp(torch.from_numpy(self.action_bounds['low']), torch.from_numpy(self.action_bounds['high']))

            Q1_target_next, Q2_target_next = self.critic_target(next_obs, next_actions)
            Q_target_next = torch.min(Q1_target_next, Q2_target_next)
            Q_target = reward + self.gamma * Q_target_next * (1 - done)

        critic_loss = F.mse_loss(Q_current1, Q_target) + F.mse_loss(Q_current2, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.t % self.policy_freq == 0:
            actor_loss = -self.critic(obs, self.actor(obs))[0].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
