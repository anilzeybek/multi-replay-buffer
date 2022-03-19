import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from cpprb import ReplayBuffer
from sklearn.mixture import GaussianMixture

from models import Actor, Critic


class TD3Agent:
    def __init__(self, obs_dim, action_dim, action_bounds, env_name, expl_noise=0.1, start_timesteps=25000, buffer_size=200000, actor_lr=1e-3, critic_lr=1e-3, batch_size=256, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, mer=False, number_of_rbs=0, pow=1):
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
        self.mer = mer
        self.pow = pow

        self.actor = Actor(obs_dim, action_dim, self.max_action)
        self.actor_target = deepcopy(self.actor)

        self.critic = Critic(obs_dim, action_dim)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.rb = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}
        })

        if self.mer:
            self.cluster_rbs = [self._create_rb() for _ in range(number_of_rbs)]
            self.clustering_model = GaussianMixture(n_components=number_of_rbs, tol=5e-3)

        self.t = 0

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
        self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

        if self.t >= self.start_timesteps:
            if self.mer and self.t % 25000 == 0:
                self._cluster()

            self._learn()

    def save(self):
        identifier = "mer" if self.mer else "orig"

        os.makedirs(f"saved_networks/{identifier}/{self.env_name}", exist_ok=True)
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "t": self.t
                    }, f"saved_networks/{identifier}/{self.env_name}/actor_critic.pt")

    def load(self):
        identifier = "mer" if self.mer else "orig"
        checkpoint = torch.load(f"saved_networks/{identifier}/{self.env_name}/actor_critic.pt")

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target = deepcopy(self.actor)

        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target = deepcopy(self.critic)

        self.t = checkpoint["t"]

    def _create_rb(self):
        rb = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}
        })
        return rb

    def _cluster(self):
        for cluster_rb in self.cluster_rbs:
            if cluster_rb.get_stored_size() > 0:
                all_cluster_rb_transitions = cluster_rb.get_all_transitions()
                self.rb.add(obs=all_cluster_rb_transitions['obs'],
                            action=all_cluster_rb_transitions['action'],
                            reward=all_cluster_rb_transitions['reward'],
                            next_obs=all_cluster_rb_transitions['next_obs'],
                            done=all_cluster_rb_transitions['done'])

                cluster_rb.clear()

        all_transitions = self.rb.get_all_transitions()

        preds = self.clustering_model.fit_predict(np.concatenate(
            [all_transitions['obs'], all_transitions['action']], axis=1))

        idxs = []
        for i in range(len(self.cluster_rbs)):
            idxs.append(np.where(preds == i))

        for i in range(len(self.cluster_rbs)):
            self.cluster_rbs[i].add(obs=all_transitions['obs'][idxs[i]],
                                    action=all_transitions['action'][idxs[i]],
                                    reward=all_transitions['reward'][idxs[i]],
                                    next_obs=all_transitions['next_obs'][idxs[i]],
                                    done=all_transitions['done'][idxs[i]])

        self.rb.clear()

    def _sample(self, batch_size):
        total = self.rb.get_stored_size() ** self.pow
        for cluster_rb in self.cluster_rbs:
            total += cluster_rb.get_stored_size() ** self.pow

        samples_cluster_rbs = []
        for cluster_rb in self.cluster_rbs:
            samples_cluster_rbs.append(cluster_rb.sample(int(batch_size * cluster_rb.get_stored_size()**self.pow / total) + 1))

        samples_rb = self.rb.sample(int(batch_size * self.rb.get_stored_size()**self.pow / total))

        samples_all = {}
        for key in samples_rb:
            samples_all[key] = np.concatenate([samples_rb[key], *[cluster_rb[key]
                                              for cluster_rb in samples_cluster_rbs]])

        return samples_all

    def _learn(self):
        if self.mer:
            sample = self._sample(self.batch_size)
        else:
            sample = self.rb.sample(self.batch_size)
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
