import os
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from cpprb import ReplayBuffer
from sklearn.cluster import KMeans

from models import Actor, Critic
from normalizer import Normalizer


class TD3Agent:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 action_bounds,
                 env_name,
                 expl_noise,
                 start_timesteps,
                 buffer_size,
                 actor_lr,
                 critic_lr,
                 batch_size,
                 gamma,
                 tau,
                 policy_noise,
                 noise_clip,
                 policy_freq,
                 number_of_rbs,
                 clustering_freq,
                 alpha,
                 beta):
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
        self.number_of_rbs = number_of_rbs
        self.clustering_freq = clustering_freq
        self.alpha = alpha
        self.beta = beta

        self.actor = Actor(obs_dim, action_dim, self.max_action)
        self.actor_target = deepcopy(self.actor)

        self.critic = Critic(obs_dim, action_dim)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.normalizer = Normalizer(self.obs_dim)

        self.rb = ReplayBuffer(self.buffer_size, env_dict={
            "obs": {"shape": self.obs_dim},
            "action": {"shape": self.action_dim},
            "reward": {},
            "next_obs": {"shape": self.obs_dim},
            "done": {}
        })

        if number_of_rbs > 1:
            self.cluster_rbs = [self._create_rb() for _ in range(number_of_rbs - 1)]
            self.clustering_model = KMeans(n_clusters=number_of_rbs - 1)

        self.t = 0

    def act(self, obs, train_mode=True):
        normalized_obs = self.normalizer.normalize(obs)
        with torch.no_grad():
            if not train_mode:
                action = self.actor(torch.Tensor(normalized_obs)).numpy()
            else:
                if self.t < self.start_timesteps:
                    action = np.random.uniform(low=self.action_bounds['low'], high=self.action_bounds['high'],
                                               size=self.action_dim)
                else:
                    action = (
                            self.actor(torch.Tensor(normalized_obs)).numpy()
                            + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                    )

        action = np.clip(action, self.action_bounds['low'], self.action_bounds['high'])
        return action

    def step(self, obs, action, reward, next_obs, done):
        self.t += 1
        self.rb.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        self.normalizer.update(obs)

        if self.t >= self.start_timesteps:
            if self.number_of_rbs > 1 and self.t % self.clustering_freq == 0:
                self._cluster()

            self._learn()

        if done:
            self.rb.on_episode_end()

    def save(self, seed):
        os.makedirs(f"checkpoints", exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "normalizer_mean": self.normalizer.mean,
            "normalizer_std": self.normalizer.std,
            "normalizer_running_sum": self.normalizer.running_sum,
            "normalizer_running_sumsq": self.normalizer.running_sumsq,
            "normalizer_running_count": self.normalizer.running_count,
            "t": self.t
        },
            f"checkpoints/{self.env_name}_seed{seed}_norb{self.number_of_rbs}_cf{self.clustering_freq}_alpha{self.alpha}.pt")

    def load(self, seed):
        load_path = f"checkpoints/{self.env_name}_seed{seed}_norb{self.number_of_rbs}_cf{self.clustering_freq}_alpha{self.alpha}.pt"
        checkpoint = torch.load(load_path)
        print(f"loading {load_path}")

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target = deepcopy(self.actor)

        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target = deepcopy(self.critic)

        self.normalizer.mean = checkpoint["normalizer_mean"]
        self.normalizer.std = checkpoint["normalizer_std"]
        self.normalizer.running_sum = checkpoint["normalizer_running_sum"]
        self.normalizer.running_sumsq = checkpoint["normalizer_running_sumsq"]
        self.normalizer.running_count = checkpoint["normalizer_running_count"]

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

    def _sample(self, batch_size, beta=1):
        is_weights = []

        total = 0
        total_weighted = 0
        for rb in [*self.cluster_rbs, self.rb]:
            rb_size = rb.get_stored_size()

            total += rb_size
            total_weighted += rb_size ** self.alpha

        samples = []
        for rb in [*self.cluster_rbs, self.rb]:
            normal_sample_amount = round(batch_size * rb.get_stored_size() / total)
            weighted_sample_amount = round(batch_size * rb.get_stored_size() ** self.alpha / total_weighted)

            if weighted_sample_amount > 0:
                samples.append(rb.sample(weighted_sample_amount))
                is_weights += weighted_sample_amount * [(normal_sample_amount / weighted_sample_amount) ** beta]

        normalized_is_weights = torch.Tensor(is_weights).unsqueeze(dim=1) / max(is_weights)

        samples_dict = {}
        for key in samples[0]:
            samples_dict[key] = np.concatenate([*[rb[key] for rb in samples]])

        return samples_dict, normalized_is_weights

    def _learn(self):
        if self.number_of_rbs > 1:
            sample, is_weights = self._sample(self.batch_size, beta=self.beta)
            self.beta = min(1.0, self.beta + 2.5e-6)
            # TODO: 2.5e-6 should be dynamic based on max timesteps: (1-self.beta) / max_timesteps
        else:
            sample = self.rb.sample(self.batch_size)
            is_weights = 1

        obs = torch.Tensor(sample['obs'])
        action = torch.Tensor(sample['action'])
        reward = torch.Tensor(sample['reward'])
        next_obs = torch.Tensor(sample['next_obs'])
        done = torch.Tensor(sample['done'])

        normalized_obs = self.normalizer.normalize(obs).float()
        normalized_next_obs = self.normalizer.normalize(next_obs).float()

        Q_current1, Q_current2 = self.critic(normalized_obs, action)
        with torch.no_grad():
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (
                    self.actor_target(normalized_next_obs) + noise
            ).clamp(torch.from_numpy(self.action_bounds['low']), torch.from_numpy(self.action_bounds['high']))

            Q1_target_next, Q2_target_next = self.critic_target(normalized_next_obs, next_actions)
            Q_target_next = torch.min(Q1_target_next, Q2_target_next)
            Q_target = reward + self.gamma * Q_target_next * (1 - done)

        critic_loss = (is_weights * ((Q_current1 - Q_target) ** 2 + (Q_current2 - Q_target) ** 2)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.t % self.policy_freq == 0:
            actor_loss = -self.critic(normalized_obs, self.actor(normalized_obs))[0].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
