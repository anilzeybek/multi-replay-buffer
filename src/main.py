import argparse
import os
import random
from time import time

import gym
import numpy as np
import torch

from td3_agent import TD3Agent


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--max_timesteps", type=int, default=int(1.5e+5))
    parser.add_argument("--expl_noise", type=float, default=0.1)
    parser.add_argument("--start_timesteps", type=int, default=10000)
    parser.add_argument("--buffer_size", type=int, default=200000)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--mer", default=False, action='store_true')
    parser.add_argument("--number_of_rbs", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=0.8)
    parser.add_argument("--beta", type=int, default=0.5)

    args = parser.parse_args()
    return args


def test(env, args):
    agent = TD3Agent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        env_name=env.unwrapped.spec.id,
        mer=args.mer
    )
    agent.load(args.seed)

    scores = []
    for _ in range(1, 10):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(obs, train_mode=False)
            next_obs, reward, done, _ = env.step(action)

            obs = next_obs
            score += reward

        scores.append(score)

    with open("result.txt", "a") as f:
        f.write(f"{np.array(scores).mean()}\n")


def train(env, args):
    agent = TD3Agent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        env_name=env.unwrapped.spec.id,
        expl_noise=args.expl_noise,
        start_timesteps=args.start_timesteps,
        buffer_size=args.buffer_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        mer=args.mer,
        number_of_rbs=args.number_of_rbs,
        alpha=args.alpha,
        beta=args.beta
    )

    if args.cont:
        agent.load()

    start = time()
    scores = []

    obs = env.reset()
    score = 0
    for i in range(1, args.max_timesteps+1):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)

        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        score += reward

        if done:
            scores.append(score)
            obs = env.reset()
            score = 0

        if i % 1000 == 0:
            print(f"{i} | {scores[-1]:.2f}")

    end = time()
    print(f"training completed, elapsed time: {end - start}\n")

    agent.save(args.seed)
    return scores


def main():
    args = get_args()
    env = gym.make(args.env_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    print(f"env: {args.env_name} | mer: {args.mer} | seed: {args.seed}")

    if args.test:
        test(env, args)
    else:
        scores = train(env, args)

        def moving_average(a, n):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        ma_scores = moving_average(scores, n=10)

        os.makedirs("results/", exist_ok=True)
        np.savetxt(f"results/{args.env_name}_s{args.seed}_mer{args.mer}.txt", ma_scores)


if __name__ == "__main__":
    main()
