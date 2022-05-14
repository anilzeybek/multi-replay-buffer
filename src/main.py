import argparse
import random
from time import time
from datetime import datetime

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from td3_agent import TD3Agent


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v3')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--max_timesteps", type=int, default=int(2e+5))
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
    parser.add_argument("--number_of_rbs", type=int, default=5)
    parser.add_argument("--clustering_freq", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.5)

    args = parser.parse_args()
    return args


def test(env, agent, args):
    agent.load(args.seed)

    scores = []
    for _ in range(1, 20):
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


def train(env, agent, args):
    if args.cont:
        agent.load()

    start = time()
    scores = []
    writer = SummaryWriter(f"./checkpoints/{args.env_name}-{args.seed}-{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")

    obs = env.reset()
    score = 0
    for i in range(1, args.max_timesteps + 1):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)

        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        score += reward

        if done:
            writer.add_scalar("score", score, len(scores))
            scores.append(score)

            obs = env.reset()
            score = 0

        if i % 1000 == 0:
            print(f"{i}/{args.max_timesteps} | {scores[-1]:.2f}")

    end = time()
    print(f"training completed, elapsed time: {end - start}\n")

    writer.add_hparams({
        'number_of_rbs': args.number_of_rbs,
        'clustering_freq': args.clustering_freq,
        'alpha': args.alpha
    }, {"score": np.array(scores[-50:]).mean()})

    agent.save(args.seed)


def main():
    args = get_args()
    env = gym.make(args.env_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    print('env: ', args.env_name)
    print('seed: ', args.seed)
    print('number_of_rbs: ', args.number_of_rbs)
    print('clustering_freq: ', args.clustering_freq)
    print('alpha: ', args.alpha)
    print('---')

    agent = TD3Agent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        env_name=args.env_name,
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
        number_of_rbs=args.number_of_rbs,
        clustering_freq=args.clustering_freq,
        alpha=args.alpha,
        beta=args.beta
    )

    if args.test:
        test(env, agent, args)
    else:
        train(env, agent, args)


if __name__ == "__main__":
    main()
