import argparse
import random

import gym
import numpy as np
import torch
import wandb

from td3_agent import TD3Agent


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v3')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--file_name', type=str, default='result.txt')
    parser.add_argument("--wandb", default=False, action='store_true')

    parser.add_argument("--max_timesteps", type=int, default=int(1e+6))
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
    parser.add_argument("--normalize_is", default=False, action='store_true')
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.5)

    args = parser.parse_args()
    return args


def eval_agent(env, agent, times=1, print_score=False, render=False):
    scores = []

    for _ in range(times):
        obs = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.act(obs, train_mode=False)
            next_obs, reward, done, _ = env.step(action)
            if render:
                env.render()

            obs = next_obs
            score += reward

        scores.append(score)
        if print_score:
            print(score)

    return sum(scores) / len(scores)


def test(env, agent, file_name):
    scores = []
    # TODO: following 3 may require change
    for s in range(3):
        agent.load(s)

        score = eval_agent(env, agent, times=100)
        print(score)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    with open(file_name, "a") as f:
        f.write(f"{avg_score:.2f}\n")


def train(env, agent, args):
    if args.wandb:
        wandb.init(project="multi-replay-buffer-v1", entity="anilz")
        wandb.run.name = f"{args.env_name}_norb{args.number_of_rbs}_{args.seed}"
        wandb.run.save()

        wandb.config.env = args.env_name
        wandb.config.seed = args.seed
        wandb.config.number_of_rbs = args.number_of_rbs

    print('==============================')
    print('env: ', args.env_name)
    print('seed: ', args.seed)
    print('number_of_rbs: ', args.number_of_rbs)
    print('clustering_freq: ', args.clustering_freq)
    print('normalize_is: ', args.normalize_is)
    print('alpha: ', args.alpha)
    print('---')

    if args.cont:
        agent.load()

    obs = env.reset(seed=args.seed)
    score = 0
    for t in range(1, args.max_timesteps + 1):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)

        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        score += reward

        if done:
            print(f'{t}/{args.max_timesteps} | ep score: {score:.2f}')
            if args.wandb:
                wandb.log({"score": score})

            score = 0
            obs = env.reset()

    avg_score = eval_agent(env, agent, times=100)
    print(f"Eval score: {avg_score}")
    if args.wandb:
        wandb.log({"eval_score": avg_score})

    agent.save(args.seed)


def main():
    args = get_args()
    env = gym.make(args.env_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.action_space.seed(args.seed)

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
        normalize_is=args.normalize_is,
        alpha=args.alpha,
        beta=args.beta
    )

    if args.test:
        test(env, agent, args.file_name)
    else:
        train(env, agent, args)


if __name__ == "__main__":
    main()
