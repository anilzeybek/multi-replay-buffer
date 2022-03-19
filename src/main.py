import random
import gym
import numpy as np
import json
import torch
import argparse
from td3_agent import TD3Agent
from time import time
import matplotlib.pyplot as plt


def read_hyperparams():
    with open('src/hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--mer', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")

    args = parser.parse_args()
    return args


def test(env, mer):
    agent = TD3Agent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        env_name=env.unwrapped.spec.id,
        mer=mer
    )

    agent.load()

    for _ in range(1, 1000):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(obs, train_mode=False)
            next_obs, reward, done, _ = env.step(action)
            env.render()

            obs = next_obs
            score += reward

        print(f"score: {score:.2f}")


def train(env, mer, cont):
    hyperparams = read_hyperparams()

    agent = TD3Agent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_bounds={"low": env.action_space.low, "high": env.action_space.high},
        env_name=env.unwrapped.spec.id,
        expl_noise=hyperparams['expl_noise'],
        start_timesteps=hyperparams['start_timesteps'],
        buffer_size=hyperparams['buffer_size'],
        actor_lr=hyperparams['actor_lr'],
        critic_lr=hyperparams['critic_lr'],
        batch_size=hyperparams['batch_size'],
        gamma=hyperparams['gamma'],
        tau=hyperparams['tau'],
        policy_noise=hyperparams['policy_noise'],
        noise_clip=hyperparams['noise_clip'],
        policy_freq=hyperparams['policy_freq'],
        mer=mer,
        number_of_rbs=hyperparams['number_of_rbs'],
        pow=hyperparams['pow']
    )

    if cont:
        agent.load()

    start = time()
    scores = []

    t = 0
    while t < 5e+5:
        obs = env.reset()
        score = 0
        done = False
        while not done:
            t += 1

            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            real_done = done if env._elapsed_steps < env._max_episode_steps else False

            agent.step(obs, action, reward, next_obs, real_done)
            obs = next_obs
            score += reward

        scores.append(score)

        if t % 100 == 0:
            print(f'timestep: {t} | avg of last 10 scores: {np.mean(scores[-10:])}')

    end = time()
    print("training completed, elapsed time: ", end - start)
    print()

    agent.save()

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
        test(env, args.mer)
    else:
        print("------TRAIN")
        scores = train(env, args.mer, args.cont)

        def moving_average(a, n=3):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        ma_scores = moving_average(scores, n=10)

        plt.plot(ma_scores)
        plt.savefig(f"results/{args.env_name}_s{args.seed}_mer{args.mer}.png")

        np.savetxt(f"results/{args.env_name}_s{args.seed}_mer{args.mer}.txt", ma_scores)


if __name__ == "__main__":
    main()
