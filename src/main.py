import random
import gym
import numpy as np
import json
import torch
import argparse
from orig_td3_agent import OrigTD3Agent
from mer_td3_agent import MERTD3Agent
from time import time
import pandas as pd


def read_hyperparams():
    with open('src/hyperparams.json') as f:
        hyperparams = json.load(f)
        return hyperparams


def get_args():
    parser = argparse.ArgumentParser(description='options')
    parser.add_argument('--env_name', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--mer', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cont', default=False, action='store_true', help="use already saved policy in training")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return args


def test(env, mer):
    if mer:
        agent = MERTD3Agent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_bounds={"low": env.action_space.low, "high": env.action_space.high},
            env_name=env.unwrapped.spec.id
        )
    else:
        agent = OrigTD3Agent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_bounds={"low": env.action_space.low, "high": env.action_space.high},
            env_name=env.unwrapped.spec.id
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

    if mer:
        agent = MERTD3Agent(
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
            policy_freq=hyperparams['policy_freq']
        )
    else:
        agent = OrigTD3Agent(
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
            policy_freq=hyperparams['policy_freq']
        )

    if cont:
        agent.load()

    start = time()
    scores = []

    max_episodes = hyperparams['max_episodes']
    for i in range(1, max_episodes+1):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            real_done = done if env._elapsed_steps < env._max_episode_steps else False

            agent.step(obs, action, reward, next_obs, real_done)
            obs = next_obs
            score += reward

        if i % 100 == 0:
            agent.save()

        print(f'ep: {i}/{max_episodes} | score: {score:.2f}')
        scores.append(score)

        if np.mean(scores[-10:]) >= 200:
            print("200 score reached in mean of last 10 episodes")
            break

    end = time()
    print("training completed, elapsed time: ", end - start)

    moving_avg = pd.DataFrame(scores).rolling(window=10).mean().dropna()

    file_name = "mer_results.csv" if mer else "orig_results.csv"
    moving_avg.to_csv(file_name, sep=' ', index=False, header=False)

    agent.save()


def main():
    args = get_args()
    env = gym.make(args.env_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    if args.test:
        test(env, args.mer)
    else:
        train(env, args.mer, args.cont)


if __name__ == "__main__":
    main()
