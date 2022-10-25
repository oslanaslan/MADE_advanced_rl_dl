""" Monte-Carlo """
from typing import List

import gym
import numpy as np
from tqdm import tqdm


def simple_strategy(player_current_sum: int, *args) -> int:
        """ Simple strategy """
        if player_current_sum in (19, 20, 21):
            return 0 # stand
        else:
            return 1 # hit


def get_mc_mean_reward(samples_cnt: int = 1000, exp_cnt: int = 1000, max_num_steps: int = 1000) -> List[int]:
    """ Monte-Carlo method """
    mean_reward_lst = []
    env = gym.make("Blackjack-v1")
    env._max_episode_steps = max_num_steps

    for _ in tqdm(range(samples_cnt)):
        total_reward_lst = []

        for _ in range(exp_cnt):
            obs = env.reset()

            for step in range(max_num_steps):
                obs, reward, terminate, _, _ = env.step(simple_strategy(*obs))

                if terminate:
                    total_reward_lst.append(reward)
                    break

        mean_reward_lst.append(np.mean(total_reward_lst))

    env.close()

    return mean_reward_lst
