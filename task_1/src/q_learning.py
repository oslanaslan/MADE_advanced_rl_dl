""" Tools """
from functools import reduce

import numpy as np
from tqdm import tqdm

from src.blackjack_with_double import BlackjackWithDoubleEnv
from src.blackjack_with_count import BlackjackWithCountEnv


def get_random_q(env, with_count=False):
    state_size = reduce(lambda a, x: a * x, [x.n for x in env.observation_space])
    action_size = env.action_space.n
    Q = np.random.random(size=(state_size, action_size))
    action = {}
    idx = 0

    for cur_sum in range(env.observation_space[0].n):
        for dealer_card in range(env.observation_space[1].n):
            for is_usable_ace in range(env.observation_space[2].n):
                if with_count:
                    for s in range(-1 * env.observation_space[3].n // 2, env.observation_space[3].n // 2 + 1):
                        action[(cur_sum, dealer_card, is_usable_ace, s)] = idx
                else:
                    action[(cur_sum, dealer_card, is_usable_ace)] = idx
                idx += 1

    return Q, action

def compute_policy_by_q(Q, state):
    return np.argmax(Q[state])

def q_learning_episode(env, Q, action, alpha, epsilon, gamma, estimate=False):
    """ One episode of Q-learning """
    s = env.reset()[0]
    a = compute_policy_by_q(Q, action[s]) if (np.random.rand() > epsilon or estimate) else env.action_space.sample()

    for _ in range(1000):
        s_prime, reward, terminated, _, _ = env.step(a)
        a_prime = compute_policy_by_q(Q, action[s_prime]) if (np.random.rand() > epsilon or estimate) else env.action_space.sample()

        if not estimate:
            Q[action[s]][a] = Q[action[s]][a] + alpha * (reward + gamma * np.max(Q[action[s_prime]]) - Q[action[s]][a])

        s, a = s_prime, a_prime

        if terminated:
            break

    return reward

def run_qlearning(episodes_cnt=1000, alpha=0.05, epsilon=0.1, gamma=0.8, with_count: bool = False):
    """ Run Q-learning """
    env = BlackjackWithCountEnv() if with_count else BlackjackWithDoubleEnv()
    Q, action = get_random_q(env, with_count=with_count)
    mean_rewards_lst = []
    cur_mean_reward = 0
    min_reward = np.inf

    for cnt in tqdm(range(episodes_cnt)):
        q_learning_episode(env, Q, action, alpha, epsilon, gamma)
        reward = q_learning_episode(env, Q, action, alpha, epsilon, gamma, estimate=True)
        cur_mean_reward = (cur_mean_reward * cnt + reward) / (cnt + 1)

        if min_reward > cur_mean_reward:
            min_reward = cur_mean_reward

        mean_rewards_lst.append(cur_mean_reward)

    env.close()
    
    return mean_rewards_lst
