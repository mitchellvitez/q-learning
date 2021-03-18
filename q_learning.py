import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
from IPython import display

env_name = 'Taxi-v3'
env = gym.make(env_name)

print(env.observation_space.n, env.action_space.n)

# Q(S, A), implemented as a lookup table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

def q_learn(env,
            num_episodes=1000,
            learning_rate=0.1,
            discount_factor=0.75,
            exploration_chance=0.1,
            render=False,
            slow_taxi=False):

    # stores final reward value for each episode
    final_rewards = []

    for current_episode in range(num_episodes):
        state = env.reset()
        done = False
        reward = 0

        while not done:
            # there's a chance of exploring,
            # otherwise pick the action that maximizes Q for this state
            if random.random() < exploration_chance:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # run the step
            observation, reward, done, info = env.step(action)

            # visually show what's happening
            if render or slow_taxi:
                display.clear_output(wait=True)
                env.render()

                # so you can watch the taxi's actions in real time
                if slow_taxi:
                    time.sleep(0.5)

            # update the Q table
            old_value = q_table[state, action]
            q_table[state, action] = old_value + learning_rate * (reward + discount_factor *
                                                                  np.max(q_table[observation] - old_value))
            state = observation

        final_rewards.append(reward)

    return final_rewards

rewards = q_learn(env, render=False)

# moving average of last n trials
moving_average_num = 100
def moving_average(x, n=moving_average_num):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

plt.scatter(np.arange(len(rewards)), rewards, label='individual episodes', alpha=0.8, color='#a1d9f7')
plt.plot(moving_average(rewards), label=f'moving average of last {moving_average_num} episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.title(f'Q Learning on {env_name}')
plt.show()
