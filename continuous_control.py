#!/usr/bin/env python3

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

from unityagents import UnityEnvironment
from ddpg_agent import Agent
import torch

# Set to True if you want to (re)run the training or False to just watch the trained agent in action
train_now = True

# Initialize Unity environment with 1 or 20 agents
#env = UnityEnvironment(file_name='/home/schmitz/work/deep-reinforcement-learning/p2_continuous-control/solution/Reacher_1_Linux/Reacher.x86_64')
env = UnityEnvironment(file_name='/home/schmitz/work/deep-reinforcement-learning/p2_continuous-control/solution/Reacher_20_Linux/Reacher.x86_64')

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents in the environment
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# Size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# Examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Create an agent that controls the agent in the environment
agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)

# Train the agent with DDPG
if train_now:
    n_episodes=200
    print_every=10
    scores_deque = deque(maxlen=print_every)
    scores = []
    # Train for n_episodes
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = 0
        actions = [np.array([0.0,0.0,0.0,0.0])]*num_agents
        # Train until environment ends the episode
        while True:
            for env_agent_idx in range(num_agents):
                # Let deep learning agent act based on states
                actions[env_agent_idx] = agent.act(states[env_agent_idx])
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            for env_agent_idx in range(num_agents):
                # Save to replay buffer
                agent.memorize(states[env_agent_idx], actions[env_agent_idx], \
                           rewards[env_agent_idx], next_states[env_agent_idx], \
                           dones[env_agent_idx])
            # Learn
            agent.step()
            states = next_states
            score += np.sum(rewards)/len(rewards)
            if np.any(dones):
                break
        # Check and track scores
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            # Save coefficients to file but only if results are really good
            if np.mean(scores_deque) >= 36:
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#888888', linestyle='-', alpha=0.5)
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig('training_scores.png')
    plt.show()

# Now watch a trained agent in action

agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
agent.reset()

env_info = env.reset(train_mode=False)[brain_name]
states = env_info.vector_observations
actions = [np.array([0.0,0.0,0.0,0.0])]*num_agents
while True:
    for env_agent_idx in range(num_agents):
        # Let deep learning agent act based on states
        actions[env_agent_idx] = agent.act(states[env_agent_idx])
    env_info = env.step(actions)[brain_name]
    states = env_info.vector_observations
    dones = env_info.local_done
    if np.any(dones):
        break

# Clean up
env.close()
