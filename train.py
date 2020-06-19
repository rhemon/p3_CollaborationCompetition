from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
import torch
from agent import *

def maddpg(agents, env, brain_name, state_size, target_score=0.5, max_t=1000, gamma=GAMMA):
    """
    Training function used to expose the agent to the environment and run through it repeated until
    target score is achieved.

    :param agents: Agents that will be trained
    :param env: Environment the agents will be learning
    :param brain_name: Brain name of the environment
    :param state_size: Size of each state
    :param target_score: Target score the agent needs to achieve
    :param max_t: Number of times to iterate each episode
    :param gammma: Discount rate
    """

    scores = []
    scores_window = deque(maxlen=100)
    i_ep = 0

    while True:
        i_ep += 1
        env_info = env.reset(train_mode=True)[brain_name]
        states = np.array(env_info.vector_observations)
        score = np.zeros(2)
        for t in range(max_t):
            state0 = torch.from_numpy(states[0]).float()
            state1 = torch.from_numpy(states[1]).float()
            action0 = np.array(agents[0].act(state0))
            action1 = np.array(agents[1].act(state1))
            env_info = env.step([action0, action1])[brain_name]
            next_states = np.array(env_info.vector_observations)
            dones = np.array(env_info.local_done)
            rewards = np.array(env_info.rewards)
            actions = np.array([action0, action1])
            Agent.add_memory(states[0], action0, rewards[0], next_states[0], dones[0],
                             states[1], action1, rewards[1], next_states[1], dones[1])
            agents[0].step(0)
            agents[1].step(1)
            score += rewards
            states = next_states
            
            done = np.any(dones)
            if done:
                break
                
        scores_window.append(np.max(score))
        scores.append(np.max(score))
        print('\rEpisode {} \tAverage Score: {:.2f} \tMax score: {}'.format(i_ep, np.mean(scores_window), np.max(scores_window)), end="")
        np.savez("scores.npz", scores=scores)
        
        if i_ep % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_ep, np.mean(scores_window)))
            torch.save(agents[0].network.actor.state_dict(), f'actor_0_checkpoint.pth') 
            torch.save(agents[0].network.critic.state_dict(), f'critic_0_checkpoint.pth')
            torch.save(agents[1].network.actor.state_dict(), f'actor_1_checkpoint.pth') 
            torch.save(agents[1].network.critic.state_dict(), f'critic_1_checkpoint.pth')

        if np.mean(scores_window) > target_score and i_episode > 100:
            print('\rSolved goal on episode {} with average score {}'.format(i_ep, np.mean(scores_window)))
            torch.save(agents[0].network.actor.state_dict(), f'actor_0_solution.pth') 
            torch.save(agents[0].network.critic.state_dict(), f'critic_0_solution.pth')
            torch.save(agents[1].network.actor.state_dict(), f'actor_1_solution.pth') 
            torch.save(agents[1].network.critic.state_dict(), f'critic_1_solution.pth')
            break
    return scores

if __name__ == "__main__":
    
    env = UnityEnvironment(file_name='Tennis/Tennis.exe')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)

    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    agents = [Agent(state_size, action_size, 0), Agent(state_size, action_size, 0)]

    scores = maddpg(agents, env, brain_name, state_size, max_t=1000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()