from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
import torch
from agent import *

def maddpg(agents, env, brain_name, state_size, target_score=0.5, max_t=1000, gamma=GAMMA):
    
    scores = []
    scores_window = deque(maxlen=100)
    i_ep = 0

    while True:
        i_ep += 1
        env_info = env.reset(train_mode=True)[brain_name]
        states = torch.from_numpy(np.array(env_info.vector_observations)).view(state_size*2)
        score = 0
        for t in range(max_t):
            action0_0 = np.array(agents[0].act(states[:state_size]))
            action1_1 = np.array(agents[1].act(states[state_size:]))
            env_info = env.step([action0_0, action1_1])[brain_name]
            next_states = torch.from_numpy(np.array(env_info.vector_observations)).view(state_size*2)
            rewards = env_info.rewards
            dones = env_info.local_done
            agents[0].step(states, np.concatenate([action0_0, action1_1]), rewards[0], next_states, dones[0], 0)
            agents[1].step(states, np.concatenate([action1_1, action0_0]), rewards[1], next_states, dones[1], 1)
            score += np.max(rewards)
            states = next_states
            if np.any(dones):
                break
                
        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {} \tAverage Score: {:.2f}, Max score: {}, Min score: {}'.format(i_ep, np.mean(scores_window), np.max(scores_window), np.min(scores_window)), end="")
        np.savez("scores.npz", scores=scores)
        
        if i_ep % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_ep, np.mean(scores_window)))
            torch.save(agents[0].network.actor.state_dict(), f'actor_0_checkpoint.pth') 
            torch.save(agents[0].network.critic.state_dict(), f'critic_0_checkpoint.pth')
            torch.save(agents[1].network.actor.state_dict(), f'actor_1_checkpoint.pth') 
            torch.save(agents[1].network.critic.state_dict(), f'critic_1_checkpoint.pth')

        if np.mean(scores_window) > target_score and i_episode > 100:
            print('\rSolved goal on episode {} with average score {}'.format(i_episode, np.mean(scores_window)))
            torch.save(agents[0].network.actor.state_dict(), f'actor_0_solution.pth') 
            torch.save(agents[0].network.critic.state_dict(), f'critic_0_solution.pth')
            torch.save(agents[0].network.actor.state_dict(), f'actor_1_solution.pth') 
            torch.save(agents[0].network.critic.state_dict(), f'critic_1_solution.pth')
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
    agents = [Agent(state_size, action_size, 1), Agent(state_size, action_size, 1)]

    scores = maddpg(agents, env, brain_name, state_size, max_t=1000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()