import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            # if isinstance(obs, int):
            action = np.argmax(self.Q[obs])
            # else:
            #     action = np.argmax(self.Q[obs[0]])

        return action
# misses import in unit test
import random

def sarsa(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        policy: A policy which allows us to sample As with its sample_action method.
        Q: Q value function, numpy array Q[s,a] -> S-action value.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        # YOUR CODE HERE
        S = env.reset()
        A = policy.sample_action(S)
        while True:
            S_prime, reward, done, _ = env.step(A)
            A_prime = policy.sample_action(S_prime)
            
            policy.Q[S,A] += alpha * (reward + discount_factor * policy.Q[S_prime, A_prime] - policy.Q[S,A]) 

            S = S_prime
            A = A_prime

            R += reward 
            i += 1
            if done:
                break

        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)

def q_learning(env, policy, Q, num_episodes, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        # YOUR CODE HERE
        S = env.reset()
        while True:
            A = policy.sample_action(S)
            S_prime, reward, done, _ = env.step(A)
            # A_prime = policy.sample_action(S_prime)
            
            policy.Q[S,A] += alpha * (reward + discount_factor * max(policy.Q[S_prime]) - policy.Q[S,A]) 

            S = S_prime
            # A = A_prime

            R += reward 
            i += 1
            if done:
                break
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)
