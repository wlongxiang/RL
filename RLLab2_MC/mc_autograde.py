import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """

    def __init__(self):
        self.probs = np.zeros((22, 2))
        for i in range(np.size(self.probs, 0)):
            if i == 20 or i == 21:
                self.probs[i,0] = 1 
            else:
                self.probs[i,1] = 1 

    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        # Only relevant state variable is player sum
        if isinstance(states, list):
            states = [state[0] for state in states]
        else:
            states = [state[0] for state in [states]]
        probs = np.zeros(len(actions))
        for i, action in enumerate(actions):
            probs[i] = self.probs[states[i], action]

        return probs
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE

        # Only relevant state variable is player sum
        state = state[0]
        action_probs = self.probs[state]
        action = np.random.choice([0,1], 1, p=action_probs)

        return action

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length. 
        Hint: Do not include the state after the termination in the list of states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # YOUR CODE HERE
    states.append(env.reset())
    actions.append(policy.sample_action(states[-1]))

    while True:
        state, reward, done, _ = env.step(int(actions[-1]))
        rewards.append(reward)
        dones.append(done)
        if not done:
            states.append(state)
            actions.append(policy.sample_action(states[-1]))
        else:
            break
    
    assert len(states) == len(actions) == len(rewards) == len(dones)

    return (states, actions, rewards, dones)


def mc_prediction(env, policy, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    
    # YOUR CODE HERE
    # for i in tqdm(range(num_episodes)):
    #     states, actions, rewards, dones = sample_episode(env, policy) 

    #     for state in states:
    #         returns_count[state] += 1
    #         V[state] += 1/returns_count[state] * (rewards[-1] - V[state])

    for i in tqdm(range(num_episodes)):
        S, A, R, dones = sample_episode(env, policy) 
        G = 0
        # for state in states:
        for i in reversed(range(len(S))):
            G = discount_factor * G + R[i]
            if S[i] not in S[:i]:
                returns_count[S[i]] += 1
                V[S[i]] += 1/returns_count[S[i]] * G
            # returns_count[states] += 1
            # V[state] += 1/returns_count[state] * (rewards[-1] - V[state])
            
    
    return V

class RandomBlackjackPolicy(object):
    """
    A random BlackJack policy.
    """

    def __init__(self):
        self.probs = np.ones((22, 2))/2

    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains 
        a probability of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        if isinstance(states, list):
            states = [state[0] for state in states]
        else:
            states = [state[0] for state in [states]]
        probs = np.zeros(len(actions))
        for i, action in enumerate(actions):
            probs[i] = self.probs[states[i], action]

        return probs
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        state = state[0]
        action_probs = self.probs[state]
        action = np.random.choice([0,1], 1, p=action_probs)

        return action

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)

    # C = defaultdict(float)

    # YOUR CODE HERE
    # for i in tqdm(range(num_episodes)):
    #     states, actions, rewards, dones = sample_episode(env, behavior_policy) 
    #     G = 0
    #     W = 0

        # for i, state in enumerate(states):
        #     returns_count[state] += 1
        #     # V[s] = V[s] + ((1/returns_count[s]) * ((W*G) - V[s]))
        #     V[state] += 1/returns_count[state] * (rewards[-1]*(target_policy.get_probs(state, actions[i])/behavior_policy.get_probs(state, actions[i])) - V[state]) 

    for i in tqdm(range(num_episodes)):
        S, A, R, dones = sample_episode(env, behavior_policy) 
        G = 0
        W = 1

        # for state in states:
        for i in reversed(range(len(S))):
            G = discount_factor * G + R[i]
            if S[i] not in S[:i]:
                returns_count[S[i]] += 1
                V[S[i]] += 1/returns_count[S[i]] * ((G*W) - V[S[i]])
                W = W * (target_policy.get_probs(S[i], A[i]) / behavior_policy.get_probs(S[i], A[i]))

    return V
