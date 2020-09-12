import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """
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

        # init probs as states by actions sized array
        probs = np.zeros((len(states), len(actions)))

        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,1] = 1 
            else:
                probs[i,0] = 1 

        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        probs = self.get_probs(state)


        return action

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """

    # def __init__():
    #     self.probs = self.get_probs

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

        # init probs as states by actions sized array
        probs = np.zeros((21, len(actions)))

        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE

        # need the state action probs for this, but without any init in this class? reinit?
        probs = np.zeros((21, 2))
        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        state = state[0]
        action_probs = self.probs[state]
        action = np.random.choice(0,1, p=action_probs)

        return action

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """

    # def __init__():
    #     self.probs = self.get_probs

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

        # init probs as states by actions sized array
        probs = np.zeros((21, len(actions)))

        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE

        # need the state action probs for this, but without any init in this class? reinit?
        probs = np.zeros((21, 2))
        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        state = state[0]
        action_probs = probs[state]
        action = np.random.choice(0,1, p=action_probs)

        return action

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """

    # def __init__():
    #     self.probs = self.get_probs

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

        # init probs as states by actions sized array
        probs = np.zeros((21, len(actions)))

        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE

        # need the state action probs for this, but without any init in this class? reinit?
        probs = np.zeros((21, 2))
        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        state = state[0]
        action_probs = probs[state]
        print(action_probs)
        action = np.random.choice(0,1, p=action_probs)

        return action

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """

    # def __init__():
    #     self.probs = self.get_probs

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

        # init probs as states by actions sized array
        probs = np.zeros((21, len(actions)))

        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE

        # need the state action probs for this, but without any init in this class? reinit?
        probs = np.zeros((21, 2))
        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        state = state[0]
        action_probs = probs[state]
        print(action_probs)
        action = np.random.choice([0,1], 1, p=action_probs)

        return action

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """

    # def __init__():
    #     self.probs = self.get_probs

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

        # init probs as states by actions sized array
        probs = np.zeros((21, len(actions)))

        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE

        # need the state action probs for this, but without any init in this class? reinit?
        probs = np.zeros((21, 2))
        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        state = state[0]
        action_probs = probs[state]
        action = np.random.choice([0,1], 1, p=action_probs)

        return action

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """

    # def __init__():
    #     self.probs = self.get_probs

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

        # init probs as states by actions sized array
        probs = np.zeros((21, len(actions)))

        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE

        # need the state action probs for this, but without any init in this class? reinit?
        probs = np.zeros((21, 2))
        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        state = state[0]
        action_probs = probs[state]
        action = np.random.choice([0,1], 1, p=action_probs)

        return action

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """

    # def __init__():
    #     self.probs = self.get_probs

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

        # init probs as states by actions sized array
        probs = np.zeros((22, len(actions)))

        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE

        # need the state action probs for this, but without any init in this class? reinit?
        probs = np.zeros((22, 2))
        # loop over states
        for i in range(np.size(probs, 0)):
            if i == 20 or i == 21:
                probs[i,0] = 1 
            else:
                probs[i,1] = 1 

        state = state[0]
        action_probs = probs[state]
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
        state, reward, done = env.step(actions[-1])
        rewards.append(reward)
        dones.append(done)
        if not done:
            states.append(state)
            actions.append(policy.sample_action(states[-1]))

    return states, actions, rewards, dones

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
        print(actions[-1])
        state, reward, done = env.step(actions[-1])
        rewards.append(reward)
        dones.append(done)
        if not done:
            states.append(state)
            actions.append(policy.sample_action(states[-1]))

    return states, actions, rewards, dones

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
        print(actions[-1][0])
        state, reward, done = env.step(actions[-1])
        rewards.append(reward)
        dones.append(done)
        if not done:
            states.append(state)
            actions.append(policy.sample_action(states[-1]))

    return states, actions, rewards, dones

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
        print(actions[-1][0])
        state, reward, done = env.step(actions[-1][0])
        rewards.append(reward)
        dones.append(done)
        if not done:
            states.append(state)
            actions.append(policy.sample_action(states[-1]))

    return states, actions, rewards, dones

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
        print(env.step(actions[-1][0]))
        state, reward, done = env.step(actions[-1][0])
        rewards.append(reward)
        dones.append(done)
        if not done:
            states.append(state)
            actions.append(policy.sample_action(states[-1]))

    return states, actions, rewards, dones

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
        # print(env.step(actions[-1][0]))
        state, reward, done, _ = env.step(actions[-1][0])
        rewards.append(reward)
        dones.append(done)
        if not done:
            states.append(state)
            actions.append(policy.sample_action(states[-1]))

    return states, actions, rewards, dones

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
        # print(env.step(actions[-1][0]))
        state, reward, done, _ = env.step(actions[-1][0])
        rewards.append(reward)
        dones.append(done)
        if not done:
            states.append(state)
            actions.append(policy.sample_action(states[-1]))
        else:
            break

    return states, actions, rewards, dones

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
        # print(env.step(actions[-1][0]))
        state, reward, done, _ = env.step(actions[-1][0])
        rewards.append(reward)
        dones.append(done)
        if not done:
            states.append(state)
            actions.append(policy.sample_action(states[-1])[0])
        else:
            break

    return states, actions, rewards, dones

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
        # print(env.step(actions[-1][0]))
        state, reward, done, _ = env.step(actions[-1][0])
        rewards.append(reward)
        dones.append(done)
        if not done:
            states.append(state)
            actions.append(policy.sample_action(states[-1]))
        else:
            break

    return states, actions, rewards, dones

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
    for i in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sample_episode(env, policy) 
        for state in states:
            V[state] += 1/returns_count * (rewards[-1] - V[state])
    
    return V

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
    for i in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sample_episode(env, policy) 
        for state in states:
            returns_count[state] += 1
            V[state] += 1/returns_count[state] * (rewards[-1] - V[state])
            
    
    return V

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
    for i in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sample_episode(env, policy) 
        for state in states[0]:
            returns_count[state] += 1
            V[state] += 1/returns_count[state] * (rewards[-1] - V[state])
            
    
    return V

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
    for i in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sample_episode(env, policy) 

        states = [state[0] for state in states]

        for state in states:
            returns_count[state] += 1
            V[state] += 1/returns_count[state] * (rewards[-1] - V[state])
            
    
    return V

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
    for i in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sample_episode(env, policy) 

        # (filter only part of state that is taken into account) 
        # states = [state[0] for state in states]

        for state in states:
            returns_count[state] += 1
            V[state] += 1/returns_count[state] * (rewards[-1] - V[state])
            
    
    return V

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
    for i in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sample_episode(env, policy) 

        # (filter only part of state that is taken into account) 
        states = [state[0] for state in states]

        for state in states:
            returns_count[state] += 1
            V[state] += 1/returns_count[state] * (rewards[-1] - V[state])
            
    
    return V

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
    for i in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sample_episode(env, policy) 

        # (filter only part of state that is taken into account) 
        # states = [state[0] for state in states]

        for state in states:
            returns_count[state] += 1
            V[state] += 1/returns_count[state] * (rewards[-1] - V[state])
            
    
    return V
