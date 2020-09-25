import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        # YOUR CODE HERE
        if len(x.shape) != 2:
            x = x.unsqueeze(0)
            assert x.shape[1] == 4
        out = self.l2(self.relu(self.l1(x)))

        return self.softmax(out)
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        # YOUR CODE HERE
        probs = self.forward(obs)
        action_probs = probs[torch.arange(len(actions)), actions.long().squeeze()]
        return action_probs.unsqueeze(1)
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        with torch.no_grad():
            probs = self.forward(obs.T)

        distr = torch.distributions.Categorical(probs)
        action = distr.sample()

        return int(action.item())
        
        

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # YOUR CODE HERE
    states.append(torch.FloatTensor(env.reset()).unsqueeze(1))
    actions.append(policy.sample_action(states[-1]))

    while True:
        state, reward, done, _ = env.step(int(actions[-1]))
        rewards.append(reward)
        dones.append(done)
        if not done:
            states.append(torch.FloatTensor(state).unsqueeze(1))
            actions.append(policy.sample_action(states[-1]))
        else:
            break
    
    assert len(states) == len(actions) == len(rewards) == len(dones)

    states = torch.stack(states).squeeze()
    actions = torch.FloatTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    dones = torch.Tensor(dones).unsqueeze(1)
    
    return states, actions, rewards, dones

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    # Make sure that your function runs in LINEAR TIME
    # Note that the rewards/returns should be maximized 
    # while the loss should be minimized so you need a - somewhere
    # YOUR CODE HERE
    # print(episode)
    states, actions, rewards, _ = episode

    action_probs = policy.get_probs(states, actions)
    log_probs = -torch.log(action_probs)

    # discount_factors = discount_factor ** torch.arange(len(rewards))
    # total_returns = rewards * discount_factors
    
    Gt = []
    for t in range(len(rewards)):
        summed_rewards = 0
        discount_power = 0
        for reward in rewards[t:]:
            summed_rewards += reward * (discount_factor ** discount_power)
            discount_power += 1
        Gt.append(summed_rewards)
    Gt = torch.FloatTensor(Gt).squeeze()

    loss = torch.sum(log_probs.squeeze() * Gt)

    return loss

def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)
    
    episode_durations = []
    for i in range(num_episodes):
        
        # YOUR CODE HERE
        episode = sampling_function(env, policy)
        episode_durations.append(len(episode[0]))

        loss = compute_reinforce_loss(policy, episode, discount_factor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                           
        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        
    return episode_durations
