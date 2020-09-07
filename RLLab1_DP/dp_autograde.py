import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        for s in range(policy.shape[0]):
            v = V[s]
            # calculate the updated value
            # assume we have same action space for each state
            updated_value = 0
            for a in range(policy.shape[1]):
                pi_a_s = policy[s][a]
                next_value = 0
                for state_transit in env.P[s][a]:
                    p = state_transit[0]
                    reward = state_transit[2]
                    next_state = state_transit[1]
                    next_value += p*(reward+discount_factor*V[next_state])
                updated_value += pi_a_s * next_value
                
            V[s] = updated_value
            delta = max(delta, abs(v- V[s]))
        print(delta)
        if delta < theta:
            break
        
    return np.array(V)

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    # YOUR CODE HERE
    while True:
        delta = 0
        for s in range(policy.shape[0]):
            v = V[s]
            # calculate the updated value
            # assume we have same action space for each state
            updated_value = 0
            for a in range(policy.shape[1]):
                pi_a_s = policy[s][a]
                next_value = 0
                for state_transit in env.P[s][a]:
                    p = state_transit[0]
                    reward = state_transit[2]
                    next_state = state_transit[1]
                    next_value += p*(reward+discount_factor*V[next_state])
                updated_value += pi_a_s * next_value
                
            V[s] = updated_value
            delta = max(delta, abs(v- V[s]))
#         print(delta)
        if delta < theta:
            break
    return np.array(V)
