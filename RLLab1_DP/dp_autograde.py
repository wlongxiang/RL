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

def get_Q(state, V, env, discount_factor):
    """
    Calculate the Q for a given state
        
    Returns:
        Vector of length env.nA representing the Q values
    """
    Q = np.zeros(env.nA)
    for action in range(env.nA): 
        for trans_prob, next_state, reward, done in env.P[state][action]:
            Q[action] += trans_prob * (reward + discount_factor * V[next_state])
    return Q

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        # YOUR CODE HERE
        plicy_stable = True
        V = policy_eval_v(policy, env)
        for s in range(policy.shape[0]):
            old_action = np.argmax(policy[s]) # with this we get the index that has max prob.
            # now let's find out the new action
            all_possible_values = []
            for a in range(policy.shape[1]):
                pi_a_s = policy[s][a]
                next_value = 0
                for state_transit in env.P[s][a]:
                    p = state_transit[0]
                    reward = state_transit[2]
                    next_state = state_transit[1]
                    next_value += p*(reward+discount_factor*V[next_state])
                all_possible_values.append(next_value)
            new_action = np.argmax(all_possible_values)
#             print("state is", s)
#             print("old action is", old_action)
#             print("new action is", new_action)
#             print("new action distribution over state is", all_possible_values)
            # we update the policy for this state to deterministic prob.
            policy[s] = np.eye(policy.shape[1])[new_action]
            if new_action != old_action:
                plicy_stable = False
                # we have found a new policy, we assign full probability to the one that is found better
                # kinda of greedy policy update
        if plicy_stable:
            break
    return policy, V

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.        
    """
    
    # Start with an all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        # YOUR CODE HERE
        plicy_stable = True
        for s in range(policy.shape[0]):
            
            all_possible_values = []
            for a in range(policy.shape[1]):
                pi_a_s = policy[s][a]
                next_value = 0
                for state_transit in env.P[s][a]:
                    p = state_transit[0]
                    reward = state_transit[2]
                    next_state = state_transit[1]
                    next_action = np.argmax(policy[next_state])
                    next_value += p*(reward+discount_factor*Q[next_state][next_action])
                all_possible_values.append(next_value)
            Q[s] = all_possible_values

            old_action = np.argmax(policy[s]) # with this we get the index that has max prob.
            new_action = np.argmax(all_possible_values)
            policy[s] = np.eye(policy.shape[1])[new_action]
            if new_action != old_action:
                plicy_stable = False
                # we have found a new policy, we assign full probability to the one that is found better
                # kinda of greedy policy update
        if plicy_stable:
            break
    return policy, Q
