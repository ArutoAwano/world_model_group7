import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class RandomAgent:
    def __init__(self):
        pass

    def update(self, rollouts):
        # Return reasonable dummy values for logging compatibility
        # These values should be small positive numbers to avoid issues
        value_loss = 0.001
        action_loss = 0.001 
        dist_entropy = 0.001
        return value_loss, action_loss, dist_entropy


class RandomPolicy(object):
    def __init__(self, env, num_processes):
        super(RandomPolicy, self).__init__()
        self.env = env
        self.num_processes = num_processes
        
        # Get action space info
        if hasattr(env, 'action_space'):
            self.action_space = env.action_space
        elif hasattr(env, 'envs') and len(env.envs) > 0:
            self.action_space = env.envs[0].action_space
        else:
            # Fallback - assume discrete action space
            self.action_space = type('ActionSpace', (), {'n': 6})()
        
        print(f"RandomPolicy initialized with action space: {self.action_space}")
        if hasattr(self.action_space, 'n'):
            print(f"Number of actions: {self.action_space.n}")

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 1

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        """
        Generate random actions for each process
        Returns: value, action, action_log_probs, rnn_hxs
        """
        batch_size = inputs.shape[0]
        
        # Generate random actions
        if hasattr(self.action_space, 'n'):
            # Discrete action space
            actions = torch.randint(0, self.action_space.n, (batch_size, 1)).long()
        else:
            # Continuous action space fallback
            actions = torch.randn(batch_size, 1)
        
        # Dummy values (zero value function, random log probs)
        values = torch.zeros(batch_size, 1).float()
        
        # For random policy, log prob is uniform: log(1/n_actions)
        if hasattr(self.action_space, 'n'):
            action_log_probs = torch.full((batch_size, 1), -np.log(self.action_space.n)).float()
        else:
            action_log_probs = torch.zeros(batch_size, 1).float()
        
        # Unchanged hidden states
        new_rnn_hxs = rnn_hxs
        
        return values, actions, action_log_probs, new_rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        """Return zero value estimates for all inputs"""
        batch_size = inputs.shape[0]
        return torch.zeros(batch_size, 1).float()

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        """
        Evaluate given actions under random policy
        Returns: values, action_log_probs, dist_entropy, rnn_hxs
        """
        batch_size = inputs.shape[0]
        
        # Zero values (no value function for random policy)
        values = torch.zeros(batch_size, 1).float()
        
        # For random policy, all actions have equal probability
        if hasattr(self.action_space, 'n'):
            action_log_probs = torch.full((batch_size, 1), -np.log(self.action_space.n)).float()
            # Entropy for uniform distribution: log(n)
            dist_entropy = np.log(self.action_space.n)
        else:
            action_log_probs = torch.zeros(batch_size, 1).float()
            dist_entropy = 1.0  # Dummy entropy
        
        return values, action_log_probs, dist_entropy, rnn_hxs