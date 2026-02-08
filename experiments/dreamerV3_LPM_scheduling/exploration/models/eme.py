import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for visual inputs - same architecture as RND for fair comparison"""
    def __init__(self, input_shape):
        super(CNNFeatureExtractor, self).__init__()
        
        print(f"EME CNNFeatureExtractor - input_shape: {input_shape}")
        
        # input_shape is (channels, height, width) - use directly
        channels, height, width = input_shape
        
        # Validate dimensions
        if height < 36 or width < 36:
            raise ValueError(f"Input dimensions {height}x{width} too small. Minimum size is 36x36 for this CNN architecture.")
        
        # Standard Atari CNN architecture (same as RND for comparable size)
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate output size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        
        print(f"EME Conv output dimensions: {conv_height} x {conv_width}")
        
        if conv_width <= 0 or conv_height <= 0:
            raise ValueError(f"Convolution output dimensions are invalid: {conv_height}x{conv_width}")
        
        conv_output_size = conv_width * conv_height * 64
        print(f"EME Conv output size: {conv_output_size}")
        
        # Dense layer for fixed 512-dim output
        self.fc = nn.Linear(conv_output_size, 512)
        self.feature_size = 512

    def forward(self, x):
        # Apply convolutions with LeakyReLU
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Final dense layer
        x = F.leaky_relu(self.fc(x))
        
        return x

class PolicyNet(nn.Module):
    """Simple policy network for computing action probabilities"""
    def __init__(self, input_dim, num_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.policy_head(x), dim=-1)

class RewardEnsemble(nn.Module):
    """Ensemble of reward models for diversity-enhanced scaling factor"""
    def __init__(self, input_dim, ensemble_size=6):
        super(RewardEnsemble, self).__init__()
        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(), 
                nn.Linear(128, 1)
            ) for _ in range(ensemble_size)
        ])
        
        # Initialize each model differently
        for i, model in enumerate(self.models):
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2) * (1 + i * 0.1))
                    nn.init.constant_(m.bias, 0.01 * i)
    
    def forward(self, states, actions):
        # Combine state and action features
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(-1)
        inputs = torch.cat([states, actions.float()], dim=-1)
        
        predictions = []
        for model in self.models:
            pred = model(inputs)
            predictions.append(pred)
        
        return torch.stack(predictions, dim=0)  # (ensemble_size, batch_size, 1)
    
    def compute_variance(self, states, actions):
        """Compute variance of predictions across ensemble"""
        predictions = self.forward(states, actions)  # (ensemble_size, batch_size, 1)
        variance = torch.var(predictions, dim=0).squeeze(-1)  # (batch_size,)
        return variance

class EMENetwork(nn.Module):
    """EME Network with target and predictor networks plus reward ensemble"""
    
    def __init__(self, input_shape, num_actions, embedding_dim=512, lr=0.001, ensemble_size=6):
        super(EMENetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        
        # Target network (randomly initialized, frozen)
        self.target_network = CNNFeatureExtractor(input_shape)
        self.target_head = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize and freeze target network
        self._init_target_network()
        for param in self.target_network.parameters():
            param.requires_grad = False
        for param in self.target_head.parameters():
            param.requires_grad = False
            
        # Predictor network (trainable)
        self.predictor_network = CNNFeatureExtractor(input_shape)
        self.predictor_head = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Policy network for computing action distributions
        self.policy_net = PolicyNet(512, num_actions)
        
        # Reward ensemble for diversity-enhanced scaling
        self.reward_ensemble = RewardEnsemble(512 + 1, ensemble_size)  # +1 for action
        
        # Initialize predictor network
        self._init_predictor_network()
        
        # Optimizers
        self.predictor_optimizer = optim.Adam(
            list(self.predictor_network.parameters()) + 
            list(self.predictor_head.parameters()) +
            list(self.policy_net.parameters()), 
            lr=lr
        )
        
        self.ensemble_optimizer = optim.Adam(
            self.reward_ensemble.parameters(),
            lr=lr
        )
        
        print(f"EME Target network parameters: {sum(p.numel() for p in self.target_network.parameters()):,}")
        print(f"EME Predictor network parameters: {sum(p.numel() for p in self.predictor_network.parameters()):,}")
        
        # Test initial prediction error
        self._test_initial_diversity()
    
    def _init_target_network(self):
        """Initialize target network with orthogonal weights"""
        for module in self.target_network.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
                
        for module in self.target_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def _init_predictor_network(self):
        """Initialize predictor network with xavier uniform weights"""
        for module in self.predictor_network.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.normal_(module.bias, 0, 0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.normal_(module.bias, 0, 0.01)
                
        for module in self.predictor_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.normal_(module.bias, 0, 0.01)
    
    def _test_initial_diversity(self):
        """Test that target and predictor networks produce different outputs"""
        test_input = torch.randn(1, *self.input_shape)
        
        with torch.no_grad():
            target_features = self.target_network(test_input)
            target_emb = self.target_head(target_features)
            
            pred_features = self.predictor_network(test_input)
            pred_emb = self.predictor_head(pred_features)
            
            initial_error = F.mse_loss(pred_emb, target_emb).item()
            print(f"EME Initial prediction error: {initial_error:.6f}")
            
            if initial_error < 1e-6:
                print("WARNING: EME networks are too similar!")
            else:
                print("EME networks properly initialized with sufficient diversity.")
        
    def forward(self, states):
        """Forward pass through both networks"""
        # Target network (no gradients)
        with torch.no_grad():
            target_features = self.target_network(states)
            target_embeddings = self.target_head(target_features)
            
        # Predictor network (with gradients)
        predictor_features = self.predictor_network(states)
        predictor_embeddings = self.predictor_head(predictor_features)
        
        return target_embeddings, predictor_embeddings, predictor_features
    
    def compute_eme_metric(self, states1, states2, actions1, actions2):
        """Compute the EME metric between two state-action pairs"""
        # Get features for policy computation
        features1 = self.predictor_network(states1)
        features2 = self.predictor_network(states2)
        
        # Compute policy distributions
        policy1 = self.policy_net(features1)
        policy2 = self.policy_net(features2)
        
        # Compute KL divergence term
        kl_div = F.kl_div(torch.log(policy1 + 1e-8), policy2, reduction='none').sum(dim=-1)
        
        # Compute reward difference (using ensemble mean as approximation)
        reward_pred1 = self.reward_ensemble(features1, actions1).mean(dim=0).squeeze(-1)
        reward_pred2 = self.reward_ensemble(features2, actions2).mean(dim=0).squeeze(-1)
        reward_diff = torch.abs(reward_pred1 - reward_pred2)
        
        # Compute state embedding distance (next state term)
        _, pred_emb1, _ = self.forward(states1)
        _, pred_emb2, _ = self.forward(states2) 
        state_dist = torch.norm(pred_emb1 - pred_emb2, dim=-1)
        
        # EME metric: reward diff + gamma * state dist + gamma * KL div
        gamma = 0.99
        eme_distance = reward_diff + gamma * state_dist + gamma * kl_div
        
        return eme_distance
    
    def compute_intrinsic_reward(self, states, actions, next_states, next_actions):
        """Compute EME intrinsic rewards"""
        with torch.no_grad():
            # Compute EME distance between current and next states
            eme_dist = self.compute_eme_metric(states, next_states, actions, next_actions)
            
            # Get features for scaling factor
            features = self.predictor_network(states)
            
            # Compute diversity-enhanced scaling factor using reward ensemble variance
            scaling_factor = self.reward_ensemble.compute_variance(features, actions)
            
            # Apply scaling with min/max bounds
            scaling_factor = torch.clamp(scaling_factor, min=1.0, max=10.0)  # M=10 for Atari
            
            # Final intrinsic reward
            intrinsic_reward = eme_dist * scaling_factor
            
        return intrinsic_reward
    
    def compute_predictor_loss(self, states):
        """Compute loss for training the predictor network"""
        target_embeddings, predictor_embeddings, _ = self.forward(states)
        
        # MSE loss between predictor and target
        loss = F.mse_loss(predictor_embeddings, target_embeddings)
        
        return loss
    
    def compute_ensemble_loss(self, states, actions, rewards):
        """Compute loss for training the reward ensemble"""
        features = self.predictor_network(states).detach()
        predictions = self.reward_ensemble(features, actions)  # (ensemble_size, batch_size, 1)
        
        # MSE loss for each model in ensemble
        target_rewards = rewards.unsqueeze(0).unsqueeze(-1).expand(predictions.shape)
        loss = F.mse_loss(predictions, target_rewards)
        
        return loss

class EffectiveMetricExploration:
    """
    EME (Effective Metric-based Exploration) Curiosity model
    
    This implements the method from "Rethinking Exploration in Reinforcement Learning 
    with Effective Metric-Based Exploration Bonus" (NeurIPS 2024)
    """
    
    def __init__(self, obs_size, act_size, state_size, device='cuda' if torch.cuda.is_available() else 'cpu',
                 eta=1.0, lr=0.001, embedding_dim=512, ensemble_size=6, **kwargs):
        """
        Initialize the EME Curiosity model.
        
        Args:
            obs_size: Observation space shape (channels, height, width)
            act_size: Number of possible actions
            state_size: State size (same as obs_size for visual inputs)
            device: Device to run computation on
            eta: Scaling factor for intrinsic rewards
            lr: Learning rate
            embedding_dim: Dimension of the embedding space
            ensemble_size: Number of models in reward ensemble
        """
        print(f"EffectiveMetricExploration - obs_size: {obs_size}, act_size: {act_size}")
        
        self.device = device
        self.eta = eta
        self.act_size = act_size
        self.embedding_dim = embedding_dim
        
        # Validate input shape
        if len(obs_size) != 3:
            raise ValueError(f"Expected obs_size to have 3 dimensions, got {len(obs_size)}: {obs_size}")
        
        self.input_shape = obs_size
        print(f"Using input_shape: {self.input_shape}")
        
        # Initialize EME network
        self.network = EMENetwork(
            self.input_shape, 
            act_size, 
            embedding_dim, 
            lr, 
            ensemble_size
        ).to(device)
        
        print(f"EffectiveMetricExploration initialized with eta={eta}, lr={lr}, embedding_dim={embedding_dim}")
        
    def _preprocess_observations(self, observations):
        """Convert observations to proper tensor format"""
        # Convert to tensor if needed
        if not isinstance(observations, torch.Tensor):
            observations = torch.FloatTensor(observations)
        
        # Move to device
        observations = observations.to(self.device)
        
        return observations
        
    def curiosity(self, observations, actions, next_observations):
        """
        Compute curiosity rewards for a batch of transitions.
        
        Args:
            observations: Current observations
            actions: Actions taken
            next_observations: Next observations
        Returns:
            intrinsic_rewards: Numpy array of intrinsic rewards
        """
        self.network.eval()
        
        # Preprocess inputs
        obs = self._preprocess_observations(observations)
        next_obs = self._preprocess_observations(next_observations)
        
        # Convert actions to tensor
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions)
        actions = actions.to(self.device)
        
        # For next actions, we need to compute them or use dummy values
        # In practice, you might want to pass next_actions as a parameter
        # For now, we'll use the current actions as an approximation
        next_actions = actions  # This is a simplification
        
        # Debug: Check input validity
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print("WARNING: EME received invalid observations (NaN/Inf)")
            return np.zeros(obs.shape[0])
        
        # Compute intrinsic rewards
        intrinsic_rewards = self.network.compute_intrinsic_reward(obs, actions, next_obs, next_actions)
        
        # Debug: Check reward validity
        if torch.isnan(intrinsic_rewards).any() or torch.isinf(intrinsic_rewards).any():
            print("WARNING: EME computed invalid rewards (NaN/Inf)")
            return np.zeros(intrinsic_rewards.shape[0])
        
        # Debug: Print statistics occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 1000 == 0:
            print(f"EME Debug - Raw rewards: mean={intrinsic_rewards.mean().item():.6f}, "
                  f"std={intrinsic_rewards.std().item():.6f}, "
                  f"min={intrinsic_rewards.min().item():.6f}, "
                  f"max={intrinsic_rewards.max().item():.6f}")
        
        # Scale by eta
        intrinsic_rewards = self.eta * intrinsic_rewards
        
        return intrinsic_rewards.cpu().numpy()
    
    def train(self, observations, actions, next_observations, rewards=None):
        """
        Update the EME networks.
        
        Args:
            observations: Current observations
            actions: Actions taken
            next_observations: Next observations
            rewards: Environment rewards (for training reward ensemble)
        Returns:
            loss_info: Tuple of (predictor_loss, ensemble_loss)
        """
        self.network.train()
        
        # Preprocess inputs
        obs = self._preprocess_observations(observations)
        next_obs = self._preprocess_observations(next_observations)
        
        # Convert actions to tensor
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions)
        actions = actions.to(self.device)
        
        # Train predictor network
        predictor_loss = self.network.compute_predictor_loss(next_obs)
        
        self.network.predictor_optimizer.zero_grad()
        predictor_loss.backward()
        self.network.predictor_optimizer.step()
        
        # Train reward ensemble if rewards are provided
        ensemble_loss = torch.tensor(0.0)
        if rewards is not None:
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.FloatTensor(rewards)
            rewards = rewards.to(self.device)
            
            ensemble_loss = self.network.compute_ensemble_loss(obs, actions, rewards)
            
            self.network.ensemble_optimizer.zero_grad()
            ensemble_loss.backward()
            self.network.ensemble_optimizer.step()
        
        return (predictor_loss.item(), ensemble_loss.item())
    
    def save(self, filepath):
        """Save the EME model"""
        torch.save({
            'target_network_state_dict': self.network.target_network.state_dict(),
            'target_head_state_dict': self.network.target_head.state_dict(),
            'predictor_network_state_dict': self.network.predictor_network.state_dict(),
            'predictor_head_state_dict': self.network.predictor_head.state_dict(),
            'policy_net_state_dict': self.network.policy_net.state_dict(),
            'reward_ensemble_state_dict': self.network.reward_ensemble.state_dict(),
            'predictor_optimizer_state_dict': self.network.predictor_optimizer.state_dict(),
            'ensemble_optimizer_state_dict': self.network.ensemble_optimizer.state_dict(),
            'hyperparameters': {
                'input_shape': self.input_shape,
                'act_size': self.act_size,
                'eta': self.eta,
                'embedding_dim': self.embedding_dim
            }
        }, filepath)
        print(f"EME model saved to {filepath}")
    
    def load(self, filepath):
        """Load the EME model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network states
        self.network.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.network.target_head.load_state_dict(checkpoint['target_head_state_dict'])
        self.network.predictor_network.load_state_dict(checkpoint['predictor_network_state_dict'])
        self.network.predictor_head.load_state_dict(checkpoint['predictor_head_state_dict'])
        self.network.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.network.reward_ensemble.load_state_dict(checkpoint['reward_ensemble_state_dict'])
        self.network.predictor_optimizer.load_state_dict(checkpoint['predictor_optimizer_state_dict'])
        self.network.ensemble_optimizer.load_state_dict(checkpoint['ensemble_optimizer_state_dict'])
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.eta = hyperparams['eta']
        self.embedding_dim = hyperparams['embedding_dim']
        
        print(f"EME model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    print("Testing EffectiveMetricExploration...")
    
    # Example for Atari-style visual input
    obs_size = (4, 84, 84)  # Frame stack format (channels, height, width)
    act_size = 6            # Typical Atari action space
    state_size = obs_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize EME curiosity model with Atari-specific hyperparameters
    curiosity_model = EffectiveMetricExploration(
        obs_size, act_size, state_size, device, 
        eta=0.5,  # From paper's Atari experiments
        lr=0.0001,  # From paper's hyperparameters 
        embedding_dim=512,
        ensemble_size=6  # Default from paper
    )
    
    # Test with random data - normalized as in real usage
    batch_size = 4
    # Generate random pixel values and apply normalization
    obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
    next_obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
    
    # Apply z-score normalization (mean=0, std=1)
    obs = (obs_raw - obs_raw.mean()) / (obs_raw.std() + 1e-8)
    next_obs = (next_obs_raw - next_obs_raw.mean()) / (next_obs_raw.std() + 1e-8)
    actions = torch.randint(0, act_size, (batch_size,))
    rewards = torch.randn(batch_size)  # Random rewards for ensemble training
    
    print(f"Testing with batch_size={batch_size}")
    print(f"obs shape: {obs.shape}, next_obs shape: {next_obs.shape}, actions shape: {actions.shape}")
    
    # Test curiosity computation
    intrinsic_rewards = curiosity_model.curiosity(obs, actions, next_obs)
    print(f"Intrinsic rewards shape: {intrinsic_rewards.shape}, mean: {intrinsic_rewards.mean():.4f}")
    
    # Test training
    loss_info = curiosity_model.train(obs, actions, next_obs, rewards)
    print(f"Training losses - Predictor: {loss_info[0]:.4f}, Ensemble: {loss_info[1]:.4f}")
    
    # Test multiple training iterations
    print(f"\nTesting learning over multiple iterations...")
    initial_rewards = intrinsic_rewards.copy()
    
    for i in range(10):
        loss_info = curiosity_model.train(obs, actions, next_obs, rewards)
        if i % 3 == 0:
            print(f"  Iteration {i+1}: Predictor Loss = {loss_info[0]:.4f}, Ensemble Loss = {loss_info[1]:.4f}")
    
    # Check if rewards change after training
    final_rewards = curiosity_model.curiosity(obs, actions, next_obs)
    print(f"\nReward change after training:")
    print(f"  Initial rewards mean: {initial_rewards.mean():.4f}")
    print(f"  Final rewards mean: {final_rewards.mean():.4f}")
    print(f"  Change: {final_rewards.mean() - initial_rewards.mean():.4f}")
    
    print("EffectiveMetricExploration test completed successfully!")