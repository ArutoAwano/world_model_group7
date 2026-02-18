import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super(CNNFeatureExtractor, self).__init__()
        
        print(f"RND CNNFeatureExtractor - input_shape: {input_shape}")
        
        # input_shape is (channels, height, width) - use directly
        channels, height, width = input_shape
        
        # Validate dimensions
        if height < 36 or width < 36:
            raise ValueError(f"Input dimensions {height}x{width} too small. Minimum size is 36x36 for this CNN architecture.")
        
        # Standard Atari CNN architecture (same as improve.py for comparable size)
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate output size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        
        print(f"RND Conv output dimensions: {conv_height} x {conv_width}")
        
        if conv_width <= 0 or conv_height <= 0:
            raise ValueError(f"Convolution output dimensions are invalid: {conv_height}x{conv_width}")
        
        conv_output_size = conv_width * conv_height * 64
        print(f"RND Conv output size: {conv_output_size}")
        
        # Dense layer for fixed 512-dim output (comparable to other methods)
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

class RNDNetwork(nn.Module):
    """Random Network Distillation implementation with target and predictor networks"""
    
    def __init__(self, input_shape, embedding_dim=512, lr=0.001):
        super(RNDNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        
        # Target network (randomly initialized, frozen)
        self.target_network = CNNFeatureExtractor(input_shape)
        self.target_head = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize target network with different seed for diversity
        self._init_target_network()
        
        # Freeze target network completely
        for param in self.target_network.parameters():
            param.requires_grad = False
        for param in self.target_head.parameters():
            param.requires_grad = False
            
        # Predictor network (trainable) - initialized differently
        self.predictor_network = CNNFeatureExtractor(input_shape)
        self.predictor_head = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize predictor network with different initialization
        self._init_predictor_network()
        
        # Optimizer for predictor network only
        self.optimizer = optim.Adam(
            list(self.predictor_network.parameters()) + 
            list(self.predictor_head.parameters()), 
            lr=lr
        )
        
        print(f"RND Target network parameters: {sum(p.numel() for p in self.target_network.parameters()):,}")
        print(f"RND Predictor network parameters: {sum(p.numel() for p in self.predictor_network.parameters()):,}")
        
        # Test initial prediction error to ensure networks are different
        self._test_initial_diversity()
    
    def _init_target_network(self):
        """Initialize target network with specific random weights"""
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
        """Initialize predictor network with different random weights"""
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
            print(f"RND Initial prediction error: {initial_error:.6f}")
            
            if initial_error < 1e-6:
                print("WARNING: RND networks are too similar! Consider different initialization.")
            else:
                print("RND networks properly initialized with sufficient diversity.")
        
    def forward(self, states):
        """
        Forward pass through both networks
        Args:
            states: Input states (batch_size, channels, height, width)
        Returns:
            target_embeddings: Target network embeddings
            predictor_embeddings: Predictor network embeddings
        """
        # Target network (no gradients)
        with torch.no_grad():
            target_features = self.target_network(states)
            target_embeddings = self.target_head(target_features)
            
        # Predictor network (with gradients)
        predictor_features = self.predictor_network(states)
        predictor_embeddings = self.predictor_head(predictor_features)
        
        return target_embeddings, predictor_embeddings
    
    def compute_intrinsic_reward(self, states):
        """
        Compute RND intrinsic rewards based on prediction error
        Args:
            states: Input states (batch_size, channels, height, width)
        Returns:
            intrinsic_rewards: Prediction errors as intrinsic rewards (batch_size,)
        """
        with torch.no_grad():
            target_embeddings, predictor_embeddings = self.forward(states)
            
            # Calculate MSE between predictor and target (per sample)
            prediction_errors = torch.mean((predictor_embeddings - target_embeddings) ** 2, dim=1)
            
            # Add small epsilon to avoid exactly zero rewards
            prediction_errors = prediction_errors + 1e-8
            
        return prediction_errors
    
    def compute_loss(self, states):
        """
        Compute RND loss for training the predictor network
        Args:
            states: Input states (batch_size, channels, height, width)
        Returns:
            loss: MSE loss between predictor and target embeddings
        """
        target_embeddings, predictor_embeddings = self.forward(states)
        
        # MSE loss between predictor and target
        loss = F.mse_loss(predictor_embeddings, target_embeddings)
        
        return loss

class RandomNetworkDistillationCuriosity:
    """
    Random Network Distillation (RND) Curiosity model
    
    RND measures intrinsic motivation based on how well a predictor network
    can predict the output of a randomly initialized target network.
    The prediction error serves as the intrinsic reward signal.
    """
    
    def __init__(self, obs_size, act_size, state_size, device='cuda' if torch.cuda.is_available() else 'cpu',
                 eta=1.0, lr=0.001, embedding_dim=512, **kwargs):
        """
        Initialize the RND Curiosity model.
        
        Args:
            obs_size: Observation space shape (channels, height, width)
            act_size: Number of possible actions (not used in RND, but kept for interface compatibility)
            state_size: State size (same as obs_size for visual inputs)
            device: Device to run computation on
            eta: Scaling factor for intrinsic rewards
            lr: Learning rate for predictor network
            embedding_dim: Dimension of the embedding space
        """
        print(f"RandomNetworkDistillationCuriosity - obs_size: {obs_size}, act_size: {act_size}")
        
        self.device = device
        self.eta = eta
        self.act_size = act_size
        self.embedding_dim = embedding_dim
        
        # Validate input shape
        if len(obs_size) != 3:
            raise ValueError(f"Expected obs_size to have 3 dimensions, got {len(obs_size)}: {obs_size}")
        
        self.input_shape = obs_size
        print(f"Using input_shape: {self.input_shape}")
        
        # Initialize RND network
        self.network = RNDNetwork(self.input_shape, embedding_dim, lr).to(device)
        
        print(f"RandomNetworkDistillationCuriosity initialized with eta={eta}, lr={lr}, embedding_dim={embedding_dim}")
        
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
        
        Note: For RND, we typically use next_observations as the prediction target,
        as we want to measure how "novel" the new state is.
        
        Args:
            observations: Current observations (not used in RND)
            actions: Actions taken (not used in RND)
            next_observations: Next observations to compute novelty for
        Returns:
            intrinsic_rewards: Numpy array of intrinsic rewards
        """
        self.network.eval()
        
        # Preprocess inputs - for RND we use next_observations
        next_obs = self._preprocess_observations(next_observations)
        
        # Debug: Check input validity
        if torch.isnan(next_obs).any() or torch.isinf(next_obs).any():
            print("WARNING: RND received invalid observations (NaN/Inf)")
            return np.zeros(next_obs.shape[0])
        
        # Compute intrinsic rewards based on prediction error
        intrinsic_rewards = self.network.compute_intrinsic_reward(next_obs)
        
        # Debug: Check reward validity before scaling
        if torch.isnan(intrinsic_rewards).any() or torch.isinf(intrinsic_rewards).any():
            print("WARNING: RND computed invalid rewards (NaN/Inf)")
            return np.zeros(intrinsic_rewards.shape[0])
        
        # Debug: Print raw reward statistics occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 1000 == 0:  # Print every 1000 calls
            print(f"RND Debug - Raw rewards: mean={intrinsic_rewards.mean().item():.6f}, "
                  f"std={intrinsic_rewards.std().item():.6f}, "
                  f"min={intrinsic_rewards.min().item():.6f}, "
                  f"max={intrinsic_rewards.max().item():.6f}")
        
        # Scale by eta
        intrinsic_rewards = self.eta * intrinsic_rewards

        # intrinsic_rewards = torch.clamp(intrinsic_rewards, -2, 2)
        
        return intrinsic_rewards.cpu().numpy()
    
    def train(self, observations, actions, next_observations):
        """
        Update the RND predictor network.
        
        Args:
            observations: Current observations (not used in RND)
            actions: Actions taken (not used in RND) 
            next_observations: Next observations to train predictor on
        Returns:
            loss_info: Tuple of (rnd_loss, 0) for compatibility with other methods
        """
        self.network.train()
        
        # Preprocess inputs - for RND we use next_observations
        next_obs = self._preprocess_observations(next_observations)
        
        # Compute RND loss
        rnd_loss = self.network.compute_loss(next_obs)
        
        # Backward pass
        self.network.optimizer.zero_grad()
        rnd_loss.backward()
        self.network.optimizer.step()
        
        # Return loss info (second value set to 0 for compatibility)
        return (rnd_loss.item(), 0.0)
    
    def save(self, filepath):
        """
        Save the RND model
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'target_network_state_dict': self.network.target_network.state_dict(),
            'target_head_state_dict': self.network.target_head.state_dict(),
            'predictor_network_state_dict': self.network.predictor_network.state_dict(),
            'predictor_head_state_dict': self.network.predictor_head.state_dict(),
            'optimizer_state_dict': self.network.optimizer.state_dict(),
            'hyperparameters': {
                'input_shape': self.input_shape,
                'act_size': self.act_size,
                'eta': self.eta,
                'embedding_dim': self.embedding_dim
            }
        }, filepath)
        print(f"RandomNetworkDistillationCuriosity model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the RND model
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network states
        self.network.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.network.target_head.load_state_dict(checkpoint['target_head_state_dict'])
        self.network.predictor_network.load_state_dict(checkpoint['predictor_network_state_dict'])
        self.network.predictor_head.load_state_dict(checkpoint['predictor_head_state_dict'])
        self.network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.eta = hyperparams['eta']
        self.embedding_dim = hyperparams['embedding_dim']
        
        print(f"RandomNetworkDistillationCuriosity model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    print("Testing RandomNetworkDistillationCuriosity...")
    
    # Example for Atari-style visual input
    obs_size = (4, 84, 84)  # Frame stack format (channels, height, width)
    act_size = 6            # Typical Atari action space
    state_size = obs_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize RND curiosity model
    curiosity_model = RandomNetworkDistillationCuriosity(
        obs_size, act_size, state_size, device, 
        eta=1.0, lr=0.001, embedding_dim=512
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
    
    print(f"Testing with batch_size={batch_size}")
    print(f"obs shape: {obs.shape}, obs mean: {obs.mean():.3f}, obs std: {obs.std():.3f}")
    print(f"next_obs range: [{next_obs.min():.3f}, {next_obs.max():.3f}], actions shape: {actions.shape}")
    
    # Test curiosity computation
    rewards = curiosity_model.curiosity(obs, actions, next_obs)
    print(f"Intrinsic rewards shape: {rewards.shape}, mean: {rewards.mean():.4f}")
    
    # Test training
    loss_info = curiosity_model.train(obs, actions, next_obs)
    print(f"Training losses - RND: {loss_info[0]:.4f}, Placeholder: {loss_info[1]:.4f}")
    
    # Test multiple training iterations to see learning
    print(f"\nTesting learning over multiple iterations...")
    initial_rewards = rewards.copy()
    
    for i in range(10):
        loss_info = curiosity_model.train(obs, actions, next_obs)
        if i % 3 == 0:
            print(f"  Iteration {i+1}: RND Loss = {loss_info[0]:.4f}")
    
    # Check if rewards change after training (they should decrease as predictor gets better)
    final_rewards = curiosity_model.curiosity(obs, actions, next_obs)
    print(f"\nReward change after training:")
    print(f"  Initial rewards mean: {initial_rewards.mean():.4f}")
    print(f"  Final rewards mean: {final_rewards.mean():.4f}")
    print(f"  Change: {final_rewards.mean() - initial_rewards.mean():.4f}")
    
    print("RandomNetworkDistillationCuriosity test completed successfully!")