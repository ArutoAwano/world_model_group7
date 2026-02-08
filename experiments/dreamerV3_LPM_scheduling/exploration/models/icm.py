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
        
        print(f"ICM CNNFeatureExtractor - input_shape: {input_shape}")
        
        # input_shape is (channels, height, width) - use directly
        channels, height, width = input_shape
        
        # Validate dimensions
        if height < 36 or width < 36:
            raise ValueError(f"Input dimensions {height}x{width} too small. Minimum size is 36x36 for this CNN architecture.")
        
        # Standard Atari CNN architecture (same as other methods for comparable size)
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate output size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        
        print(f"ICM Conv output dimensions: {conv_height} x {conv_width}")
        
        if conv_width <= 0 or conv_height <= 0:
            raise ValueError(f"Convolution output dimensions are invalid: {conv_height}x{conv_width}")
        
        conv_output_size = conv_width * conv_height * 64
        print(f"ICM Conv output size: {conv_output_size}")
        
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

class ICMNetwork(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) implementation
    
    ICM consists of:
    1. Feature network φ(s) that encodes states into features
    2. Inverse model that predicts action from φ(s_t) and φ(s_{t+1})
    3. Forward model that predicts φ(s_{t+1}) from φ(s_t) and a_t
    4. Intrinsic reward = η * ||φ(s_{t+1}) - φ̂(s_{t+1})||^2
    """
    
    def __init__(self, input_shape, num_actions, eta=1.0, beta=0.2, lr=0.001):
        super(ICMNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.eta = eta  # Intrinsic reward scaling
        self.beta = beta  # Loss weighting between inverse and forward models
        
        # Feature network φ(s) - learnable CNN feature extractor
        self.feature_extractor = CNNFeatureExtractor(input_shape)
        feature_size = self.feature_extractor.feature_size
        
        # Inverse model: predicts action given φ(s_t) and φ(s_{t+1})
        # Takes concatenated features [φ(s_t), φ(s_{t+1})] and outputs action probabilities
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
        # Forward model: predicts φ(s_{t+1}) given φ(s_t) and a_t
        # Takes concatenated [φ(s_t), one_hot(a_t)] and outputs predicted φ(s_{t+1})
        self.forward_model = nn.Sequential(
            nn.Linear(feature_size + num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_size)
        )
        
        # Optimizer for the entire ICM model
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        print(f"ICM Feature extractor parameters: {sum(p.numel() for p in self.feature_extractor.parameters()):,}")
        print(f"ICM Inverse model parameters: {sum(p.numel() for p in self.inverse_model.parameters()):,}")
        print(f"ICM Forward model parameters: {sum(p.numel() for p in self.forward_model.parameters()):,}")
        
    def encode_state(self, state):
        """
        Encode state into feature representation using φ(s)
        
        Args:
            state: Raw state observation (batch_size, channels, height, width)
            
        Returns:
            features: Encoded state features φ(s) (batch_size, feature_size)
        """
        return self.feature_extractor(state)
    
    def predict_action(self, state_features, next_state_features):
        """
        Inverse model: predict action given current and next state features
        
        Args:
            state_features: φ(s_t) (batch_size, feature_size)
            next_state_features: φ(s_{t+1}) (batch_size, feature_size)
            
        Returns:
            action_logits: Predicted action logits (batch_size, num_actions)
        """
        # Concatenate current and next state features
        combined_features = torch.cat([state_features, next_state_features], dim=1)
        
        # Predict action
        action_logits = self.inverse_model(combined_features)
        
        return action_logits
    
    def predict_next_state_features(self, state_features, actions):
        """
        Forward model: predict next state features given current features and action
        
        Args:
            state_features: φ(s_t) (batch_size, feature_size)
            actions: a_t as indices (batch_size,)
            
        Returns:
            predicted_next_features: φ̂(s_{t+1}) (batch_size, feature_size)
        """
        # Convert actions to one-hot encoding
        action_one_hot = F.one_hot(actions, num_classes=self.num_actions).float()
        
        # Concatenate state features and action
        combined_input = torch.cat([state_features, action_one_hot], dim=1)
        
        # Predict next state features
        predicted_next_features = self.forward_model(combined_input)
        
        return predicted_next_features
    
    def compute_intrinsic_reward(self, states, next_states, actions):
        """
        Calculate intrinsic reward based on forward model prediction error:
        r_i = η * ||φ(s_{t+1}) - φ̂(s_{t+1})||^2
        
        Args:
            states: Current states (batch_size, channels, height, width)
            next_states: Next states (batch_size, channels, height, width)
            actions: Actions taken (batch_size,)
            
        Returns:
            intrinsic_rewards: Prediction errors as intrinsic rewards (batch_size,)
        """
        with torch.no_grad():
            # Encode states into features
            state_features = self.encode_state(states)
            next_state_features = self.encode_state(next_states)
            
            # Predict next state features using forward model
            predicted_next_features = self.predict_next_state_features(state_features, actions)
            
            # Calculate prediction error (MSE per sample)
            prediction_errors = torch.mean((predicted_next_features - next_state_features) ** 2, dim=1)
            
            # Scale by eta to get intrinsic reward
            intrinsic_rewards = self.eta * prediction_errors
            
        return intrinsic_rewards
    
    def compute_loss(self, states, next_states, actions):
        """
        Update ICM model using combined inverse and forward model losses
        
        Total Loss = β * L_inverse + (1-β) * L_forward
        where:
        - L_inverse = CrossEntropy(predicted_action, actual_action)
        - L_forward = MSE(φ̂(s_{t+1}), φ(s_{t+1}))
        
        Args:
            states: Current states (batch_size, channels, height, width)
            next_states: Next states (batch_size, channels, height, width)
            actions: Actions taken (batch_size,)
            
        Returns:
            total_loss: Combined ICM loss
            inverse_loss: Inverse model loss
            forward_loss: Forward model loss
        """
        # Encode states into features
        state_features = self.encode_state(states)
        next_state_features = self.encode_state(next_states)
        
        # === Inverse Model Loss ===
        # Predict actions given state features
        predicted_action_logits = self.predict_action(state_features, next_state_features)
        
        # Calculate inverse model loss (cross-entropy)
        inverse_loss = F.cross_entropy(predicted_action_logits, actions)
        
        # === Forward Model Loss ===
        # Predict next state features
        predicted_next_features = self.predict_next_state_features(state_features, actions)
        
        # Calculate forward model loss (MSE)
        forward_loss = F.mse_loss(predicted_next_features, next_state_features)
        
        # === Combined Loss ===
        # ICM loss: β * L_inverse + (1-β) * L_forward
        total_loss = self.beta * inverse_loss + (1 - self.beta) * forward_loss
        
        return total_loss, inverse_loss, forward_loss

class IntrinsicCuriosityModuleCuriosity:
    """
    Intrinsic Curiosity Module (ICM) Curiosity model
    
    ICM learns state representations by training both an inverse model and forward model.
    The forward model prediction error serves as the intrinsic reward signal.
    This encourages exploration of states that are difficult to predict.
    """
    
    def __init__(self, obs_size, act_size, state_size, device='cuda' if torch.cuda.is_available() else 'cpu',
                 eta=1.0, beta=0.2, lr=0.001, **kwargs):
        """
        Initialize the ICM Curiosity model.
        
        Args:
            obs_size: Observation space shape (channels, height, width)
            act_size: Number of possible actions
            state_size: State size (same as obs_size for visual inputs)
            device: Device to run computation on
            eta: Scaling factor for intrinsic rewards
            beta: Loss weighting between inverse (β) and forward (1-β) models
            lr: Learning rate for ICM networks
        """
        print(f"IntrinsicCuriosityModuleCuriosity - obs_size: {obs_size}, act_size: {act_size}")
        
        self.device = device
        self.eta = eta
        self.beta = beta
        self.act_size = act_size
        
        # Validate input shape
        if len(obs_size) != 3:
            raise ValueError(f"Expected obs_size to have 3 dimensions, got {len(obs_size)}: {obs_size}")
        
        self.input_shape = obs_size
        print(f"Using input_shape: {self.input_shape}")
        
        # Initialize ICM network
        self.network = ICMNetwork(self.input_shape, act_size, eta, beta, lr).to(device)
        
        print(f"IntrinsicCuriosityModuleCuriosity initialized with eta={eta}, beta={beta}, lr={lr}")
        
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
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions)
        actions = actions.to(self.device)
        
        # Ensure actions are 1D
        if actions.dim() > 1:
            actions = actions.squeeze()
        
        # Debug: Check input validity
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print("WARNING: ICM received invalid observations (NaN/Inf)")
            return np.zeros(obs.shape[0])
        
        if torch.isnan(next_obs).any() or torch.isinf(next_obs).any():
            print("WARNING: ICM received invalid next_observations (NaN/Inf)")
            return np.zeros(next_obs.shape[0])
        
        # Compute intrinsic rewards based on forward model prediction error
        intrinsic_rewards = self.network.compute_intrinsic_reward(obs, next_obs, actions)
        
        # Debug: Check reward validity
        if torch.isnan(intrinsic_rewards).any() or torch.isinf(intrinsic_rewards).any():
            print("WARNING: ICM computed invalid rewards (NaN/Inf)")
            return np.zeros(intrinsic_rewards.shape[0])
        
        # Debug: Print raw reward statistics occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 1000 == 0:  # Print every 1000 calls
            print(f"ICM Debug - Raw rewards: mean={intrinsic_rewards.mean().item():.6f}, "
                  f"std={intrinsic_rewards.std().item():.6f}, "
                  f"min={intrinsic_rewards.min().item():.6f}, "
                  f"max={intrinsic_rewards.max().item():.6f}")
        
        return intrinsic_rewards.cpu().numpy()
    
    def train(self, observations, actions, next_observations):
        """
        Update the ICM model.
        
        Args:
            observations: Current observations
            actions: Actions taken
            next_observations: Next observations
        Returns:
            loss_info: Tuple of (total_loss, inverse_loss, forward_loss)
        """
        self.network.train()
        
        # Preprocess inputs
        obs = self._preprocess_observations(observations)
        next_obs = self._preprocess_observations(next_observations)
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions)
        actions = actions.to(self.device)
        
        # Ensure actions are 1D
        if actions.dim() > 1:
            actions = actions.squeeze()
        
        # Compute ICM losses
        total_loss, inverse_loss, forward_loss = self.network.compute_loss(obs, next_obs, actions)
        
        # Backward pass
        self.network.optimizer.zero_grad()
        total_loss.backward()
        self.network.optimizer.step()
        
        # Return loss info (total_loss, forward_loss for compatibility with other methods)
        return (total_loss.item(), forward_loss.item())
    
    def save(self, filepath):
        """
        Save the ICM model
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.network.optimizer.state_dict(),
            'hyperparameters': {
                'input_shape': self.input_shape,
                'act_size': self.act_size,
                'eta': self.eta,
                'beta': self.beta
            }
        }, filepath)
        print(f"IntrinsicCuriosityModuleCuriosity model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the ICM model
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network states
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.eta = hyperparams['eta']
        self.beta = hyperparams['beta']
        
        print(f"IntrinsicCuriosityModuleCuriosity model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    print("Testing IntrinsicCuriosityModuleCuriosity...")
    
    # Example for Atari-style visual input
    obs_size = (4, 84, 84)  # Frame stack format (channels, height, width)
    act_size = 6            # Typical Atari action space
    state_size = obs_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize ICM curiosity model
    curiosity_model = IntrinsicCuriosityModuleCuriosity(
        obs_size, act_size, state_size, device, 
        eta=1.0, beta=0.2, lr=0.001
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
    print(f"Training losses - Total: {loss_info[0]:.4f}, Forward: {loss_info[1]:.4f}")
    
    # Test multiple training iterations to see learning
    print(f"\nTesting learning over multiple iterations...")
    initial_rewards = rewards.copy()
    
    for i in range(10):
        loss_info = curiosity_model.train(obs, actions, next_obs)
        if i % 3 == 0:
            print(f"  Iteration {i+1}: Total Loss = {loss_info[0]:.4f}, Forward Loss = {loss_info[1]:.4f}")
    
    # Check if rewards change after training
    final_rewards = curiosity_model.curiosity(obs, actions, next_obs)
    print(f"\nReward change after training:")
    print(f"  Initial rewards mean: {initial_rewards.mean():.4f}")
    print(f"  Final rewards mean: {final_rewards.mean():.4f}")
    print(f"  Change: {final_rewards.mean() - initial_rewards.mean():.4f}")
    
    print("IntrinsicCuriosityModuleCuriosity test completed successfully!")