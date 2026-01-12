import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class JustPixelsFeatureExtractor(nn.Module):
    """
    Pix2Pix Feature Extractor - just normalizes pixels without CNN processing
    Equivalent to JustPixels class in original implementation
    """
    def __init__(self, input_shape):
        super(JustPixelsFeatureExtractor, self).__init__()
        
        self.input_shape = input_shape
        channels, height, width = input_shape
        
        # Feature size is just the flattened pixel dimensions
        self.feature_size = channels * height * width
        
        # Learnable normalization parameters (equivalent to ob_mean, ob_std)
        self.register_buffer('ob_mean', torch.zeros(input_shape))
        self.register_buffer('ob_std', torch.ones(()))
        
        print(f"JustPixelsFeatureExtractor - input_shape: {input_shape}, feature_size: {self.feature_size}")
        
    def update_normalization_stats(self, observations):
        """Update running mean and std for observations"""
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)
        
        self.ob_mean = observations.mean(dim=0, keepdim=False)
        self.ob_std = observations.std().item()
        
    def forward(self, x):
        """Input is already standardized, just pass through"""
        return x.float()  # No additional normalization needed

class UNetEncoder(nn.Module):
    """UNet Encoder with action conditioning"""
    def __init__(self, input_channels, num_actions):
        super(UNetEncoder, self).__init__()
        
        self.num_actions = num_actions
        
        # Encoder layers with correct padding to get exact output sizes
        # Conv1: 96x96 -> 32x32 (stride=3, kernel=8)
        self.conv1 = nn.Conv2d(input_channels + num_actions, 32, kernel_size=8, stride=3, padding=3)
        # Conv2: 32x32 -> 16x16 (stride=2, kernel=8)  
        self.conv2 = nn.Conv2d(32 + num_actions, 64, kernel_size=8, stride=2, padding=3)
        # Conv3: 16x16 -> 8x8 (stride=2, kernel=4)
        self.conv3 = nn.Conv2d(64 + num_actions, 64, kernel_size=4, stride=2, padding=1)
        
        self.activation = nn.LeakyReLU()
        
    def add_action_conditioning(self, x, action_onehot):
        """Add action conditioning to feature maps"""
        batch_size, channels, height, width = x.shape
        
        # Expand action to spatial dimensions
        action_expanded = action_onehot.view(batch_size, self.num_actions, 1, 1)
        action_expanded = action_expanded.expand(batch_size, self.num_actions, height, width)
        
        # Concatenate along channel dimension
        return torch.cat([x, action_expanded], dim=1)
        
    def forward(self, x, action_onehot):
        """
        Args:
            x: Input features (batch_size, channels, height, width)  
            action_onehot: One-hot encoded actions (batch_size, num_actions)
        Returns:
            features: Encoded features
            skip_connections: List of intermediate features for skip connections
        """
        skip_connections = []
        
        # Add padding to match original implementation (84x84 -> 96x96)
        x = F.pad(x, (6, 6, 6, 6))
        
        # Encoder with action conditioning
        x = self.add_action_conditioning(x, action_onehot)
        x = self.activation(self.conv1(x))  # 96x96 -> 32x32
        skip_connections.append(x)
        
        x = self.add_action_conditioning(x, action_onehot)
        x = self.activation(self.conv2(x))  # 32x32 -> 16x16
        skip_connections.append(x)
        
        x = self.add_action_conditioning(x, action_onehot)
        x = self.activation(self.conv3(x))  # 16x16 -> 8x8
        skip_connections.append(x)
        
        return x, skip_connections

class UNetDecoder(nn.Module):
    """UNet Decoder with skip connections"""
    def __init__(self, feat_dim, num_actions, output_channels):
        super(UNetDecoder, self).__init__()
        
        self.feat_dim = feat_dim
        self.num_actions = num_actions
        self.output_channels = output_channels
        
        # Bottleneck - correct input size: 64 * 8 * 8 + num_actions = 4096 + 6 = 4102
        self.bottleneck = nn.Linear(64 * 8 * 8 + num_actions, feat_dim)
        
        # Residual blocks in bottleneck
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(feat_dim, num_actions) for _ in range(4)
        ])
        
        # Decoder projection
        self.decoder_proj = nn.Linear(feat_dim + num_actions, 64 * 8 * 8)
        
        # Decoder layers with proper input channel calculations for skip connections
        # deconv1: input = 64 (from proj) + 64 (skip from conv3) = 128
        self.deconv1 = nn.ConvTranspose2d(64 + 64 + num_actions, 64, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        # deconv2: input = 64 (from deconv1) + 64 (skip from conv2) = 128  
        self.deconv2 = nn.ConvTranspose2d(64 + 64 + num_actions, 32, kernel_size=8, stride=2, padding=3)  # 16x16 -> 32x32
        # deconv3: input = 32 (from deconv2) + 32 (skip from conv1) = 64
        self.deconv3 = nn.ConvTranspose2d(32 + 32 + num_actions, output_channels, kernel_size=8, stride=3, padding=2)  # 32x32 -> 97x97
        
        self.activation = nn.Tanh()
        
    def _make_residual_block(self, feat_dim, num_actions):
        """Create a residual block with action conditioning"""
        class ResidualBlock(nn.Module):
            def __init__(self, feat_dim, num_actions):
                super().__init__()
                self.dense1 = nn.Linear(feat_dim + num_actions, feat_dim)
                self.dense2 = nn.Linear(feat_dim + num_actions, feat_dim)
                
            def forward(self, x, action_onehot):
                # First dense layer with action conditioning
                res = torch.cat([x, action_onehot], dim=1)
                res = F.leaky_relu(self.dense1(res))
                
                # Second dense layer with action conditioning  
                res = torch.cat([res, action_onehot], dim=1)
                res = self.dense2(res)
                
                return x + res
                
        return ResidualBlock(feat_dim, num_actions)
        
    def add_action_conditioning_1d(self, x, action_onehot):
        """Add action conditioning to 1D features"""
        return torch.cat([x, action_onehot], dim=1)
        
    def add_action_conditioning_2d(self, x, action_onehot):
        """Add action conditioning to 2D feature maps"""
        batch_size, channels, height, width = x.shape
        action_expanded = action_onehot.view(batch_size, self.num_actions, 1, 1)
        action_expanded = action_expanded.expand(batch_size, self.num_actions, height, width)
        return torch.cat([x, action_expanded], dim=1)
        
    def forward(self, x, action_onehot, skip_connections):
        """
        Args:
            x: Encoded features (batch_size, 64, 8, 8)
            action_onehot: One-hot encoded actions (batch_size, num_actions)
            skip_connections: List of skip connection features
        Returns:
            output: Decoded output (batch_size, output_channels, 84, 84)
        """
        # Flatten for bottleneck
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # Bottleneck with action conditioning
        x = self.add_action_conditioning_1d(x, action_onehot)
        x = F.leaky_relu(self.bottleneck(x))
        
        # Residual blocks with proper action conditioning
        for residual_block in self.residual_blocks:
            x = residual_block(x, action_onehot)
            
        # Decoder projection
        x = self.add_action_conditioning_1d(x, action_onehot)
        x = F.leaky_relu(self.decoder_proj(x))
        x = x.view(batch_size, 64, 8, 8)
        
        # Decoder with skip connections and action conditioning
        x = torch.cat([x, skip_connections[2]], dim=1)  # Add skip connection
        x = self.add_action_conditioning_2d(x, action_onehot)
        x = self.activation(self.deconv1(x))  # 8x8 -> 16x16
        
        x = torch.cat([x, skip_connections[1]], dim=1)  # Add skip connection
        x = self.add_action_conditioning_2d(x, action_onehot)
        x = self.activation(self.deconv2(x))  # 16x16 -> 32x32
        
        x = torch.cat([x, skip_connections[0]], dim=1)  # Add skip connection
        x = self.add_action_conditioning_2d(x, action_onehot)
        x = self.deconv3(x)  # 32x32 -> 96x96, no activation on final layer
        
        # Crop to original size (97x97 -> 84x84)
        x = x[:, :, 6:-7, 6:-7]
        
        return x

class AMAUNetPredictionNetwork(nn.Module):
    """
    AMA UNet Prediction Network for pixel-level predictions
    Implements the UNet dynamics model with dual heads for mean and variance
    """
    def __init__(self, input_shape, num_actions, feat_dim=512, uncertainty_penalty=1.0, 
                 reward_scaling=1.0, clip_val=1e6, lr=0.001):
        super(AMAUNetPredictionNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.feat_dim = feat_dim
        self.uncertainty_penalty = uncertainty_penalty
        self.reward_scaling = reward_scaling
        self.clip_val = clip_val
        
        channels, height, width = input_shape
        
        # Feature extractor (just pixel normalization)
        self.feature_extractor = JustPixelsFeatureExtractor(input_shape)
        
        # UNet encoder (shared)
        self.encoder = UNetEncoder(channels, num_actions)
        
        # UNet decoders (dual heads for mean and log variance)
        self.decoder_mu = UNetDecoder(feat_dim, num_actions, channels)
        self.decoder_log_sigma_sq = UNetDecoder(feat_dim, num_actions, channels)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        print(f"AMAUNetPredictionNetwork initialized - feat_dim: {feat_dim}")
        
    def forward(self, state, action):
        """
        Forward pass through UNet with dual heads
        Args:
            state: Current state (batch_size, channels, height, width)
            action: Action indices (batch_size,)
        Returns:
            mu: Mean prediction (batch_size, channels, height, width)
            log_sigma_sq: Log variance prediction (batch_size, channels, height, width)
        """
        # Extract features (just normalize pixels)
        features = self.feature_extractor(state)
        
        # Convert actions to one-hot
        action_onehot = F.one_hot(action, num_classes=self.num_actions).float()
        
        # Encode
        encoded, skip_connections = self.encoder(features, action_onehot)
        
        # Decode with dual heads
        mu = self.decoder_mu(encoded, action_onehot, skip_connections.copy())
        log_sigma_sq = self.decoder_log_sigma_sq(encoded, action_onehot, skip_connections.copy())
        
        return mu, log_sigma_sq
    
    def compute_loss(self, states, next_states, actions):
        """
        Compute AMA loss with uncertainty penalty
        Args:
            states: Current states (batch_size, channels, height, width)
            next_states: Next states (batch_size, channels, height, width)
            actions: Actions (batch_size,)
        Returns:
            loss: AMA loss tensor
            mse: Mean squared error
            uncertainty_loss: Uncertainty penalty term
        """
        # Normalize next states for comparison
        next_features = self.feature_extractor(next_states)
        
        # Forward pass
        mu, log_sigma_sq = self.forward(states, actions)
        
        mse = torch.square(mu - next_features.detach())
        
        # Bayesian loss: precision-weighted MSE + uncertainty penalty
        precision = torch.exp(-log_sigma_sq)
        data_loss = precision * mse
        uncertainty_loss = self.uncertainty_penalty * log_sigma_sq
        
        # Average over spatial dimensions
        loss = torch.mean(data_loss + uncertainty_loss, dim=[1, 2, 3])
        
        return loss.mean(), torch.mean(mse), torch.mean(uncertainty_loss)
    
    def compute_intrinsic_reward(self, states, next_states, actions):
        """
        Compute AMA intrinsic rewards based on uncertainty-error mismatch
        Args:
            states: Current states (batch_size, channels, height, width)
            next_states: Next states (batch_size, channels, height, width)  
            actions: Actions (batch_size,)
        Returns:
            rewards: Intrinsic reward values (batch_size,)
        """
        with torch.no_grad():
            # Normalize next states
            next_features = self.feature_extractor(next_states)
            
            # Forward pass
            mu, log_sigma_sq = self.forward(states, actions)
            
            # Compute prediction error and predicted variance
            mse = torch.square(mu - next_features.detach())
            predicted_variance = torch.exp(log_sigma_sq)
            
            # AMA reward: mismatch between prediction error and predicted uncertainty
            # Using non-absolute version (as abs_ama="false" in typical configs)
            uncertainty_error_mismatch = mse - predicted_variance
            
            # Average over spatial dimensions
            rewards = torch.mean(uncertainty_error_mismatch, dim=[1, 2, 3])
            
            # Apply reward scaling and clipping
            rewards = rewards * self.reward_scaling
            # rewards = torch.clamp(rewards, -self.clip_val, self.clip_val)
            
        return rewards

class AMAPix2PixCuriosity:
    """
    AMA (Aleatoric-epistemic Mismatch for Action) Curiosity with Pix2Pix features
    
    This implements the exact same algorithm as the original TensorFlow implementation,
    using raw normalized pixels as features and UNet dynamics model.
    """
    
    def __init__(self, obs_size, act_size, state_size, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 lr=1e-4, feat_dim=512, uncertainty_penalty=1.0, reward_scaling=1.0, 
                 clip_val=1e6, **kwargs):
        """
        Initialize the AMA Pix2Pix Curiosity model.
        
        Args:
            obs_size: Observation space shape (channels, height, width)
            act_size: Number of possible actions  
            state_size: State representation size (same as obs_size for pixels)
            device: Device to run computation on
            lr: Learning rate
            feat_dim: Feature dimension for UNet bottleneck
            uncertainty_penalty: Weight for uncertainty regularization
            reward_scaling: Scaling factor for intrinsic rewards
            clip_val: Clipping value for intrinsic rewards
        """
        print(f"AMAPix2PixCuriosity - obs_size: {obs_size}, act_size: {act_size}")
        
        self.device = device
        self.act_size = act_size
        
        # Store AMA hyperparameters
        self.feat_dim = feat_dim
        self.uncertainty_penalty = uncertainty_penalty
        self.reward_scaling = reward_scaling
        self.clip_val = clip_val
        
        # Only works with visual inputs
        if len(obs_size) != 3:
            raise ValueError("AMAPix2PixCuriosity only works with visual inputs (C, H, W)")
        
        self.is_visual = True
        self.input_shape = obs_size
        
        # Initialize UNet prediction network
        self.network = AMAUNetPredictionNetwork(
            self.input_shape, act_size, feat_dim=feat_dim,
            uncertainty_penalty=uncertainty_penalty, reward_scaling=reward_scaling,
            clip_val=clip_val, lr=lr
        ).to(device)
        
        print(f"AMAPix2PixCuriosity initialized with lr={lr}, feat_dim={feat_dim}")
        print(f"Hyperparameters - uncertainty_penalty: {uncertainty_penalty}, reward_scaling: {reward_scaling}")
        
    def _preprocess_observations(self, observations):
        """
        Convert observations to proper tensor format
        
        Args:
            observations: Input observations (numpy array or tensor)
        Returns:
            Preprocessed observations as torch tensor
        """
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations)
        
        if observations.dim() == 3:
            observations = observations.unsqueeze(0)
            
        observations = observations.to(self.device)
        
        # Ensure correct data type
        if observations.dtype != torch.float32:
            observations = observations.float()
            
        return observations
        
    def curiosity(self, observations, actions, next_observations):
        """
        Compute curiosity rewards for a batch of transitions.
        
        Args:
            observations: Current observations (batch_size, C, H, W) or (C, H, W)
            actions: Actions taken (batch_size,) or scalar
            next_observations: Next observations (batch_size, C, H, W) or (C, H, W)
        Returns:
            intrinsic_rewards: Numpy array of intrinsic rewards
        """
        self.network.eval()
        
        # Preprocess inputs
        obs = self._preprocess_observations(observations)
        next_obs = self._preprocess_observations(next_observations)
        
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        elif isinstance(actions, (int, float)):
            actions = torch.tensor([actions])
            
        actions = actions.to(self.device).long()
        
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
            
        # Compute intrinsic rewards
        rewards = self.network.compute_intrinsic_reward(obs, next_obs, actions)
        
        return rewards.cpu().numpy()
    
    def train(self, observations, actions, next_observations):
        """
        Update the AMA UNet model.
        
        Args:
            observations: Current observations
            actions: Actions taken  
            next_observations: Next observations
        Returns:
            loss_info: Tuple of (total_loss, mse_loss, uncertainty_loss)
        """
        self.network.train()
        
        # Preprocess inputs
        obs = self._preprocess_observations(observations)
        next_obs = self._preprocess_observations(next_observations)
        
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        elif isinstance(actions, (int, float)):
            actions = torch.tensor([actions])
            
        actions = actions.to(self.device).long()
        
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
            
        # Compute loss
        total_loss, mse_loss, uncertainty_loss = self.network.compute_loss(obs, next_obs, actions)
        
        # Backward pass
        self.network.optimizer.zero_grad()
        total_loss.backward()
        self.network.optimizer.step()
        
        return (total_loss.item(), uncertainty_loss.item())
    
    def update_normalization_stats(self, observations):
        """
        Update normalization statistics for feature extractor
        
        Args:
            observations: Batch of observations for computing statistics
        """
        obs = self._preprocess_observations(observations)
        self.network.feature_extractor.update_normalization_stats(obs)
    
    def save(self, filepath):
        """
        Save the AMA model
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.network.optimizer.state_dict(),
            'hyperparameters': {
                'input_shape': self.input_shape,
                'act_size': self.act_size,
                'feat_dim': self.feat_dim,
                'uncertainty_penalty': self.uncertainty_penalty,
                'reward_scaling': self.reward_scaling,
                'clip_val': self.clip_val
            }
        }, filepath)
        print(f"AMAPix2PixCuriosity model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the AMA model
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.feat_dim = hyperparams['feat_dim']
        self.uncertainty_penalty = hyperparams['uncertainty_penalty']
        self.reward_scaling = hyperparams['reward_scaling']
        self.clip_val = hyperparams['clip_val']
        
        print(f"AMAPix2PixCuriosity model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    print("Testing AMAPix2PixCuriosity...")
    
    # Example for Atari-style visual input
    obs_size = (4, 84, 84)  # Frame stack format (channels, height, width)
    act_size = 6            # Typical Atari action space
    state_size = obs_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize AMA curiosity model
    curiosity_model = AMAPix2PixCuriosity(
        obs_size, act_size, state_size, device, 
        lr=0.001, feat_dim=512, uncertainty_penalty=1.0, 
        reward_scaling=1.0, clip_val=1e6
    )
    
    # Test with random data - standardized (z-score normalized) as in real usage
    batch_size = 4
    # Generate random pixel values and apply z-score normalization
    obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
    next_obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
    
    # Apply z-score normalization (mean=0, std=1)
    obs = (obs_raw - obs_raw.mean()) / (obs_raw.std() + 1e-8)
    next_obs = (next_obs_raw - next_obs_raw.mean()) / (next_obs_raw.std() + 1e-8)
    actions = torch.randint(0, act_size, (batch_size,))
    
    print(f"Testing with batch_size={batch_size}")
    print(f"obs shape: {obs.shape}, obs mean: {obs.mean():.3f}, obs std: {obs.std():.3f}")
    print(f"obs range: [{obs.min():.3f}, {obs.max():.3f}], actions shape: {actions.shape}")
    
    # Test curiosity computation
    rewards = curiosity_model.curiosity(obs, actions, next_obs)
    print(f"Intrinsic rewards shape: {rewards.shape}, mean: {rewards.mean():.4f}")
    
    # Test training
    loss_info = curiosity_model.train(obs, actions, next_obs)
    print(f"Training losses - Total: {loss_info[0]:.4f}, Uncertainty: {loss_info[1]:.4f}")
    
    print("AMAPix2PixCuriosity implementation completed successfully!")