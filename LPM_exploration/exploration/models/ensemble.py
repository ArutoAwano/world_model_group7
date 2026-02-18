import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class RandomCNNFeatureExtractor(nn.Module):
    """Random (frozen) CNN feature extractor as used in the ensemble disagreement paper"""
    def __init__(self, input_shape):
        super(RandomCNNFeatureExtractor, self).__init__()
        
        print(f"RandomCNNFeatureExtractor - input_shape: {input_shape}")
        
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
        
        print(f"Ensemble Conv output dimensions: {conv_height} x {conv_width}")
        
        if conv_width <= 0 or conv_height <= 0:
            raise ValueError(f"Convolution output dimensions are invalid: {conv_height}x{conv_width}")
        
        conv_output_size = conv_width * conv_height * 64
        print(f"Ensemble Conv output size: {conv_output_size}")
        
        # Dense layer for fixed 512-dim output (comparable to other methods)
        self.fc = nn.Linear(conv_output_size, 512)
        self.feature_size = 512
        
        # Initialize with random weights and freeze
        self._initialize_random_weights()
        self._freeze_parameters()

    def _initialize_random_weights(self):
        """Initialize with random weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def _freeze_parameters(self):
        """Freeze all parameters (random features should not be trained)"""
        for param in self.parameters():
            param.requires_grad = False

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

class EnsembleForwardModel(nn.Module):
    """Single forward model in the ensemble that predicts next state features"""
    def __init__(self, feature_size, num_actions):
        super(EnsembleForwardModel, self).__init__()
        
        self.feature_size = feature_size
        self.num_actions = num_actions
        
        # Forward model: state features + action -> predicted next state features
        self.forward_net = nn.Sequential(
            nn.Linear(feature_size + num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, feature_size)  # Predict next state features
        )
        
    def forward(self, state_features, action_onehot):
        """
        Predict next state features given current state features and action
        
        Args:
            state_features: Current state features (batch_size, feature_size)
            action_onehot: One-hot encoded actions (batch_size, num_actions)
        Returns:
            predicted_next_features: Predicted next state features (batch_size, feature_size)
        """
        # Concatenate state features and action
        combined_input = torch.cat([state_features, action_onehot], dim=1)
        
        # Predict next state features
        predicted_next_features = self.forward_net(combined_input)
        
        return predicted_next_features

class EnsembleDisagreementNetwork(nn.Module):
    """
    Ensemble Disagreement Network for curiosity-driven exploration
    
    Based on "Self-Supervised Exploration via Disagreement" (Pathak et al.)
    Uses an ensemble of forward models and disagreement (variance) as intrinsic reward.
    """
    
    def __init__(self, input_shape, num_actions, eta=1.0, num_ensemble=5, lr=0.001):
        super(EnsembleDisagreementNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.eta = eta  # Intrinsic reward scaling
        self.num_ensemble = num_ensemble
        
        # Random (frozen) feature extractor
        self.feature_extractor = RandomCNNFeatureExtractor(input_shape)
        feature_size = self.feature_extractor.feature_size
        
        # Ensemble of forward models
        self.forward_models = nn.ModuleList()
        self.optimizers = []  # Separate optimizer for each model
        
        for i in range(num_ensemble):
            forward_model = EnsembleForwardModel(feature_size, num_actions)
            self.forward_models.append(forward_model)
            
            # Each model has its own optimizer for bootstrap training
            optimizer = optim.Adam(forward_model.parameters(), lr=lr)
            self.optimizers.append(optimizer)
        
        print(f"EnsembleDisagreementNetwork initialized:")
        print(f"  Number of ensemble models: {num_ensemble}")
        print(f"  Feature size: {feature_size}")
        print(f"  Frozen feature extractor parameters: {sum(p.numel() for p in self.feature_extractor.parameters()):,}")
        
        total_forward_params = sum(sum(p.numel() for p in model.parameters()) for model in self.forward_models)
        print(f"  Total forward model parameters: {total_forward_params:,}")
        print(f"  Total parameters: {total_forward_params:,} (feature extractor frozen)")
        
    def encode_states(self, states):
        """Extract features from states using frozen random CNN"""
        with torch.no_grad():  # Feature extractor is frozen
            return self.feature_extractor(states)
    
    def compute_intrinsic_reward(self, states, next_states, actions):
        """
        Compute intrinsic rewards based on ensemble disagreement (variance)
        
        Args:
            states: Current states (batch_size, channels, height, width)
            next_states: Next states (batch_size, channels, height, width)  
            actions: Actions taken (batch_size,)
        Returns:
            intrinsic_rewards: Disagreement-based rewards (batch_size,)
        """
        with torch.no_grad():
            # Extract features from current states
            state_features = self.encode_states(states)
            
            # Convert actions to one-hot encoding
            action_onehot = F.one_hot(actions, num_classes=self.num_actions).float()
            
            # Get predictions from all ensemble models
            predictions = []
            for forward_model in self.forward_models:
                predicted_features = forward_model(state_features, action_onehot)
                predictions.append(predicted_features)
            
            # Stack predictions: (num_ensemble, batch_size, feature_dim)
            predictions = torch.stack(predictions, dim=0)
            
            # Calculate variance across ensemble predictions (Equation 1 from paper)
            prediction_variance = torch.var(predictions, dim=0)
            
            # Mean variance across all feature dimensions
            intrinsic_rewards = self.eta * prediction_variance.mean(dim=1)
            
        return intrinsic_rewards
    
    def compute_loss(self, states, next_states, actions):
        """
        Update ensemble models using bootstrap sampling
        
        Args:
            states: Current states (batch_size, channels, height, width)
            next_states: Next states (batch_size, channels, height, width)
            actions: Actions taken (batch_size,)
        Returns:
            total_loss: Average loss across all ensemble models
        """
        batch_size = states.shape[0]
        
        # Extract features for current and next states
        state_features = self.encode_states(states)
        next_state_features = self.encode_states(next_states)
        
        # Convert actions to one-hot encoding
        action_onehot = F.one_hot(actions, num_classes=self.num_actions).float()
        
        total_loss = 0.0
        
        # Update each model in the ensemble with bootstrap sampling
        for i, (forward_model, optimizer) in enumerate(zip(self.forward_models, self.optimizers)):
            # Bootstrap sampling: sample with replacement to create diversity
            bootstrap_indices = torch.randint(0, batch_size, (batch_size,), device=states.device)
            
            # Get bootstrap samples
            bootstrap_state_features = state_features[bootstrap_indices]
            bootstrap_action_onehot = action_onehot[bootstrap_indices]
            bootstrap_target_features = next_state_features[bootstrap_indices]
            
            # Forward pass for this model
            predicted_features = forward_model(bootstrap_state_features, bootstrap_action_onehot)
            
            # Calculate loss (MSE in feature space)
            loss = F.mse_loss(predicted_features, bootstrap_target_features)
            
            # Update this specific model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.num_ensemble

class EnsembleDisagreementCuriosity:
    """
    Ensemble Disagreement Curiosity method
    
    Implementation of "Self-Supervised Exploration via Disagreement" (Pathak et al.)
    Trains an ensemble of forward dynamics models and uses their disagreement
    (variance) as the intrinsic reward signal for exploration.
    """
    
    def __init__(self, obs_size, act_size, state_size, device='cuda' if torch.cuda.is_available() else 'cpu',
                 eta=1.0, num_ensemble=5, lr=0.001, **kwargs):
        """
        Initialize the Ensemble Disagreement Curiosity model.
        
        Args:
            obs_size: Observation space shape (channels, height, width)
            act_size: Number of possible actions
            state_size: State size (same as obs_size for visual inputs)
            device: Device to run computation on
            eta: Scaling factor for intrinsic rewards
            num_ensemble: Number of forward models in ensemble
            lr: Learning rate for forward models
        """
        print(f"EnsembleDisagreementCuriosity - obs_size: {obs_size}, act_size: {act_size}")
        
        self.device = device
        self.eta = eta
        self.num_ensemble = num_ensemble
        self.act_size = act_size
        
        # Validate input shape
        if len(obs_size) != 3:
            raise ValueError(f"Expected obs_size to have 3 dimensions, got {len(obs_size)}: {obs_size}")
        
        self.input_shape = obs_size
        print(f"Using input_shape: {self.input_shape}")
        
        # Initialize ensemble disagreement network
        self.network = EnsembleDisagreementNetwork(
            self.input_shape, act_size, eta, num_ensemble, lr
        ).to(device)
        
        print(f"EnsembleDisagreementCuriosity initialized with eta={eta}, num_ensemble={num_ensemble}, lr={lr}")
        
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
            print("WARNING: Ensemble received invalid observations (NaN/Inf)")
            return np.zeros(obs.shape[0])
        
        if torch.isnan(next_obs).any() or torch.isinf(next_obs).any():
            print("WARNING: Ensemble received invalid next_observations (NaN/Inf)")
            return np.zeros(next_obs.shape[0])
        
        # Compute intrinsic rewards based on ensemble disagreement
        intrinsic_rewards = self.network.compute_intrinsic_reward(obs, next_obs, actions)
        
        # Debug: Check reward validity
        if torch.isnan(intrinsic_rewards).any() or torch.isinf(intrinsic_rewards).any():
            print("WARNING: Ensemble computed invalid rewards (NaN/Inf)")
            return np.zeros(intrinsic_rewards.shape[0])
        
        # Debug: Print raw reward statistics occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 1000 == 0:  # Print every 1000 calls
            print(f"Ensemble Debug - Raw rewards: mean={intrinsic_rewards.mean().item():.6f}, "
                  f"std={intrinsic_rewards.std().item():.6f}, "
                  f"min={intrinsic_rewards.min().item():.6f}, "
                  f"max={intrinsic_rewards.max().item():.6f}")
        
        return intrinsic_rewards.cpu().numpy()
    
    def train(self, observations, actions, next_observations):
        """
        Update the ensemble models using bootstrap sampling.
        
        Args:
            observations: Current observations
            actions: Actions taken
            next_observations: Next observations
        Returns:
            loss_info: Tuple of (ensemble_loss, 0) for compatibility with other methods
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
        
        # Compute ensemble loss (bootstrap training happens inside)
        ensemble_loss = self.network.compute_loss(obs, next_obs, actions)
        
        # Return loss info (second value set to 0 for compatibility)
        return (ensemble_loss, 0.0)
    
    def save(self, filepath):
        """
        Save the ensemble disagreement model
        
        Args:
            filepath: Path to save the model
        """
        # Save each forward model and optimizer separately
        model_states = []
        optimizer_states = []
        
        for i, (model, optimizer) in enumerate(zip(self.network.forward_models, self.network.optimizers)):
            model_states.append(model.state_dict())
            optimizer_states.append(optimizer.state_dict())
        
        torch.save({
            'feature_extractor_state_dict': self.network.feature_extractor.state_dict(),
            'forward_model_states': model_states,
            'optimizer_states': optimizer_states,
            'hyperparameters': {
                'input_shape': self.input_shape,
                'act_size': self.act_size,
                'eta': self.eta,
                'num_ensemble': self.num_ensemble
            }
        }, filepath)
        print(f"EnsembleDisagreementCuriosity model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the ensemble disagreement model
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load feature extractor (though it's frozen)
        self.network.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        
        # Load each forward model and optimizer
        for i, (model, optimizer) in enumerate(zip(self.network.forward_models, self.network.optimizers)):
            model.load_state_dict(checkpoint['forward_model_states'][i])
            optimizer.load_state_dict(checkpoint['optimizer_states'][i])
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.eta = hyperparams['eta']
        self.num_ensemble = hyperparams['num_ensemble']
        
        print(f"EnsembleDisagreementCuriosity model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    print("Testing EnsembleDisagreementCuriosity...")
    
    # Example for Atari-style visual input
    obs_size = (4, 84, 84)  # Frame stack format (channels, height, width)
    act_size = 6            # Typical Atari action space
    state_size = obs_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize Ensemble Disagreement curiosity model
    curiosity_model = EnsembleDisagreementCuriosity(
        obs_size, act_size, state_size, device, 
        eta=1.0, num_ensemble=5, lr=0.001
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
    print(f"Training losses - Ensemble: {loss_info[0]:.4f}, Placeholder: {loss_info[1]:.4f}")
    
    # Test multiple training iterations to see learning
    print(f"\nTesting learning over multiple iterations...")
    initial_rewards = rewards.copy()
    
    for i in range(10):
        loss_info = curiosity_model.train(obs, actions, next_obs)
        if i % 3 == 0:
            print(f"  Iteration {i+1}: Ensemble Loss = {loss_info[0]:.4f}")
    
    # Check if rewards change after training (they should change as models learn)
    final_rewards = curiosity_model.curiosity(obs, actions, next_obs)
    print(f"\nReward change after training:")
    print(f"  Initial rewards mean: {initial_rewards.mean():.4f}")
    print(f"  Final rewards mean: {final_rewards.mean():.4f}")
    print(f"  Change: {final_rewards.mean() - initial_rewards.mean():.4f}")
    
    print("EnsembleDisagreementCuriosity test completed successfully!")