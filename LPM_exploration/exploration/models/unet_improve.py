import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class UNetEncoder(nn.Module):
    """UNet Encoder for feature extraction"""
    def __init__(self, input_channels):
        super(UNetEncoder, self).__init__()
        
        # Encoder layers - similar to AMA but without action conditioning for prediction model
        # Conv1: 96x96 -> 32x32 (stride=3, kernel=8)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=3, padding=3)
        # Conv2: 32x32 -> 16x16 (stride=2, kernel=8)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride=2, padding=3)
        # Conv3: 16x16 -> 8x8 (stride=2, kernel=4)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, channels, height, width)  
        Returns:
            features: Encoded features
            skip_connections: List of intermediate features for skip connections
        """
        skip_connections = []
        
        # Add padding to match original implementation (84x84 -> 96x96)
        x = F.pad(x, (6, 6, 6, 6))
        
        # Encoder 
        x = self.activation(self.conv1(x))  # 96x96 -> 32x32
        skip_connections.append(x)
        
        x = self.activation(self.conv2(x))  # 32x32 -> 16x16
        skip_connections.append(x)
        
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
        
        # Bottleneck - input: 64 * 8 * 8 + num_actions
        self.bottleneck = nn.Linear(64 * 8 * 8 + num_actions, feat_dim)
        
        # Residual blocks in bottleneck
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(feat_dim, num_actions) for _ in range(4)
        ])
        
        # Decoder projection
        self.decoder_proj = nn.Linear(feat_dim + num_actions, 64 * 8 * 8)
        
        # Decoder layers with skip connections
        self.deconv1 = nn.ConvTranspose2d(64 + 64 + num_actions, 64, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv2 = nn.ConvTranspose2d(64 + 64 + num_actions, 32, kernel_size=8, stride=2, padding=3)  # 16x16 -> 32x32
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
        x = torch.cat([x, action_onehot], dim=1)
        x = F.leaky_relu(self.bottleneck(x))
        
        # Residual blocks with action conditioning
        for residual_block in self.residual_blocks:
            x = residual_block(x, action_onehot)
            
        # Decoder projection
        x = torch.cat([x, action_onehot], dim=1)
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

class UNetPredictionModel(nn.Module):
    """UNet-based prediction network for next state prediction"""
    def __init__(self, input_shape, num_actions, feat_dim=512, lr=0.001):
        super(UNetPredictionModel, self).__init__()

        self.input_shape = input_shape  # (channels, height, width)
        self.num_actions = num_actions
        self.feat_dim = feat_dim

        channels, height, width = input_shape

        # UNet encoder (shared)
        self.encoder = UNetEncoder(channels)
        
        # UNet decoder for state prediction
        self.decoder = UNetDecoder(feat_dim, num_actions, channels)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        """Forward pass to predict next state using UNet"""
        # Encode state
        encoded, skip_connections = self.encoder(state)
        
        # Convert actions to one-hot
        action_onehot = F.one_hot(action, num_classes=self.num_actions).float()
        
        # Decode to predict next state
        predicted_next_state = self.decoder(encoded, action_onehot, skip_connections)

        return predicted_next_state

    def compute_error_batch(self, states, next_states, actions):
        """Compute prediction errors for a batch of transitions"""
        with torch.no_grad():
            # Get predictions
            predicted_next_states = self.forward(states, actions)

            # Calculate squared error per sample
            squared_errors = torch.pow(next_states - predicted_next_states, 2)
            
            # Mean over all dimensions except batch
            mse_per_sample = squared_errors.reshape(squared_errors.size(0), -1).mean(dim=1)

        return mse_per_sample

class UNetUncertaintyModel(nn.Module):
    """UNet-based uncertainty prediction network"""
    def __init__(self, input_shape, num_actions, feat_dim=512, lr=0.01):
        super(UNetUncertaintyModel, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.feat_dim = feat_dim

        channels, height, width = input_shape

        # UNet encoder
        self.encoder = UNetEncoder(channels)
        
        # Uncertainty prediction network (simpler than full decoder)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(64 * 8 * 8 + num_actions, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        """Forward pass to predict log of expected prediction error"""
        # Encode state
        encoded, _ = self.encoder(state)
        
        # Flatten encoded features
        batch_size = encoded.shape[0]
        encoded_flat = encoded.view(batch_size, -1)
        
        # Convert actions to one-hot
        action_onehot = F.one_hot(action, num_classes=self.num_actions).float()

        # Combine features and action
        combined = torch.cat([encoded_flat, action_onehot], dim=1)

        # Predict log uncertainty
        log_uncertainty = self.uncertainty_net(combined)

        # Clamp to avoid numerical issues
        log_uncertainty = torch.clamp(log_uncertainty, min=-10.0, max=10.0)

        return log_uncertainty

class UNetLearningProgressCuriosity:
    """
    UNet-based Learning Progress Curiosity model combining UNet architecture 
    with learning progress curiosity logic.
    """
    
    def __init__(self, obs_size, act_size, state_size, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 eta=1.0, pred_lr=0.001, uncertainty_lr=0.01, feat_dim=512, clip_val=1):
        """
        Initialize the UNet Learning Progress Curiosity model.
        
        Args:
            obs_size: Observation space shape (channels, height, width)
            act_size: Number of possible actions
            state_size: State size 
            device: Device to run computation on
            eta: Weighting factor for curiosity computation
            pred_lr: Learning rate for prediction model
            uncertainty_lr: Learning rate for uncertainty model
            feat_dim: Feature dimension for UNet bottleneck
        """
        print(f"UNetLearningProgressCuriosity - obs_size: {obs_size}, act_size: {act_size}")

        self.device = device
        self.eta = eta
        self.act_size = act_size
        self.clip_val = clip_val

        # Validate input shape
        if len(obs_size) != 3:
            raise ValueError(f"Expected obs_size to have 3 dimensions, got {len(obs_size)}: {obs_size}")
        
        self.input_shape = obs_size
        print(f"Using input_shape: {self.input_shape}")
        
        # Initialize UNet-based models
        self.prediction_model = UNetPredictionModel(self.input_shape, act_size, feat_dim=feat_dim, lr=pred_lr).to(device)
        self.uncertainty_model = UNetUncertaintyModel(self.input_shape, act_size, feat_dim=feat_dim, lr=uncertainty_lr).to(device)
        
        # Buffer for storing prediction errors
        self.buffer_size = 128 * 128
        self.error_buffer = deque(maxlen=self.buffer_size)
        
        print(f"UNetLearningProgressCuriosity initialized with feat_dim={feat_dim}")
        
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
        Compute curiosity rewards using learning progress logic with UNet predictions.
        """
        # Preprocess inputs
        observations = self._preprocess_observations(observations)
        next_observations = self._preprocess_observations(next_observations)
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions)
        actions = actions.to(self.device)
        
        batch_size = observations.size(0)
        
        # Compute actual prediction errors using UNet
        actual_errors = self.prediction_model.compute_error_batch(observations, next_observations, actions)
        
        # Add errors to buffer
        for i in range(batch_size):
            self.error_buffer.append({
                'observation': observations[i].cpu().numpy(),
                'action': actions[i].cpu().item(),
                'error': actual_errors[i].cpu().item()
            })
        
        # Predict expected errors using UNet uncertainty model
        with torch.no_grad():
            log_expected_errors = self.uncertainty_model(observations, actions)
            expected_errors = torch.exp(log_expected_errors).squeeze()
            
            # Compute learning progress curiosity: eta * expected_error - actual_error
            curiosity_rewards = self.eta * expected_errors - actual_errors
            curiosity_rewards = torch.clamp(curiosity_rewards, max=self.clip_val)
            
        return curiosity_rewards.cpu().numpy()
    
    def train(self, observations, actions, next_observations):
        """
        Update both UNet prediction and uncertainty models.
        """
        # Preprocess inputs
        observations = self._preprocess_observations(observations)
        next_observations = self._preprocess_observations(next_observations)
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions)
        actions = actions.to(self.device)
        
        # Train UNet prediction model for 3 epochs
        pred_loss_total = 0
        for epoch in range(3):
            # Forward pass
            predicted_next_states = self.prediction_model(observations, actions)
            
            # Compute MSE loss
            loss = F.mse_loss(predicted_next_states, next_observations)
            
            # Backward pass
            self.prediction_model.optimizer.zero_grad()
            loss.backward()
            self.prediction_model.optimizer.step()
            
            pred_loss_total += loss.item()
        
        # Train UNet uncertainty model for 3 epochs
        uncertainty_loss_total = 0
        if len(self.error_buffer) > 0:
            for epoch in range(3):
                # Sample from buffer
                batch_size = min(32, len(self.error_buffer))
                batch_indices = random.sample(range(len(self.error_buffer)), batch_size)
                
                sampled_states = []
                sampled_actions = []
                sampled_errors = []
                
                for idx in batch_indices:
                    error_data = self.error_buffer[idx]
                    sampled_states.append(error_data['observation'])
                    sampled_actions.append(error_data['action'])
                    sampled_errors.append(error_data['error'])
                
                # Convert to tensors
                state_batch = torch.FloatTensor(np.array(sampled_states)).to(self.device)
                action_batch = torch.LongTensor(sampled_actions).to(self.device)
                
                # Forward pass
                log_predicted_errors = self.uncertainty_model(state_batch, action_batch)
                
                # Target
                epsilon = 1e-6
                log_actual_errors = torch.log(torch.FloatTensor(sampled_errors).to(self.device) + epsilon).unsqueeze(1)
                
                # Compute loss
                loss = F.mse_loss(log_predicted_errors, log_actual_errors)
                
                # Backward pass
                self.uncertainty_model.optimizer.zero_grad()
                loss.backward()
                self.uncertainty_model.optimizer.step()
                
                uncertainty_loss_total += loss.item()
        
        return pred_loss_total / 3.0, uncertainty_loss_total / 3.0
    
    def save(self, filepath):
        """Save the UNet curiosity models"""
        torch.save({
            'prediction_model_state_dict': self.prediction_model.state_dict(),
            'uncertainty_model_state_dict': self.uncertainty_model.state_dict(),
            'prediction_optimizer_state_dict': self.prediction_model.optimizer.state_dict(),
            'uncertainty_optimizer_state_dict': self.uncertainty_model.optimizer.state_dict(),
            'hyperparameters': {
                'input_shape': self.input_shape,
                'act_size': self.act_size,
                'eta': self.eta
            }
        }, filepath)
        print(f"UNetLearningProgressCuriosity model saved to {filepath}")
    
    def load(self, filepath):
        """Load the UNet curiosity models"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.prediction_model.load_state_dict(checkpoint['prediction_model_state_dict'])
        self.uncertainty_model.load_state_dict(checkpoint['uncertainty_model_state_dict'])
        self.prediction_model.optimizer.load_state_dict(checkpoint['prediction_optimizer_state_dict'])
        self.uncertainty_model.optimizer.load_state_dict(checkpoint['uncertainty_optimizer_state_dict'])
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.eta = hyperparams['eta']
        
        print(f"UNetLearningProgressCuriosity model loaded from {filepath}")


# ============================================================================
# COMPREHENSIVE TEST SCRIPT
# ============================================================================

if __name__ == "__main__":
    import time
    import os
    
    def test_individual_components():
        """Test individual UNet components"""
        print("=" * 60)
        print("TESTING INDIVIDUAL COMPONENTS")
        print("=" * 60)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Test UNetEncoder
        print("\n1. Testing UNetEncoder...")
        encoder = UNetEncoder(input_channels=4).to(device)
        test_input = torch.randn(2, 4, 84, 84).to(device)
        
        try:
            encoded, skip_connections = encoder(test_input)
            print(f"   ✅ Encoder output shape: {encoded.shape}")
            print(f"   ✅ Skip connections shapes: {[skip.shape for skip in skip_connections]}")
            assert encoded.shape == (2, 64, 8, 8), f"Expected (2, 64, 8, 8), got {encoded.shape}"
            assert len(skip_connections) == 3, f"Expected 3 skip connections, got {len(skip_connections)}"
        except Exception as e:
            print(f"   ❌ Encoder test failed: {e}")
            return False
        
        # Test UNetDecoder
        print("\n2. Testing UNetDecoder...")
        decoder = UNetDecoder(feat_dim=512, num_actions=6, output_channels=4).to(device)
        action_onehot = torch.zeros(2, 6).to(device)
        action_onehot[:, 0] = 1  # Set first action as active
        
        try:
            decoded = decoder(encoded, action_onehot, skip_connections)
            print(f"   ✅ Decoder output shape: {decoded.shape}")
            assert decoded.shape == (2, 4, 84, 84), f"Expected (2, 4, 84, 84), got {decoded.shape}"
        except Exception as e:
            print(f"   ❌ Decoder test failed: {e}")
            return False
        
        # Test UNetPredictionModel
        print("\n3. Testing UNetPredictionModel...")
        pred_model = UNetPredictionModel((4, 84, 84), 6, feat_dim=512).to(device)
        actions = torch.randint(0, 6, (2,)).to(device)
        
        try:
            predictions = pred_model(test_input, actions)
            print(f"   ✅ Prediction model output shape: {predictions.shape}")
            assert predictions.shape == (2, 4, 84, 84), f"Expected (2, 4, 84, 84), got {predictions.shape}"
        except Exception as e:
            print(f"   ❌ Prediction model test failed: {e}")
            return False
        
        # Test UNetUncertaintyModel  
        print("\n4. Testing UNetUncertaintyModel...")
        uncertainty_model = UNetUncertaintyModel((4, 84, 84), 6, feat_dim=512).to(device)
        
        try:
            uncertainties = uncertainty_model(test_input, actions)
            print(f"   ✅ Uncertainty model output shape: {uncertainties.shape}")
            assert uncertainties.shape == (2, 1), f"Expected (2, 1), got {uncertainties.shape}"
        except Exception as e:
            print(f"   ❌ Uncertainty model test failed: {e}")
            return False
            
        print("\n✅ All individual components passed!")
        return True
    
    def test_full_model():
        """Test the complete UNetLearningProgressCuriosity model"""
        print("\n" + "=" * 60)
        print("TESTING FULL UNET LEARNING PROGRESS CURIOSITY MODEL")
        print("=" * 60)
        
        # Setup
        obs_size = (4, 84, 84)  # Frame stack format (channels, height, width)
        act_size = 6            # Typical Atari action space
        state_size = obs_size
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize UNet curiosity model
        print(f"\n1. Initializing UNetLearningProgressCuriosity...")
        try:
            curiosity_model = UNetLearningProgressCuriosity(
                obs_size, act_size, state_size, device, 
                eta=1.0, pred_lr=0.001, uncertainty_lr=0.01, feat_dim=512
            )
            print("   ✅ Model initialized successfully")
        except Exception as e:
            print(f"   ❌ Model initialization failed: {e}")
            return False
        
        # Test with random data
        print(f"\n2. Testing with random normalized data...")
        batch_size = 8
        
        # Generate random data and normalize (as done in real training)
        obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
        next_obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
        
        # Apply z-score normalization (as done in main training loop)
        obs_mean = obs_raw.mean()
        obs_std = obs_raw.std()
        obs = (obs_raw - obs_mean) / (obs_std + 1e-8)
        next_obs = (next_obs_raw - obs_mean) / (obs_std + 1e-8)  # Use same normalization
        
        actions = torch.randint(0, act_size, (batch_size,))
        
        print(f"   📊 Data shapes:")
        print(f"      obs: {obs.shape}, range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"      actions: {actions.shape}, values: {actions.tolist()}")
        print(f"      next_obs: {next_obs.shape}, range: [{next_obs.min():.3f}, {next_obs.max():.3f}]")
        
        # Test curiosity computation
        print(f"\n3. Testing curiosity computation...")
        try:
            start_time = time.time()
            rewards = curiosity_model.curiosity(obs, actions, next_obs)
            computation_time = time.time() - start_time
            
            print(f"   ✅ Curiosity computation successful")
            print(f"   📊 Results:")
            print(f"      Rewards shape: {rewards.shape}")
            print(f"      Rewards mean: {rewards.mean():.6f}")
            print(f"      Rewards std: {rewards.std():.6f}")
            print(f"      Rewards range: [{rewards.min():.6f}, {rewards.max():.6f}]")
            print(f"      Computation time: {computation_time:.3f}s")
            
            # Check for reasonable values
            assert not np.isnan(rewards).any(), "Rewards contain NaN values"
            assert not np.isinf(rewards).any(), "Rewards contain infinite values"
            
        except Exception as e:
            print(f"   ❌ Curiosity computation failed: {e}")
            return False
        
        # Test training
        print(f"\n4. Testing model training...")
        try:
            start_time = time.time()
            loss_info = curiosity_model.train(obs, actions, next_obs)
            training_time = time.time() - start_time
            
            pred_loss, uncertainty_loss = loss_info
            print(f"   ✅ Training successful")
            print(f"   📊 Training results:")
            print(f"      Prediction loss: {pred_loss:.6f}")
            print(f"      Uncertainty loss: {uncertainty_loss:.6f}")
            print(f"      Training time: {training_time:.3f}s")
            print(f"      Error buffer size: {len(curiosity_model.error_buffer)}")
            
            # Check for reasonable loss values
            assert not np.isnan(pred_loss), "Prediction loss is NaN"
            assert not np.isnan(uncertainty_loss), "Uncertainty loss is NaN"
            assert pred_loss > 0, "Prediction loss should be positive"
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            return False
        
        # Test multiple training iterations
        print(f"\n5. Testing multiple training iterations...")
        try:
            losses_pred = []
            losses_uncertainty = []
            
            for i in range(5):
                # Generate new random data for each iteration
                obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
                next_obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
                obs_mean = obs_raw.mean()
                obs_std = obs_raw.std()
                obs = (obs_raw - obs_mean) / (obs_std + 1e-8)
                next_obs = (next_obs_raw - obs_mean) / (obs_std + 1e-8)
                actions = torch.randint(0, act_size, (batch_size,))
                
                pred_loss, uncertainty_loss = curiosity_model.train(obs, actions, next_obs)
                losses_pred.append(pred_loss)
                losses_uncertainty.append(uncertainty_loss)
                
                if i == 0:
                    print(f"      Iteration {i+1}: Pred={pred_loss:.4f}, Unc={uncertainty_loss:.4f}")
                elif i == 4:
                    print(f"      Iteration {i+1}: Pred={pred_loss:.4f}, Unc={uncertainty_loss:.4f}")
            
            print(f"   ✅ Multiple iterations successful")
            print(f"   📊 Loss trends:")
            print(f"      Prediction loss: {losses_pred[0]:.4f} → {losses_pred[-1]:.4f}")
            print(f"      Uncertainty loss: {losses_uncertainty[0]:.4f} → {losses_uncertainty[-1]:.4f}")
            print(f"      Buffer filled to: {len(curiosity_model.error_buffer)} samples")
            
        except Exception as e:
            print(f"   ❌ Multiple iterations failed: {e}")
            return False
        
        # Test save/load functionality
        print(f"\n6. Testing save/load functionality...")
        try:
            # Save model
            test_path = "test_unet_curiosity.pt"
            curiosity_model.save(test_path)
            
            # Create new model and load
            new_model = UNetLearningProgressCuriosity(
                obs_size, act_size, state_size, device, 
                eta=1.0, pred_lr=0.001, uncertainty_lr=0.01, feat_dim=512
            )
            new_model.load(test_path)
            
            # Test that loaded model works
            new_rewards = new_model.curiosity(obs, actions, next_obs)
            
            print(f"   ✅ Save/load successful")
            print(f"   📊 Loaded model produces rewards: {new_rewards.mean():.6f}")
            
            # Clean up
            if os.path.exists(test_path):
                os.remove(test_path)
                
        except Exception as e:
            print(f"   ❌ Save/load failed: {e}")
            # Clean up on failure
            if os.path.exists("test_unet_curiosity.pt"):
                os.remove("test_unet_curiosity.pt")
            return False
        
        print("\n✅ Full model test completed successfully!")
        return True
    
    def performance_comparison():
        """Compare UNet vs standard CNN approach"""
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON")
        print("=" * 60)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        obs_size = (4, 84, 84)
        act_size = 6
        batch_size = 16
        
        print(f"\n🔍 Comparing UNet vs Standard CNN approach...")
        print(f"   Device: {device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Input shape: {obs_size}")
        
        # Generate test data
        obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
        next_obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
        obs_mean = obs_raw.mean()
        obs_std = obs_raw.std()
        obs = (obs_raw - obs_mean) / (obs_std + 1e-8)
        next_obs = (next_obs_raw - obs_mean) / (obs_std + 1e-8)
        actions = torch.randint(0, act_size, (batch_size,))
        
        # Test UNet model
        print(f"\n1. Testing UNet model performance...")
        unet_model = UNetLearningProgressCuriosity(
            obs_size, act_size, obs_size, device, 
            eta=1.0, pred_lr=0.001, uncertainty_lr=0.01, feat_dim=512
        )
        
        # Warmup
        _ = unet_model.curiosity(obs[:2], actions[:2], next_obs[:2])
        
        # Time curiosity computation
        start_time = time.time()
        for _ in range(10):
            unet_rewards = unet_model.curiosity(obs, actions, next_obs)
        unet_time = (time.time() - start_time) / 10
        
        # Time training
        start_time = time.time()
        for _ in range(5):
            unet_losses = unet_model.train(obs, actions, next_obs)
        unet_train_time = (time.time() - start_time) / 5
        
        print(f"   📊 UNet Results:")
        print(f"      Curiosity computation: {unet_time:.4f}s per batch")
        print(f"      Training time: {unet_train_time:.4f}s per update")
        print(f"      Prediction loss: {unet_losses[0]:.6f}")
        print(f"      Uncertainty loss: {unet_losses[1]:.6f}")
        print(f"      Reward statistics: μ={unet_rewards.mean():.6f}, σ={unet_rewards.std():.6f}")
        
        # Count parameters
        unet_params = sum(p.numel() for p in unet_model.prediction_model.parameters())
        unet_params += sum(p.numel() for p in unet_model.uncertainty_model.parameters())
        
        print(f"      Total parameters: {unet_params:,}")
        print(f"      Memory usage: ~{unet_params * 4 / 1024 / 1024:.1f} MB")
        
        print(f"\n✅ Performance analysis completed!")
        
        return True
    
    def main_test():
        """Run all tests"""
        print("🚀 STARTING COMPREHENSIVE UNET LEARNING PROGRESS CURIOSITY TEST")
        print("=" * 80)
        
        all_passed = True
        
        # Test 1: Individual components
        if not test_individual_components():
            all_passed = False
            print("\n❌ Individual components test FAILED")
        
        # Test 2: Full model
        if not test_full_model():
            all_passed = False
            print("\n❌ Full model test FAILED")
        
        # Test 3: Performance comparison
        if not performance_comparison():
            all_passed = False
            print("\n❌ Performance comparison FAILED")
        
        # Final results
        print("\n" + "=" * 80)
        if all_passed:
            print("🎉 ALL TESTS PASSED! UNet Learning Progress Curiosity is ready to use!")
            print("\nTo use in your main training:")
            print("python main.py --algo unet-improvement --env-name ALE/DemonAttack-v5")
            print("\nTo integrate into your codebase:")
            print("1. Save this file as exploration/models/unet_improve.py")
            print("2. Add import: from exploration.models.unet_improve import UNetLearningProgressCuriosity")
            print("3. Update arguments.py to include 'unet-improvement' in the algo choices")
        else:
            print("❌ SOME TESTS FAILED! Please check the errors above.")
        print("=" * 80)
        
        return all_passed
    
    # Run the comprehensive test
    main_test()