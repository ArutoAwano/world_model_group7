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
        
        print(f"CNNFeatureExtractor - input_shape: {input_shape}")
        
        # input_shape is (channels, height, width) - use directly
        channels, height, width = input_shape
        
        # Validate dimensions
        if height < 36 or width < 36:
            raise ValueError(f"Input dimensions {height}x{width} too small. Minimum size is 36x36 for this CNN architecture.")
        
        # Standard Atari CNN architecture
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate output size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        
        print(f"Conv output dimensions: {conv_height} x {conv_width}")
        
        if conv_width <= 0 or conv_height <= 0:
            raise ValueError(f"Convolution output dimensions are invalid: {conv_height}x{conv_width}")
        
        conv_output_size = conv_width * conv_height * 64
        print(f"Conv output size: {conv_output_size}")
        
        # Dense layer for fixed 512-dim output
        self.fc = nn.Linear(conv_output_size, 512)
        self.feature_size = 512

    def forward(self, x):
        # Input x is already normalized and in correct format (N, C, H, W)
        # No shape conversion needed
        
        # Apply convolutions with LeakyReLU
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Final dense layer
        x = F.leaky_relu(self.fc(x))
        
        return x

class MSEPredictionModel(nn.Module):
    """Primary network for predicting the next state using MSE loss"""
    def __init__(self, input_shape, num_actions, lr=0.001):
        super(MSEPredictionModel, self).__init__()

        self.input_shape = input_shape  # (channels, height, width)
        self.num_actions = num_actions

        # Feature extractor
        self.feature_extractor = CNNFeatureExtractor(input_shape)
        feature_size = self.feature_extractor.feature_size

        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(feature_size + num_actions, 512),
            nn.ReLU()
        )

        # Decoder to reconstruct next state
        channels, height, width = input_shape
        h_out = height // 8
        w_out = width // 8

        self.decoder = nn.Sequential(
            nn.Linear(512, h_out * w_out * 64),
            nn.ReLU(),
            nn.Unflatten(1, (64, h_out, w_out)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        """Forward pass to predict next state"""
        # state is already in correct format (N, C, H, W)
        state_features = self.feature_extractor(state)

        # One-hot encode action
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()

        # Combine features and action
        combined = torch.cat([state_features, action_one_hot], dim=1)
        features = self.forward_model(combined)

        # Predict next state
        predicted_next_state = self.decoder(features)

        return predicted_next_state

    def compute_error_batch(self, states, next_states, actions):
        """Compute prediction errors for a batch of transitions"""
        with torch.no_grad():
            # Get predictions
            predicted_next_states = self.forward(states, actions)

            # Convert targets to match prediction format (already normalized)
            next_states_normalized = next_states

            # Calculate squared error per sample
            squared_errors = torch.pow(next_states_normalized - predicted_next_states, 2)
            
            # Mean over all dimensions except batch
            mse_per_sample = squared_errors.reshape(squared_errors.size(0), -1).mean(dim=1)

        return mse_per_sample

class UncertaintyPredictionModel(nn.Module):
    """Network for predicting expected prediction errors"""
    def __init__(self, input_shape, num_actions, lr=0.01):
        super(UncertaintyPredictionModel, self).__init__()

        # Feature extractor
        self.feature_extractor = CNNFeatureExtractor(input_shape)
        feature_size = self.feature_extractor.feature_size
        self.num_actions = num_actions

        # Network to predict log of expected error
        self.uncertainty_net = nn.Sequential(
            nn.Linear(feature_size + num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        """Forward pass to predict log of expected prediction error"""
        # state is already in correct format (N, C, H, W)
        state_features = self.feature_extractor(state)

        # One-hot encode action
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()

        # Combine features and action
        combined = torch.cat([state_features, action_one_hot], dim=1)

        # Predict log uncertainty
        log_uncertainty = self.uncertainty_net(combined)

        # Clamp to avoid numerical issues
        log_uncertainty = torch.clamp(log_uncertainty, min=-10.0, max=10.0)

        return log_uncertainty

class LearningProgressCuriosity:
    """
    Learning Progress Curiosity class - NO SHAPE CONVERSION ANYWHERE
    """
    
    def __init__(self, obs_size, act_size, state_size, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 eta=1.0, pred_lr=0.001, uncertainty_lr=0.01, ismse=False):
        """
        Initialize the Learning Progress Curiosity model.
        
        Args:
            obs_size: Observation space shape (channels, height, width) - use directly
            act_size: Number of possible actions
            state_size: State size 
            device: Device to run computation on
            eta: Weighting factor for curiosity computation
            pred_lr: Learning rate for prediction model
            uncertainty_lr: Learning rate for uncertainty model
        """
        print(f"LearningProgressCuriosity - obs_size: {obs_size}, act_size: {act_size}")

        self.ismse = ismse
        self.device = device
        self.eta = eta
        self.act_size = act_size
        
        # Use obs_size directly - NO CONVERSION
        if len(obs_size) != 3:
            raise ValueError(f"Expected obs_size to have 3 dimensions, got {len(obs_size)}: {obs_size}")
        
        self.input_shape = obs_size  # Use directly: (4, 84, 84) stays (4, 84, 84)
        print(f"Using input_shape directly: {self.input_shape}")
        
        # Initialize models
        self.prediction_model = MSEPredictionModel(self.input_shape, act_size, lr=pred_lr).to(device)
        self.uncertainty_model = UncertaintyPredictionModel(self.input_shape, act_size, lr=uncertainty_lr).to(device)
        
        # Buffer for storing prediction errors
        self.buffer_size = 128 * 128
        self.error_buffer = deque(maxlen=self.buffer_size)
        
        print(f"LearningProgressCuriosity initialized successfully")
        
    def _preprocess_observations(self, observations):
        """Convert observations to proper tensor format"""
        # Convert to tensor if needed
        if not isinstance(observations, torch.Tensor):
            observations = torch.FloatTensor(observations)
        
        # Move to device
        observations = observations.to(self.device)
        
        # Normalize pixel values and ensure correct format
        # observations = observations / 255.0
        
        # observations should already be in (N, C, H, W) format
        # NO PERMUTATION NEEDED
        
        return observations
        
    def curiosity(self, observations, actions, next_observations):
        """
        Compute curiosity rewards for a batch of transitions.
        """
        # Preprocess inputs
        observations = self._preprocess_observations(observations)
        next_observations = self._preprocess_observations(next_observations)
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions)
        actions = actions.to(self.device)
        
        batch_size = observations.size(0)
        
        # Compute actual prediction errors
        actual_errors = self.prediction_model.compute_error_batch(observations, next_observations, actions)
        
        # Add errors to buffer
        for i in range(batch_size):
            self.error_buffer.append({
                'observation': observations[i].cpu().numpy(),
                'action': actions[i].cpu().item(),
                'error': actual_errors[i].cpu().item()
            })
        
        # Predict expected errors
        with torch.no_grad():
            log_expected_errors = self.uncertainty_model(observations, actions)
            expected_errors = torch.exp(log_expected_errors).squeeze()
            
            # Compute learning progress curiosity
            if self.ismse:
                # print("use MSE as reward")
                curiosity_rewards = actual_errors
            else:
                curiosity_rewards = self.eta * expected_errors - actual_errors
                # curiosity_rewards = torch.clamp(curiosity_rewards, max=0.5)
            
        return curiosity_rewards.cpu().numpy()
    
    def train(self, observations, actions, next_observations):
        """
        Update both prediction and uncertainty models.
        """
        # Preprocess inputs
        observations = self._preprocess_observations(observations)
        next_observations = self._preprocess_observations(next_observations)
        
        if not isinstance(actions, torch.Tensor):
            actions = torch.LongTensor(actions)
        actions = actions.to(self.device)
        # print("start training prediction model...")
        # Train prediction model for 3 epochs
        pred_loss_total = 0
        for epoch in range(3):
            # Forward pass
            predicted_next_states = self.prediction_model(observations, actions)
            
            # Target state (already normalized)
            target_next_states = next_observations
            
            # Compute MSE loss
            loss = F.mse_loss(predicted_next_states, target_next_states)
            
            # Backward pass
            self.prediction_model.optimizer.zero_grad()
            loss.backward()
            self.prediction_model.optimizer.step()
            
            pred_loss_total += loss.item()
        # print("start training uncertainty model...")
        # Train uncertainty model for 3 epochs
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
        """Save the curiosity models"""
        torch.save({
            'prediction_model_state_dict': self.prediction_model.state_dict(),
            'uncertainty_model_state_dict': self.uncertainty_model.state_dict(),
            'prediction_optimizer_state_dict': self.prediction_model.optimizer.state_dict(),
            'uncertainty_optimizer_state_dict': self.uncertainty_model.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load the curiosity models"""
        checkpoint = torch.load(filepath)
        self.prediction_model.load_state_dict(checkpoint['prediction_model_state_dict'])
        self.uncertainty_model.load_state_dict(checkpoint['uncertainty_model_state_dict'])
        self.prediction_model.optimizer.load_state_dict(checkpoint['prediction_optimizer_state_dict'])
        self.uncertainty_model.optimizer.load_state_dict(checkpoint['uncertainty_optimizer_state_dict'])

# Example usage:
if __name__ == "__main__":
    # Atari format: (channels, height, width) - NO CONVERSION NEEDED
    obs_size = (4, 84, 84)  
    act_size = 6
    state_size = obs_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize curiosity model
    curiosity_model = LearningProgressCuriosity(obs_size, act_size, state_size, device, 
                                               pred_lr=0.001, uncertainty_lr=0.01)
    
    # Example batch data - already in correct format
    batch_size = 16
    observations = torch.randint(0, 256, (batch_size, 4, 84, 84)).float()  # (N, C, H, W)
    actions = torch.randint(0, act_size, (batch_size,))
    next_observations = torch.randint(0, 256, (batch_size, 4, 84, 84)).float()  # (N, C, H, W)
    
    # Compute curiosity rewards
    curiosity_rewards = curiosity_model.curiosity(observations, actions, next_observations)
    print(f"Curiosity rewards shape: {curiosity_rewards.shape}")
    print(f"Sample curiosity rewards: {curiosity_rewards[:5]}")
    
    # Train the models
    pred_loss, uncertainty_loss = curiosity_model.train(observations, actions, next_observations)
    print(f"Prediction loss: {pred_loss:.4f}")
    print(f"Uncertainty loss: {uncertainty_loss:.4f}")