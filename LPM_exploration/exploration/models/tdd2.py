import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import random

class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for visual observations - matches original architecture"""
    def __init__(self, input_shape, features_dim=256):
        super(CNNFeatureExtractor, self).__init__()
        
        print(f"TDD CNNFeatureExtractor - input_shape: {input_shape}")
        
        channels, height, width = input_shape
        
        # Validate dimensions
        if height < 36 or width < 36:
            raise ValueError(f"Input dimensions {height}x{width} too small. Minimum size is 36x36 for this CNN architecture.")
        
        # Standard Atari CNN architecture - matches NatureCNN from stable_baselines3
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate output size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        
        print(f"TDD Conv output dimensions: {conv_height} x {conv_width}")
        
        if conv_width <= 0 or conv_height <= 0:
            raise ValueError(f"Convolution output dimensions are invalid: {conv_height}x{conv_width}")
        
        conv_output_size = conv_width * conv_height * 64
        print(f"TDD Conv output size: {conv_output_size}")
        
        # Dense layer for feature extraction to fixed dimension
        self.fc = nn.Linear(conv_output_size, features_dim)
        self.features_dim = features_dim

    def forward(self, x):
        # Apply convolutions with ReLU (matching original)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Final dense layer
        x = F.relu(self.fc(x))
        
        return x

def mrn_distance(x, y):
    """Metric Residual Network (MRN) distance function - exact match to original"""
    eps = 1e-8
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    max_component = torch.max(F.relu(x_prefix - y_prefix), dim=-1).values
    l2_component = torch.sqrt(torch.square(x_suffix - y_suffix).sum(dim=-1) + eps)
    return max_component + l2_component

class TDDNetwork(nn.Module):
    """
    Temporal Distance Density Network - matches original paper architecture
    """
    
    def __init__(self, input_shape, num_actions, features_dim=256, embedding_dim=64, lr=0.001, 
                 energy_fn='mrn_pot', loss_fn='infonce', aggregate_fn='min',
                 temperature=1.0, logsumexp_coef=0.1, knn_k=10, max_history_size=1000,
                 persistent_memory=True):
        super(TDDNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.features_dim = features_dim
        self.embedding_dim = embedding_dim
        self.energy_fn = energy_fn  # 'l2', 'cos', 'mrn', 'mrn_pot'
        self.loss_fn = loss_fn  # 'infonce', 'infonce_symmetric', 'infonce_backward'
        self.aggregate_fn = aggregate_fn  # 'min', 'quantile10', 'knn'
        self.temperature = temperature
        self.logsumexp_coef = logsumexp_coef
        self.knn_k = knn_k
        self.max_history_size = max_history_size
        self.persistent_memory = persistent_memory
        
        # CNN feature extractor - matches original
        self.model_cnn_extractor = CNNFeatureExtractor(input_shape, features_dim)
        
        # Encoder network (maps features to embedding space) - matches original
        self.encoder = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, embedding_dim)
        )
        
        # Potential network (for MRN potential energy function) - matches original
        self.potential_net = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # History storage for CNN features (not embeddings!) - matches original
        self.obs_history = defaultdict(lambda: None)
        
        print(f"TDDNetwork initialized:")
        print(f"  Features dim: {features_dim}, Embedding dim: {embedding_dim}")
        print(f"  Energy function: {energy_fn}")
        print(f"  Loss function: {loss_fn}")
        print(f"  Aggregate function: {aggregate_fn}")
        print(f"  CNN parameters: {sum(p.numel() for p in self.model_cnn_extractor.parameters()):,}")
        
    def _get_cnn_embeddings(self, observations):
        """Extract CNN features from observations - matches original"""
        return self.model_cnn_extractor(observations)
    
    def forward(self, curr_obs, next_obs):
        """
        Forward pass for contrastive learning - matches original exactly
        """
        # CNN Extractor
        curr_cnn_embs = self._get_cnn_embeddings(curr_obs)
        next_cnn_embs = self._get_cnn_embeddings(next_obs)
        phi_x = self.encoder(curr_cnn_embs)
        phi_y = self.encoder(next_cnn_embs)
        c_y = self.potential_net(next_cnn_embs)
        device = phi_x.device

        # Compute energy/similarity matrix - matches original exactly
        if self.energy_fn == 'l2':
            logits = -torch.sqrt(((phi_x[:, None] - phi_y[None, :])**2).sum(dim=-1) + 1e-8)
        elif self.energy_fn == 'cos':
            s_norm = torch.linalg.norm(phi_x, dim=-1, keepdim=True)
            g_norm = torch.linalg.norm(phi_y, dim=-1, keepdim=True)
            phi_x_norm = phi_x / (s_norm + 1e-8)
            phi_y_norm = phi_y / (g_norm + 1e-8)
            phi_x_norm = phi_x_norm / self.temperature
            logits = torch.einsum("ik,jk->ij", phi_x_norm, phi_y_norm)
        elif self.energy_fn == 'dot':
            logits = torch.einsum("ik,jk->ij", phi_x, phi_y)
        elif self.energy_fn == 'mrn':
            logits = -mrn_distance(phi_x[:, None], phi_y[None, :])
        elif self.energy_fn == 'mrn_pot':
            logits = c_y.T - mrn_distance(phi_x[:, None], phi_y[None, :])
        else:
            raise ValueError(f"Unknown energy function: {self.energy_fn}")
        
        batch_size = logits.size(0)
        I = torch.eye(batch_size, device=device)
        
        # Contrastive loss computation - matches original exactly
        if self.loss_fn == 'infonce':
            contrastive_loss = F.cross_entropy(logits, torch.arange(batch_size, device=device))
        elif self.loss_fn == 'infonce_backward':
            contrastive_loss = F.cross_entropy(logits.T, torch.arange(batch_size, device=device))
        elif self.loss_fn == 'infonce_symmetric':
            loss1 = F.cross_entropy(logits, torch.arange(batch_size, device=device))
            loss2 = F.cross_entropy(logits.T, torch.arange(batch_size, device=device))
            contrastive_loss = (loss1 + loss2) / 2
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")
        
        # Add logsumexp regularization - matches original
        logsumexp_loss = torch.mean(torch.logsumexp(logits + 1e-6, dim=1)**2)
        total_loss = contrastive_loss + self.logsumexp_coef * logsumexp_loss
        
        # Logging information - matches original
        logs = {
            'contrastive_loss': contrastive_loss.item(),
            'logsumexp_loss': logsumexp_loss.item(),
            'logits_pos': torch.diag(logits).mean().item(),
            'logits_neg': torch.mean(logits * (1 - I)).item(),
            'categorical_accuracy': torch.mean((torch.argmax(logits, dim=1) == torch.arange(batch_size, device=device)).float()).item(),
        }
        
        return total_loss, logs
    
    def get_intrinsic_rewards(self, curr_obs, next_obs):
        """
        Compute intrinsic rewards - matches original algorithm exactly
        """
        with torch.no_grad():
            batch_size = curr_obs.size(0)
            device = curr_obs.device
            
            # CNN Extractor
            curr_cnn_embs = self._get_cnn_embeddings(curr_obs)
            next_cnn_embs = self._get_cnn_embeddings(next_obs)
            
            int_rews = np.zeros(batch_size, dtype=np.float32)
            
            for env_id in range(batch_size):
                # Update historical observation embeddings - matches original exactly
                curr_obs_emb = curr_cnn_embs[env_id].view(1, -1)
                next_obs_emb = next_cnn_embs[env_id].view(1, -1)
                obs_embs = self.obs_history[env_id]
                
                # Concatenate embeddings - matches original
                if obs_embs is None:
                    new_embs = [curr_obs_emb, next_obs_emb]
                else:
                    new_embs = [obs_embs, next_obs_emb]
                obs_embs = torch.cat(new_embs, dim=0)
                
                # Limit history size
                if obs_embs.shape[0] > self.max_history_size:
                    obs_embs = obs_embs[-self.max_history_size:]
                
                self.obs_history[env_id] = obs_embs
                
                # Encode features to embeddings - matches original
                phi_x = self.encoder(self.obs_history[env_id][:-1])  # All historical
                phi_y = self.encoder(self.obs_history[env_id][-1].unsqueeze(0))  # Current
                
                # Compute distances - matches original exactly
                if self.energy_fn == 'l2':
                    dists = torch.sqrt(((phi_x - phi_y)**2).sum(dim=-1) + 1e-8)
                elif self.energy_fn == 'cos':
                    x_norm = torch.linalg.norm(phi_x, dim=-1, keepdim=True)
                    y_norm = torch.linalg.norm(phi_y, dim=-1, keepdim=True)
                    phi_x_norm = phi_x / (x_norm + 1e-8)
                    phi_y_norm = phi_y / (y_norm + 1e-8)
                    phi_x_norm = phi_x_norm / self.temperature
                    dists = -torch.einsum("ik,jk->ij", phi_x_norm, phi_y_norm).squeeze(-1)
                elif self.energy_fn == 'dot':
                    dists = -torch.einsum("ik,jk->ij", phi_x, phi_y).squeeze(-1)
                elif 'mrn' in self.energy_fn:
                    dists = mrn_distance(phi_x, phi_y)
                
                # Compute intrinsic reward - matches original exactly
                if self.aggregate_fn == 'min':
                    int_rew = dists.min().item()
                elif self.aggregate_fn == 'quantile10':
                    int_rew = torch.quantile(dists, 0.1).item()
                elif self.aggregate_fn == 'knn':
                    if len(dists) <= self.knn_k:
                        knn_dists = dists
                    else:
                        knn_dists, _ = torch.topk(dists, self.knn_k, largest=False)
                    int_rew = knn_dists[-1].item()
                else:
                    int_rew = dists.min().item()
                    
                int_rews[env_id] = int_rew
            
        return int_rews
    
    def reset_episode_history(self, env_ids=None):
        """Reset episode history for specified environments"""
        if not self.persistent_memory:  # Only reset if not using persistent memory
            if env_ids is None:
                self.obs_history.clear()
            else:
                for env_id in env_ids:
                    if env_id in self.obs_history:
                        del self.obs_history[env_id]

class TemporalDistanceDensityCuriosity:
    """
    Temporal Distance Density (TDD) Curiosity method - matches original paper
    """
    
    def __init__(self, obs_size, act_size, state_size, device='cuda' if torch.cuda.is_available() else 'cpu',
                 eta=1.0, lr=0.001, features_dim=256, embedding_dim=64, energy_fn='mrn_pot', 
                 loss_fn='infonce', aggregate_fn='min', temperature=1.0, logsumexp_coef=0.1, 
                 knn_k=10, max_history_size=1000, persistent_memory=True, **kwargs):
        """
        Initialize the TDD Curiosity model - matches original paper interface
        """
        print(f"TemporalDistanceDensityCuriosity - obs_size: {obs_size}, act_size: {act_size}")
        
        self.device = device
        self.eta = eta
        self.act_size = act_size
        self.features_dim = features_dim
        self.embedding_dim = embedding_dim
        
        # Validate input shape
        if len(obs_size) != 3:
            raise ValueError(f"Expected obs_size to have 3 dimensions, got {len(obs_size)}: {obs_size}")
        
        self.input_shape = obs_size
        print(f"Using input_shape: {self.input_shape}")
        
        # Initialize TDD network - matches original architecture
        self.network = TDDNetwork(
            self.input_shape, act_size, features_dim, embedding_dim, lr, energy_fn, 
            loss_fn, aggregate_fn, temperature, logsumexp_coef, knn_k, max_history_size,
            persistent_memory
        ).to(device)
        
        print(f"TemporalDistanceDensityCuriosity initialized with eta={eta}, lr={lr}")
        print(f"Features dim: {features_dim}, Embedding dim: {embedding_dim}")
        print(f"Energy function: {energy_fn}, Loss function: {loss_fn}, Aggregate function: {aggregate_fn}")
        print(f"Persistent memory across episodes: {persistent_memory}, Max history size: {max_history_size}")
        
    def _preprocess_observations(self, observations):
        """Convert observations to proper tensor format"""
        if not isinstance(observations, torch.Tensor):
            observations = torch.FloatTensor(observations)
        
        observations = observations.to(self.device)
        return observations
        
    def curiosity(self, observations, actions, next_observations):
        """
        Compute curiosity rewards - matches original interface
        """
        self.network.eval()
        
        # Preprocess inputs
        curr_obs = self._preprocess_observations(observations)
        next_obs = self._preprocess_observations(next_observations)
        
        # Debug: Check input validity
        if torch.isnan(curr_obs).any() or torch.isinf(curr_obs).any():
            print("WARNING: TDD received invalid current observations (NaN/Inf)")
            return np.zeros(curr_obs.shape[0])
            
        if torch.isnan(next_obs).any() or torch.isinf(next_obs).any():
            print("WARNING: TDD received invalid next observations (NaN/Inf)")
            return np.zeros(next_obs.shape[0])
        
        # Compute intrinsic rewards - matches original algorithm
        intrinsic_rewards = self.network.get_intrinsic_rewards(curr_obs, next_obs)
        
        # Debug: Check reward validity
        if np.isnan(intrinsic_rewards).any() or np.isinf(intrinsic_rewards).any():
            print("WARNING: TDD computed invalid rewards (NaN/Inf)")
            return np.zeros(intrinsic_rewards.shape[0])
        
        # Scale by eta
        intrinsic_rewards = self.eta * intrinsic_rewards
        
        # Debug: Print statistics occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 1000 == 0:  # Print every 1000 calls
            print(f"TDD Debug - Raw rewards: mean={intrinsic_rewards.mean():.6f}, "
                  f"std={intrinsic_rewards.std():.6f}, "
                  f"min={intrinsic_rewards.min():.6f}, "
                  f"max={intrinsic_rewards.max():.6f}")
        
        return intrinsic_rewards
    
    def train(self, observations, actions, next_observations):
        """
        Update the TDD model using contrastive learning - matches original
        """
        self.network.train()
        
        # Preprocess inputs
        curr_obs = self._preprocess_observations(observations)
        next_obs = self._preprocess_observations(next_observations)
        
        # Compute TDD loss using contrastive learning - matches original
        total_loss, logs = self.network.forward(curr_obs, next_obs)
        
        # Backward pass
        self.network.optimizer.zero_grad()
        total_loss.backward()
        self.network.optimizer.step()
        
        # Return loss info (total_loss, contrastive_loss for compatibility)
        return (total_loss.item(), logs['contrastive_loss'])
    
    def reset_episodes(self, env_ids=None):
        """Reset episode history for specified environments"""
        self.network.reset_episode_history(env_ids)
    
    def save(self, filepath):
        """Save the TDD model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.network.optimizer.state_dict(),
            'obs_history': dict(self.network.obs_history),
            'hyperparameters': {
                'input_shape': self.input_shape,
                'act_size': self.act_size,
                'eta': self.eta,
                'features_dim': self.features_dim,
                'embedding_dim': self.embedding_dim,
            }
        }, filepath)
        print(f"TemporalDistanceDensityCuriosity model saved to {filepath}")
    
    def load(self, filepath):
        """Load the TDD model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network state
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load observation history
        if 'obs_history' in checkpoint:
            self.network.obs_history = defaultdict(lambda: None, checkpoint['obs_history'])
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.eta = hyperparams['eta']
        self.features_dim = hyperparams['features_dim']
        self.embedding_dim = hyperparams['embedding_dim']
        
        print(f"TemporalDistanceDensityCuriosity model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to path for testing
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    print("Testing TemporalDistanceDensityCuriosity...")
    
    # Example for Atari-style visual input
    obs_size = (4, 84, 84)  # Frame stack format (channels, height, width)
    act_size = 6            # Typical Atari action space
    state_size = obs_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize TDD curiosity model - matches original hyperparameters
    curiosity_model = TemporalDistanceDensityCuriosity(
        obs_size, act_size, state_size, device, 
        eta=1.0, lr=0.001, features_dim=256, embedding_dim=64, energy_fn='mrn_pot',
        loss_fn='infonce', aggregate_fn='min', persistent_memory=True
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
    print(f"Training losses - Total: {loss_info[0]:.4f}, Contrastive: {loss_info[1]:.4f}")
    
    # Test multiple training iterations
    print(f"\nTesting learning over multiple iterations...")
    initial_rewards = rewards.copy()
    
    for i in range(10):
        # Generate new data for each iteration
        obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
        next_obs_raw = torch.randint(0, 256, (batch_size, *obs_size), dtype=torch.float32)
        obs = (obs_raw - obs_raw.mean()) / (obs_raw.std() + 1e-8)
        next_obs = (next_obs_raw - next_obs_raw.mean()) / (next_obs_raw.std() + 1e-8)
        
        loss_info = curiosity_model.train(obs, actions, next_obs)
        if i % 3 == 0:
            print(f"  Iteration {i+1}: Total Loss = {loss_info[0]:.4f}, Contrastive = {loss_info[1]:.4f}")
    
    # Test final rewards (should adapt as model learns)
    final_rewards = curiosity_model.curiosity(obs, actions, next_obs)
    print(f"\nReward evolution:")
    print(f"  Initial rewards mean: {initial_rewards.mean():.4f}")
    print(f"  Final rewards mean: {final_rewards.mean():.4f}")
    print(f"  Change: {final_rewards.mean() - initial_rewards.mean():.4f}")
    
    print("TemporalDistanceDensityCuriosity test completed successfully!")