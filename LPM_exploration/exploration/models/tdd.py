import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import random

class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for visual observations"""
    def __init__(self, input_shape):
        super(CNNFeatureExtractor, self).__init__()
        
        print(f"TDD CNNFeatureExtractor - input_shape: {input_shape}")
        
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
        
        print(f"TDD Conv output dimensions: {conv_height} x {conv_width}")
        
        if conv_width <= 0 or conv_height <= 0:
            raise ValueError(f"Convolution output dimensions are invalid: {conv_height}x{conv_width}")
        
        conv_output_size = conv_width * conv_height * 64
        print(f"TDD Conv output size: {conv_output_size}")
        
        # Dense layer for feature extraction
        self.fc = nn.Linear(conv_output_size, 256)
        self.feature_size = 256

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

def mrn_distance(x, y):
    """Metric Residual Network (MRN) distance function"""
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
    Temporal Distance Density Network for episodic novelty detection
    
    Uses contrastive learning to learn representations and computes intrinsic rewards
    based on temporal distance to previously observed states.
    """
    
    def __init__(self, input_shape, num_actions, embedding_dim=64, lr=0.001, 
                 energy_fn='mrn_pot', loss_fn='infonce', aggregate_fn='min',
                 temperature=1.0, logsumexp_coef=0.1, knn_k=10):
        super(TDDNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.energy_fn = energy_fn  # 'l2', 'cos', 'mrn', 'mrn_pot'
        self.loss_fn = loss_fn  # 'infonce', 'infonce_symmetric', 'infonce_backward'
        self.aggregate_fn = aggregate_fn  # 'min', 'quantile10', 'knn'
        self.temperature = temperature
        self.logsumexp_coef = logsumexp_coef
        self.knn_k = knn_k
        
        # Feature extractor
        self.feature_extractor = CNNFeatureExtractor(input_shape)
        feature_size = self.feature_extractor.feature_size
        
        # Encoder network (maps features to embedding space)
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # Potential network (for MRN potential energy function)
        self.potential_net = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # History storage for each environment (for episodic memory)
        self.observation_history = defaultdict(list)
        self.max_history_size = 1000  # Limit memory usage
        
        print(f"TDDNetwork initialized:")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Energy function: {energy_fn}")
        print(f"  Loss function: {loss_fn}")
        print(f"  Aggregate function: {aggregate_fn}")
        print(f"  Feature extractor parameters: {sum(p.numel() for p in self.feature_extractor.parameters()):,}")
        
    def _get_embeddings(self, observations):
        """Extract embeddings from observations"""
        features = self.feature_extractor(observations)
        embeddings = self.encoder(features)
        return embeddings, features
    
    def forward(self, curr_obs, next_obs):
        """
        Forward pass for contrastive learning
        
        Args:
            curr_obs: Current observations (batch_size, channels, height, width)
            next_obs: Next observations (batch_size, channels, height, width)
        Returns:
            contrastive_loss: Loss for training
            logs: Dictionary of logging information
        """
        # Get embeddings and features
        phi_x, curr_features = self._get_embeddings(curr_obs)
        phi_y, next_features = self._get_embeddings(next_obs)
        
        batch_size = phi_x.shape[0]
        device = phi_x.device
        
        # Compute energy/similarity matrix
        if self.energy_fn == 'l2':
            logits = -torch.sqrt(((phi_x[:, None] - phi_y[None, :])**2).sum(dim=-1) + 1e-8)
        elif self.energy_fn == 'cos':
            x_norm = torch.linalg.norm(phi_x, dim=-1, keepdim=True)
            y_norm = torch.linalg.norm(phi_y, dim=-1, keepdim=True)
            phi_x_norm = phi_x / (x_norm + 1e-8)
            phi_y_norm = phi_y / (y_norm + 1e-8)
            phi_x_norm = phi_x_norm / self.temperature
            logits = torch.einsum("ik,jk->ij", phi_x_norm, phi_y_norm)
        elif self.energy_fn == 'mrn':
            logits = -mrn_distance(phi_x[:, None], phi_y[None, :])
        elif self.energy_fn == 'mrn_pot':
            c_y = self.potential_net(next_features)
            logits = c_y.T - mrn_distance(phi_x[:, None], phi_y[None, :])
        else:
            raise ValueError(f"Unknown energy function: {self.energy_fn}")
        
        # Contrastive loss computation
        I = torch.eye(batch_size, device=device)
        
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
        
        # Add logsumexp regularization
        logsumexp_loss = torch.mean(torch.logsumexp(logits + 1e-6, dim=1)**2)
        total_loss = contrastive_loss + self.logsumexp_coef * logsumexp_loss
        
        # Logging information
        logs = {
            'contrastive_loss': contrastive_loss.item(),
            'logsumexp_loss': logsumexp_loss.item(),
            'logits_pos': torch.diag(logits).mean().item(),
            'logits_neg': torch.mean(logits * (1 - I)).item(),
            'categorical_accuracy': torch.mean((torch.argmax(logits, dim=1) == torch.arange(batch_size, device=device)).float()).item(),
        }
        
        return total_loss, logs
    
    def compute_intrinsic_reward(self, observations):
        """
        Compute intrinsic rewards based on temporal distance to historical observations
        
        Args:
            observations: Current observations (batch_size, channels, height, width)
        Returns:
            intrinsic_rewards: Temporal distance-based rewards (batch_size,)
        """
        with torch.no_grad():
            batch_size = observations.shape[0]
            device = observations.device
            embeddings, _ = self._get_embeddings(observations)
            intrinsic_rewards = torch.zeros(batch_size, device=device)
            
            for i in range(batch_size):
                env_id = i  # In practice, would need proper environment ID tracking
                current_embedding = embeddings[i].unsqueeze(0)
                
                # Get historical embeddings for this environment
                if env_id in self.observation_history and len(self.observation_history[env_id]) > 0:
                    # Move historical embeddings to the same device as current embedding
                    historical_embeddings = torch.stack([emb.to(device) for emb in self.observation_history[env_id]])
                    
                    # Compute distances to historical observations
                    if self.energy_fn == 'l2':
                        dists = torch.sqrt(((current_embedding - historical_embeddings)**2).sum(dim=-1) + 1e-8)
                    elif self.energy_fn == 'cos':
                        curr_norm = torch.linalg.norm(current_embedding, dim=-1, keepdim=True)
                        hist_norm = torch.linalg.norm(historical_embeddings, dim=-1, keepdim=True)
                        curr_norm_emb = current_embedding / (curr_norm + 1e-8)
                        hist_norm_emb = historical_embeddings / (hist_norm + 1e-8)
                        dists = -torch.einsum("ik,jk->ij", curr_norm_emb, hist_norm_emb).squeeze(0)
                    elif 'mrn' in self.energy_fn:
                        dists = mrn_distance(current_embedding, historical_embeddings).squeeze(0)
                    
                    # Aggregate distances to get intrinsic reward
                    if self.aggregate_fn == 'min':
                        intrinsic_rewards[i] = dists.min()
                    elif self.aggregate_fn == 'quantile10':
                        intrinsic_rewards[i] = torch.quantile(dists, 0.1)
                    elif self.aggregate_fn == 'knn':
                        if len(dists) <= self.knn_k:
                            knn_dists = dists
                        else:
                            knn_dists, _ = torch.topk(dists, self.knn_k, largest=False)
                        intrinsic_rewards[i] = knn_dists[-1]
                else:
                    # No history yet, give maximum reward
                    intrinsic_rewards[i] = 1.0
                
                # Update history with current embedding (store on CPU to save GPU memory)
                self.observation_history[env_id].append(current_embedding.squeeze(0).detach().cpu())
                
                # Limit history size to prevent memory explosion
                if len(self.observation_history[env_id]) > self.max_history_size:
                    self.observation_history[env_id].pop(0)
            
        return intrinsic_rewards
    
    def reset_episode_history(self, env_ids=None):
        """Reset episode history for specified environments"""
        if env_ids is None:
            self.observation_history.clear()
        else:
            for env_id in env_ids:
                if env_id in self.observation_history:
                    del self.observation_history[env_id]

class TemporalDistanceDensityCuriosity:
    """
    Temporal Distance Density (TDD) Curiosity method
    
    Implementation of "Episodic Novelty Through Temporal Distance" which uses contrastive learning
    to learn representations and computes intrinsic rewards based on how temporally distant
    the current state is from previously observed states in the learned embedding space.
    """
    
    def __init__(self, obs_size, act_size, state_size, device='cuda' if torch.cuda.is_available() else 'cpu',
                 eta=1.0, lr=0.001, embedding_dim=64, energy_fn='mrn_pot', loss_fn='infonce', 
                 aggregate_fn='min', temperature=1.0, logsumexp_coef=0.1, knn_k=10, **kwargs):
        """
        Initialize the TDD Curiosity model.
        
        Args:
            obs_size: Observation space shape (channels, height, width)
            act_size: Number of possible actions
            state_size: State size (same as obs_size for visual inputs)
            device: Device to run computation on
            eta: Scaling factor for intrinsic rewards
            lr: Learning rate
            embedding_dim: Dimension of the learned embeddings
            energy_fn: Energy/distance function ('l2', 'cos', 'mrn', 'mrn_pot')
            loss_fn: Contrastive loss function ('infonce', 'infonce_symmetric', 'infonce_backward')
            aggregate_fn: Distance aggregation function ('min', 'quantile10', 'knn')
            temperature: Temperature for cosine similarity
            logsumexp_coef: Coefficient for logsumexp regularization
            knn_k: K value for knn aggregation
        """
        print(f"TemporalDistanceDensityCuriosity - obs_size: {obs_size}, act_size: {act_size}")
        
        self.device = device
        self.eta = eta
        self.act_size = act_size
        self.embedding_dim = embedding_dim
        
        # Validate input shape
        if len(obs_size) != 3:
            raise ValueError(f"Expected obs_size to have 3 dimensions, got {len(obs_size)}: {obs_size}")
        
        self.input_shape = obs_size
        print(f"Using input_shape: {self.input_shape}")
        
        # Initialize TDD network
        self.network = TDDNetwork(
            self.input_shape, act_size, embedding_dim, lr, energy_fn, 
            loss_fn, aggregate_fn, temperature, logsumexp_coef, knn_k
        ).to(device)
        
        print(f"TemporalDistanceDensityCuriosity initialized with eta={eta}, lr={lr}, embedding_dim={embedding_dim}")
        print(f"Energy function: {energy_fn}, Loss function: {loss_fn}, Aggregate function: {aggregate_fn}")
        
    def _preprocess_observations(self, observations):
        """Convert observations to proper tensor format"""
        if not isinstance(observations, torch.Tensor):
            observations = torch.FloatTensor(observations)
        
        observations = observations.to(self.device)
        return observations
        
    def curiosity(self, observations, actions, next_observations):
        """
        Compute curiosity rewards for a batch of transitions.
        
        Args:
            observations: Current observations
            actions: Actions taken (not used in TDD but kept for interface compatibility)
            next_observations: Next observations (used for intrinsic reward computation)
        Returns:
            intrinsic_rewards: Numpy array of intrinsic rewards
        """
        self.network.eval()
        
        # Preprocess inputs - use next_observations for novelty detection
        next_obs = self._preprocess_observations(next_observations)
        
        # Debug: Check input validity
        if torch.isnan(next_obs).any() or torch.isinf(next_obs).any():
            print("WARNING: TDD received invalid observations (NaN/Inf)")
            return np.zeros(next_obs.shape[0])
        
        # Compute intrinsic rewards based on temporal distance
        intrinsic_rewards = self.network.compute_intrinsic_reward(next_obs)
        
        # Debug: Check reward validity
        if torch.isnan(intrinsic_rewards).any() or torch.isinf(intrinsic_rewards).any():
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
            print(f"TDD Debug - Raw rewards: mean={intrinsic_rewards.mean().item():.6f}, "
                  f"std={intrinsic_rewards.std().item():.6f}, "
                  f"min={intrinsic_rewards.min().item():.6f}, "
                  f"max={intrinsic_rewards.max().item():.6f}")
        
        return intrinsic_rewards.cpu().numpy()
    
    def train(self, observations, actions, next_observations):
        """
        Update the TDD model using contrastive learning.
        
        Args:
            observations: Current observations
            actions: Actions taken (not used in TDD training)
            next_observations: Next observations
        Returns:
            loss_info: Tuple of (total_loss, contrastive_loss) for compatibility
        """
        self.network.train()
        
        # Preprocess inputs
        obs = self._preprocess_observations(observations)
        next_obs = self._preprocess_observations(next_observations)
        
        # Compute TDD loss using contrastive learning
        total_loss, logs = self.network.forward(obs, next_obs)
        
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
        """
        Save the TDD model
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.network.optimizer.state_dict(),
            'observation_history': dict(self.network.observation_history),
            'hyperparameters': {
                'input_shape': self.input_shape,
                'act_size': self.act_size,
                'eta': self.eta,
                'embedding_dim': self.embedding_dim,
            }
        }, filepath)
        print(f"TemporalDistanceDensityCuriosity model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the TDD model
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network state
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load observation history
        if 'observation_history' in checkpoint:
            self.network.observation_history = defaultdict(list, checkpoint['observation_history'])
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.eta = hyperparams['eta']
        self.embedding_dim = hyperparams['embedding_dim']
        
        print(f"TemporalDistanceDensityCuriosity model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    print("Testing TemporalDistanceDensityCuriosity...")
    
    # Example for Atari-style visual input
    obs_size = (4, 84, 84)  # Frame stack format (channels, height, width)
    act_size = 6            # Typical Atari action space
    state_size = obs_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize TDD curiosity model
    curiosity_model = TemporalDistanceDensityCuriosity(
        obs_size, act_size, state_size, device, 
        eta=1.0, lr=0.001, embedding_dim=64, energy_fn='mrn_pot',
        loss_fn='infonce', aggregate_fn='min'
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