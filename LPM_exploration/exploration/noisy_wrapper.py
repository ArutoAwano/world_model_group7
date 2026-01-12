"""
Debug CIFAR Wrapper - Adds extensive logging to find the issue
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

class NoisyTVEnvWrapperCIFAR(gym.Wrapper):
    def __init__(self, env, get_random_cifar, num_random_actions=1):
        super().__init__(env)
        
        self.get_random_cifar = get_random_cifar
        self.num_random_actions = num_random_actions
        
        # Store original action space info
        self.original_actions = self.env.action_space.n
        self.random_actions = list(range(self.original_actions, self.original_actions + num_random_actions))
        
        # Update action space
        self.action_space = spaces.Discrete(self.original_actions + num_random_actions)
        
        # print(f"🎯 [CIFAR DEBUG] NoisyTVEnvWrapperCIFAR initialized:")
        # print(f"   Environment: {type(env).__name__}")
        # print(f"   Original action space: {self.original_actions}")
        # print(f"   Random actions: {self.random_actions}")
        # print(f"   New action space: {self.action_space.n}")
        # print(f"   Observation space: {env.observation_space}")
        
        # Test CIFAR function
        try:
            test_cifar = self.get_random_cifar()
            # print(f"   CIFAR test: shape={test_cifar.shape}")
        except Exception as e:
            print(f"   ❌ CIFAR test failed: {e}")
    
    def step(self, action):
        # print(f"🎮 [CIFAR DEBUG] Action received: {action} (type: {type(action)})")
        
        if action in self.random_actions:
            # print(f"✅ [CIFAR DEBUG] Random action {action}, passing through")
            # return self.env.step(1)
            # print(f"🎯 [CIFAR DEBUG] RANDOM ACTION TRIGGERED: {action}")
            
            # Execute NOOP instead
            obs, reward, terminated, truncated, info = self.env.step(0)
            # print(f"   Original obs shape: {obs.shape}, dtype: {obs.dtype}")
            
            # Apply CIFAR replacement
            obs_replaced = self.add_noisy_tv(obs)
            # print(f"   Replaced obs shape: {obs_replaced.shape}, dtype: {obs_replaced.dtype}")
            
            # Verify replacement worked
            # if np.array_equal(obs, obs_replaced):
            #     print(f"   ❌ WARNING: Replacement identical to original!")
            # else:
            #     print(f"   ✅ SUCCESS: Replacement different from original")
            #     print(f"   Original range: [{obs.min():.1f}, {obs.max():.1f}]")
            #     print(f"   Replaced range: [{obs_replaced.min():.1f}, {obs_replaced.max():.1f}]")
            
            return obs_replaced, reward, terminated, truncated, info
        else:
            # print(f"✅ [CIFAR DEBUG] Normal action {action}, passing through")
            return self.env.step(action)
    
    def add_noisy_tv(self, obs):
        obs = obs.copy()
        """Replace observation with CIFAR-based noise"""
        # print(f"🎨 [CIFAR DEBUG] add_noisy_tv called:")
        # print(f"   Input obs shape: {obs.shape}, dtype: {obs.dtype}")
        
        try:
            # Get random CIFAR image
            cifar_img = self.get_random_cifar()  # Should be (32, 32, 3)
            # print(f"   CIFAR image: shape={cifar_img.shape}")
            
            # Convert to grayscale
            if len(cifar_img.shape) == 3 and cifar_img.shape[2] == 3:
                cifar_gray = np.dot(cifar_img, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                # print(f"   CIFAR grayscale: shape={cifar_gray.shape}")
            else:
                cifar_gray = cifar_img
            
            # Create replacement matching obs shape exactly
            replacement = self._create_replacement(cifar_gray, obs)
            
            return replacement
            
        except Exception as e:
            # print(f"   ❌ Error in add_noisy_tv: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: simple random noise
            replacement = np.random.randint(0, 256, size=obs.shape, dtype=obs.dtype)
            # print(f"   Using fallback random noise: shape={replacement.shape}")
            return replacement
    
    def _create_replacement(self, cifar_gray, target_obs):
        """Create replacement that exactly matches target observation"""
        target_shape = target_obs.shape
        target_dtype = target_obs.dtype
        
        # print(f"   Creating replacement for shape {target_shape}")
        
        if len(target_shape) == 3:
            # 3D observation (H, W, C)
            h, w, c = target_shape
            
            # Tile CIFAR to cover the area
            tile_h = (h + 31) // 32
            tile_w = (w + 31) // 32
            
            tiled = np.tile(cifar_gray, (tile_h, tile_w))
            cropped = tiled[:h, :w]
            
            if c == 1:
                replacement = cropped.reshape(h, w, 1)
            elif c == 3:
                replacement = np.stack([cropped] * 3, axis=2)
            else:
                replacement = np.stack([cropped] * c, axis=2)
                
        elif len(target_shape) == 2:
            # 2D observation (H, W)
            h, w = target_shape
            tile_h = (h + 31) // 32
            tile_w = (w + 31) // 32
            
            tiled = np.tile(cifar_gray, (tile_h, tile_w))
            replacement = tiled[:h, :w]
            
        else:
            # print(f"   Unsupported shape {target_shape}, using random noise")
            replacement = np.random.randint(0, 256, size=target_shape, dtype=target_dtype)
        
        # Ensure correct dtype
        replacement = replacement.astype(target_dtype)
        # print(f"   Created replacement: shape={replacement.shape}, dtype={replacement.dtype}")
        
        return replacement
    
    def get_action_meanings(self):
        """Get action meanings including CIFAR actions"""
        if hasattr(self.env, 'get_action_meanings'):
            meanings = self.env.get_action_meanings()
        else:
            meanings = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN']
        
        meanings = list(meanings)
        for i in range(self.num_random_actions):
            meanings.append(f'CIFAR_DEBUG_{i+1}')
        
        return meanings

class DebugNoisyTVEnvWrapperCIFAR(gym.Wrapper):
    def __init__(self, env, get_random_cifar, num_random_actions=1):
        super().__init__(env)
        
        self.get_random_cifar = get_random_cifar
        self.num_random_actions = num_random_actions
        
        # Get original action space
        original_num_actions = self.env.action_space.n
        self.random_action_start = original_num_actions
        self.random_actions = list(range(original_num_actions, original_num_actions + num_random_actions))
        
        # Update action space
        self.action_space = spaces.Discrete(original_num_actions + num_random_actions)
        
        # print(f"🔧 [DEBUG] DebugNoisyTVEnvWrapperCIFAR initialized:")
        # print(f"   Original actions: 0-{original_num_actions-1}")
        # print(f"   Random actions: {self.random_actions}")
        # print(f"   Total action space: {self.action_space.n}")
        
        # Test CIFAR function
        try:
            test_cifar = self.get_random_cifar()
            # print(f"   CIFAR function test: shape={test_cifar.shape}, dtype={test_cifar.dtype}")
        except Exception as e:
            print(f"   ❌ CIFAR function test failed: {e}")
    
    def step(self, action):
        # print(f"🎮 [DEBUG] Action received: {action} (type: {type(action)})")
        
        if action in self.random_actions:
            # print(f"🎯 [DEBUG] RANDOM ACTION DETECTED! Action {action} is in {self.random_actions}")
            
            # Execute NOOP instead of the random action
            actual_action = 0
            # print(f"   Executing NOOP (action 0) instead")
            
            obs, reward, terminated, truncated, info = self.env.step(actual_action)
            # print(f"   Original obs shape: {obs.shape}")
            
            # Replace observation with CIFAR
            obs_replaced = self.add_noisy_tv(obs)
            # print(f"   Replaced obs shape: {obs_replaced.shape}")
            
            return obs_replaced, reward, terminated, truncated, info
        else:
            # print(f"✅ [DEBUG] Normal action {action}, passing through")
            return self.env.step(action)
    
    def add_noisy_tv(self, obs):
        """Replace observation with CIFAR images"""
        # print(f"🎨 [DEBUG] add_noisy_tv called with obs shape: {obs.shape}")
        
        try:
            # Get random CIFAR image
            random_cifar = self.get_random_cifar()
            # print(f"   Got CIFAR image: shape={random_cifar.shape}")
            
            # Convert to grayscale
            if len(random_cifar.shape) == 3 and random_cifar.shape[2] == 3:
                cifar_gray = self.grayscale(random_cifar)
                # print(f"   Converted to grayscale: shape={cifar_gray.shape}")
            else:
                cifar_gray = random_cifar
                # print(f"   Using CIFAR as-is: shape={cifar_gray.shape}")
            
            # Create replacement matching obs shape
            replacement = self._create_shape_matched_replacement(cifar_gray, obs)
            # print(f"   Created replacement: shape={replacement.shape}")
            
            # Verify replacement is different from original
            # if np.array_equal(replacement, obs):
            #     print(f"   ⚠️  WARNING: Replacement identical to original!")
            # else:
            #     print(f"   ✅ Replacement is different from original")
            #     print(f"   Original range: [{obs.min():.1f}, {obs.max():.1f}]")
            #     print(f"   Replacement range: [{replacement.min():.1f}, {replacement.max():.1f}]")
            
            return replacement
            
        except Exception as e:
            # print(f"   ❌ Error in add_noisy_tv: {e}")
            import traceback
            traceback.print_exc()
            return obs  # Return original on error
    
    def grayscale(self, color_image):
        """Convert RGB to grayscale"""
        if len(color_image.shape) == 3 and color_image.shape[2] == 3:
            grayscale = (color_image[:,:,0] * 0.2989 + 
                        color_image[:,:,1] * 0.5870 + 
                        color_image[:,:,2] * 0.1140)
            return grayscale.astype(np.uint8)
        return color_image
    
    def _create_shape_matched_replacement(self, cifar_gray, target_obs):
        """Create CIFAR replacement matching target observation shape"""
        target_shape = target_obs.shape
        target_dtype = target_obs.dtype
        
        # print(f"   Target shape: {target_shape}, dtype: {target_dtype}")
        
        # Handle different target shapes
        if len(target_shape) == 3:
            target_h, target_w, target_c = target_shape
            
            # Tile CIFAR to match dimensions
            tile_h = max(1, (target_h + 31) // 32)
            tile_w = max(1, (target_w + 31) // 32)
            
            tiled = np.tile(cifar_gray, (tile_h, tile_w))
            tiled_cropped = tiled[:target_h, :target_w]
            
            if target_c == 1:
                replacement = tiled_cropped.reshape(target_h, target_w, 1)
            elif target_c == 3:
                replacement = np.stack([tiled_cropped] * 3, axis=2)
            else:
                replacement = np.stack([tiled_cropped] * target_c, axis=2)
                
        elif len(target_shape) == 2:
            target_h, target_w = target_shape
            tile_h = max(1, (target_h + 31) // 32)
            tile_w = max(1, (target_w + 31) // 32)
            
            tiled = np.tile(cifar_gray, (tile_h, tile_w))
            replacement = tiled[:target_h, :target_w]
        else:
            # print(f"   ⚠️  Unsupported shape {target_shape}, using random noise")
            replacement = np.random.randint(0, 256, size=target_shape, dtype=target_dtype)
        
        return replacement.astype(target_dtype)
    
    def get_action_meanings(self):
        """Return action meanings"""
        if hasattr(self.env, 'get_action_meanings'):
            meanings = self.env.get_action_meanings()
        else:
            meanings = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN']
        
        meanings = list(meanings)
        for i in range(self.num_random_actions):
            meanings.append(f'DEBUG_CIFAR_{i+1}')
        return meanings


def debug_wrapper_in_pipeline():
    """Debug the wrapper within the actual pipeline"""
    
    # print("🔍 Debug CIFAR Wrapper in Pipeline")
    print("=" * 40)
    
    import torch
    import sys
    sys.path.append('.')
    
    from exploration.cifar import create_cifar_function_simple
    
    # Create CIFAR function
    get_cifar = create_cifar_function_simple()
    
    # Test CIFAR function directly
    print("🧪 Testing CIFAR function...")
    for i in range(3):
        test_img = get_cifar()
        print(f"   CIFAR {i+1}: shape={test_img.shape}, range=[{test_img.min()}, {test_img.max()}]")
    
    # Create environment with debug wrapper
    print(f"\n🎮 Creating environment with debug wrapper...")
    
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    
    # Create base environment
    env = gym.make('ALE/Breakout-v5')
    print(f"   Base env action space: {env.action_space}")
    
    # Apply debug wrapper
    debug_env = DebugNoisyTVEnvWrapperCIFAR(env, get_cifar, num_random_actions=2)
    print(f"   Wrapped env action space: {debug_env.action_space}")
    
    # Test the wrapper
    print(f"\n🔧 Testing wrapper functionality...")
    
    obs, info = debug_env.reset()
    print(f"   Reset obs shape: {obs.shape}")
    
    # Test normal action
    print(f"\n--- Testing Normal Action ---")
    obs_normal, reward_normal, terminated, truncated, info = debug_env.step(1)
    
    # Test random action
    print(f"\n--- Testing Random Action ---")
    random_action_id = debug_env.random_actions[0]
    obs_random, reward_random, terminated, truncated, info = debug_env.step(random_action_id)
    
    # Compare observations
    print(f"\n📊 Comparison:")
    print(f"   Normal obs: shape={obs_normal.shape}, range=[{obs_normal.min():.1f}, {obs_normal.max():.1f}]")
    print(f"   Random obs: shape={obs_random.shape}, range=[{obs_random.min():.1f}, {obs_random.max():.1f}]")
    
    if np.array_equal(obs_normal, obs_random):
        print(f"   ❌ PROBLEM: Observations are identical!")
    else:
        print(f"   ✅ SUCCESS: Observations are different!")
    
    # Create visual comparison
    create_debug_visualization(obs_normal, obs_random, debug_env)
    
    debug_env.close()

def create_debug_visualization(obs_normal, obs_random, env):
    """Create side-by-side comparison"""
    
    print(f"\n🎨 Creating debug visualization...")
    
    # Extract displayable frames
    def extract_frame(obs):
        if len(obs.shape) == 3:
            if obs.shape[2] == 1:
                return obs[:, :, 0]
            elif obs.shape[0] <= 4:
                return obs[-1]
            else:
                return obs[:, :, 0]
        return obs
    
    frame_normal = extract_frame(obs_normal)
    frame_random = extract_frame(obs_random)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(frame_normal, cmap='gray')
    axes[0].set_title(f'Normal Action\nShape: {frame_normal.shape}\nRange: [{frame_normal.min()}, {frame_normal.max()}]')
    axes[0].axis('off')
    
    axes[1].imshow(frame_random, cmap='gray')
    axes[1].set_title(f'Random CIFAR Action\nShape: {frame_random.shape}\nRange: [{frame_random.min()}, {frame_random.max()}]')
    axes[1].axis('off')
    
    if np.array_equal(frame_normal, frame_random):
        fig.suptitle('❌ PROBLEM: Frames are identical!', color='red', fontweight='bold')
    else:
        fig.suptitle('✅ SUCCESS: Frames are different!', color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('debug_cifar_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Debug visualization saved as 'debug_cifar_comparison.png'")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    debug_wrapper_in_pipeline()