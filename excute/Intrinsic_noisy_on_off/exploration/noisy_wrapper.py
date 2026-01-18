import numpy as np, random, gymnasium as gym
from gymnasium import spaces
from PIL import Image

class NoisyTVEnvWrapperCIFAR(gym.Wrapper):
    def __init__(self, env, get_random_cifar, num_random_actions=1):
        super().__init__(env)
        self.get_random_cifar, self.num_random_actions = get_random_cifar, num_random_actions
        self.original_actions = self.env.action_space.n
        self.random_actions = list(range(self.original_actions, self.original_actions + num_random_actions))
        self.action_space = spaces.Discrete(self.original_actions + num_random_actions)
        try: self.get_random_cifar()
        except Exception as e: print(f"   ❌ CIFAR test failed: {e}")

    def step(self, action):
        if action in self.random_actions:
            obs, reward, terminated, truncated, info = self.env.step(0) # NOOP
            info['noisy'] = True
            return self.add_noisy_tv(obs), reward, terminated, truncated, info
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
            info['noisy'] = False
            return obs, reward, terminated, truncated, info

    def add_noisy_tv(self, obs):
        try: return self._create_replacement(self.get_random_cifar(), obs)
        except Exception: return np.random.randint(0, 256, size=obs.shape, dtype=obs.dtype)

    def _create_replacement(self, cifar_img, target_obs):
        cifar_gray = np.dot(cifar_img, [0.2989, 0.5870, 0.1140]).astype(np.uint8) if len(cifar_img.shape) == 3 else cifar_img
        h, w = target_obs.shape[0], target_obs.shape[1]
        tiled = np.tile(cifar_gray, ((h+31)//32, (w+31)//32))
        cropped = tiled[:h, :w]
        if len(target_obs.shape) == 3: replacement = np.stack([cropped] * target_obs.shape[2], axis=2)
        else: replacement = cropped
        return replacement.astype(target_obs.dtype)