import sys
import atexit
import os
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# Import from student_code and exploration
from student_code import Agent, RSSM, Encoder, Decoder, RewardModel, DiscountModel, Actor, DiscreteActor, Critic, ErrorPredictor, MSE
from exploration.noisy_wrapper import NoisyTVEnvWrapperCIFAR
from exploration.cifar import create_cifar_function_simple
import gymnasium as gym
import imageio
try:
    import craftium
except ImportError:
    craftium = None
from PIL import Image
import wandb
from torchvision import datasets, transforms
import random
import os

# Try importing stable_baselines3 wrappers if available, else define minimal ones
try:
    from stable_baselines3.common.atari_wrappers import (
        NoopResetEnv, MaxAndSkipEnv, WarpFrame, ClipRewardEnv
    )
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

class MnistEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(10) # Dummy action space
        
        # Load MNIST data
        try:
            self.mnist_data = datasets.MNIST('../data', train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize((64, 64)),
                                               transforms.ToTensor()
                                           ]))
        except:
            # Fallback if download fails or internet issue, generate random noise or try local
            print("Warning: Could not load actual MNIST, using random noise placeholder for MnistEnv")
            self.mnist_data = None

        self.current_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = random.randint(0, len(self.mnist_data) - 1) if self.mnist_data else 0
        return self._get_obs(), {}

    def step(self, action):
        # Action changes the image randomly to simulate "watching" different channels/digits
        self.current_idx = random.randint(0, len(self.mnist_data) - 1) if self.mnist_data else 0
        obs = self._get_obs()
        reward = 0.0 # No extrinsic reward for just watching
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        if self.mnist_data:
            img, _ = self.mnist_data[self.current_idx]
            # Convert tensor (C, H, W) 0-1 to numpy (H, W, C) 0-255
            img = img.permute(1, 2, 0).numpy() * 255.0
            img = img.astype(np.uint8)
            # MNIST is grayscale (1 channel), convert to RGB (3 channels) for consistency
            if img.shape[2] == 1:
                img = np.concatenate([img]*3, axis=2)
            return img
        else:
            return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    def render(self):
        return self._get_obs()

# Config Class from Notebook
class Config:
    def __init__(self, **kwargs):
        # data settings
        self.buffer_size = 100_000
        self.batch_size = 16
        self.seq_length = 50
        self.imagination_horizon = 20

        # model dimensions
        self.state_dim = 32
        self.num_classes = 32
        self.rnn_hidden_dim = 400
        self.mlp_hidden_dim = 300

        # learning params
        self.model_lr = 2e-4
        self.actor_lr = 4e-5
        self.critic_lr = 1e-4
        self.epsilon = 1e-5
        self.weight_decay = 1e-6
        self.gradient_clipping = 100
        self.kl_scale = 0.1
        self.kl_balance = 0.8
        self.actor_entropy_scale = 1e-3
        self.slow_critic_update = 100
        self.reward_loss_scale = 1.0
        self.discount_loss_scale = 1.0
        self.update_freq = 80

        # lambda return params
        self.discount = 0.995
        self.lambda_ = 0.95

        # learning period settings
        self.iter = 6000
        self.seed_iter = 1000 # Reduced for faster start in dev
        self.eval_interval = 150000
        self.eval_freq = 5
        self.eval_episodes = 5
        
        # LPM Params
        self.lpm_eta = 1.0
        self.lpm_lr = 2e-4
        
        # Override with kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


# Replay Buffer
class ReplayBuffer(object):
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)
        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done):
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done
        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(initial_index <= episode_borders,
                                              episode_borders < final_index).any()
            sampled_indexes += list(range(initial_index, final_index + 1))
        
        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:])
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1])
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        return sampled_observations, sampled_actions, sampled_rewards, sampled_done

    def __len__(self):
        return self.capacity if self.is_filled else self.index

def preprocess_obs(obs):
    obs = obs.astype(np.float32)
    normalized_obs = obs / 255.0 - 0.5
    return normalized_obs

def calculate_lambda_target(rewards, discounts, values, lambda_):
    V_lambda = torch.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            V_lambda[t] = rewards[t] + discounts[t] * values[t]
        else:
            V_lambda[t] = rewards[t] + discounts[t] * ((1-lambda_) * values[t+1] + lambda_ * V_lambda[t+1])
    return V_lambda

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluation(eval_env, policy, step, cfg):
    env = eval_env
    all_ep_rewards = []
    
    # Ensure directories exist
    os.makedirs("eval_view/video", exist_ok=True)
    os.makedirs("eval_view/images", exist_ok=True)
    
    with torch.no_grad():
        for i in range(cfg.eval_episodes):
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0] # Gym API handling
            policy.reset()
            done = False
            truncated = False
            episode_reward = []
            frames = [] 
            recon_frames = []
            
            while not done and not truncated:
                # Agent returns (action, recon_img)
                # recon_img is (64, 64, 3) range 0-1
                action, recon_img = policy(obs, eval=True)
                
                # Convert action to one-hot index if needed, but here simple env
                if isinstance(env.action_space, gym.spaces.Box):
                    # Clip action to space?
                    # policy returns numpy array
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, reward, done, truncated, info = env.step(action)
                elif hasattr(env.action_space, 'n'):
                    action_scalar = np.argmax(action)
                    obs, reward, done, truncated, info = env.step(action_scalar)
                else:
                    obs, reward, done, truncated, info = env.step(action)
                
                # Render for video (Real observation)
                frame = env.render() 
                frames.append(frame)
                
                # Reconstructed frame
                if recon_img is not None:
                    # recon_img is 0-1 float. Convert to 0-255 uint8
                    recon_frame = (recon_img * 255.0).astype(np.uint8)
                    recon_frames.append(recon_frame)
                
                episode_reward.append(reward)
            
            # Save video and images for the first episode of evaluation
            if i == 0 and len(frames) > 0:
                # Save Video (Side by Side if possible, or separate)
                video_path = f"eval_view/video/eval_iter_{step}_ep_{i}.mp4"
                try:
                     # Ensure frames are uint8 numpy arrays
                     frames = [np.array(f, dtype=np.uint8) for f in frames if f is not None]
                     
                     if len(recon_frames) > 0 and len(recon_frames) == len(frames):
                         # Create side-by-side video
                         # Resize frames to match if needed (env.render might be larger than 64x64)
                         # recon is 64x64. frame depends on env.
                         
                         combined_frames = []
                         for f, r in zip(frames, recon_frames):
                             # Resize f to match r (64x64) for simple concatenation
                             f_pil = Image.fromarray(f)
                             r_pil = Image.fromarray(r)
                             
                             f_s = f_pil.resize((64, 64))
                             
                             # Concatenate horizontally
                             comb = Image.new('RGB', (128, 64))
                             comb.paste(f_s, (0, 0))
                             comb.paste(r_pil, (64, 0))
                             combined_frames.append(np.array(comb))
                             
                         imageio.mimsave(video_path, combined_frames, fps=30)
                         print(f"Saved combined video to {video_path}")
                     else:
                        imageio.mimsave(video_path, frames, fps=30)
                        print(f"Saved video to {video_path}")

                     # Save Snapshot Images (Start, Middle, End)
                     if len(combined_frames) > 0:
                         indices = [0, len(combined_frames)//2, len(combined_frames)-1]
                         for idx in indices:
                             img_path = f"eval_view/images/eval_iter_{step}_ep_{i}_frame_{idx}.png"
                             Image.fromarray(combined_frames[idx]).save(img_path)

                except Exception as e:
                     print(f"Failed to save video: {e}")

            if len(episode_reward) < 500:
                pad_val = 10 if info.get("success", False) else 0
                episode_reward = np.pad(episode_reward, (0, 500 - len(episode_reward)), "constant", constant_values=pad_val)
            
            all_ep_rewards.append(np.mean(episode_reward))
            
        mean_ep_rewards = np.mean(all_ep_rewards)
        max_ep_rewards = np.max(all_ep_rewards)
        print(f"Eval(iter={step}) mean: {mean_ep_rewards:.4f} max: {max_ep_rewards:.4f}")
        
    return mean_ep_rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='Pendulum-v1') # Changed default to current working one, or keep generic
    parser.add_argument('--noisy-tv', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=200000)
    # WandB arguments
    parser.add_argument('--wandb', action='store_true', help='Use WandB logging')
    parser.add_argument('--wandb-project', default='Dreamer-LPM', help='WandB project name')
    parser.add_argument('--wandb-entity', default=None, help='WandB entity')
    parser.add_argument('--wandb-run-name', default=None, help='Run name')
    args = parser.parse_args()
    
    cfg = Config()
    cfg.iter = args.steps
    
    # Initialize WandB
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(cfg),
            mode="online"
        )
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Environment Setup
    # Using standard gym for now, assume wrapper handles Resize etc if needed
    # But notebook used GymWrapperMetaWorld. We will use standard Gym + Resize + NoisyTV
    
    # Simple env factory
    def make_env_simple(seed, noisy=False):
        # Use Pendulum as simple test env
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
    # Custom Render Wrapper
    class RenderWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            # We don't know shape yet until render, but usually (H, W, 3)
            # Let's assume ResizeObservation comes after.
            # Pendulum default render is 500x500
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(500, 500, 3), dtype=np.uint8)
            # Ensure metadata explicitly supports rgb_array
            self.metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}
            
        def reset(self, **kwargs):
            _, info = self.env.reset(**kwargs)
            obs = self.env.render()
            return obs, info
            
        def step(self, action):
            _, reward, terminated, truncated, info = self.env.step(action)
            obs = self.env.render()
            return obs, reward, terminated, truncated, info

            return obs, reward, terminated, truncated, info

    class NoisyTVWrapperDiscrete(gym.Wrapper):
        def __init__(self, env, get_random_cifar_fn, trigger_prob=0.05):
            super().__init__(env)
            self.get_random_cifar = get_random_cifar_fn
            self.trigger_prob = trigger_prob
            
        def step(self, action):
            # For discrete, we just randomly trigger NoisyTV for exploration noise
            # or trigger on specific action? Let's use random prob for simplicity or specific "do nothing"
            is_noise = random.random() < self.trigger_prob
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if is_noise:
                noise_img = self.get_random_cifar()
                pil_img = Image.fromarray(noise_img)
                pil_img = pil_img.resize((64, 64), Image.NEAREST)
                obs = np.array(pil_img)
                reward = 0.0
            
            info["noisy"] = is_noise
            return obs, reward, terminated, truncated, info

    class NoisyTVWrapperContinuous(gym.Wrapper):
        def __init__(self, env, get_random_cifar_fn, trigger_threshold=1.5):
            super().__init__(env)
            self.get_random_cifar = get_random_cifar_fn
            self.trigger_threshold = trigger_threshold
            # Action space remains same
            
        def step(self, action):
            # Check if action triggers noise
            # Assuming action is scalar or vector.
            # Trigger if any action component > threshold
            is_noise = np.any(np.abs(action) > self.trigger_threshold)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if is_noise:
                # Replace obs with random CIFAR
                noise_img = self.get_random_cifar() # (32, 32, 3)
                # Resize to (64, 64) ??
                # get_random_cifar returns 32x32.
                # We need 64x64.
                # We can resize using PIL or scipy or just repeat.
                # Or assume get_random_cifar returns what we want?
                # create_cifar_function_simple returns 32x32.
                # Let's resize manually.
                # import cv2
                # noise_img = cv2.resize(noise_img, (64, 64), interpolation=cv2.INTER_NEAREST)
                
                # Use PIL
                pil_img = Image.fromarray(noise_img)
                pil_img = pil_img.resize((64, 64), Image.NEAREST)
                noise_img = np.array(pil_img)
                
                # Check channel first/last. noise_img is (64, 64, 3).
                # RenderWrapper returns (64, 64, 3) (from ResizeObservation).
                # So it matches.
                obs = noise_img
                
                # Maybe zero reward if watching TV? Or keep env reward (which might be low/random)?
                # Usually watching TV gives 0 extrinsic reward.
                reward = 0.0
            
            info["noisy"] = is_noise
            return obs, reward, terminated, truncated, info

    # Simple env factory
    def make_env_simple(seed, env_name, noisy=False):
        env = None
        trigger_thresh = None 
        trigger_prob = 0.01

        if env_name == "Craftium":
            # Default to OpenWorld if just "Craftium" is specified
            env = gym.make("Craftium/OpenWorld-v0", render_mode="rgb_array")
        elif env_name.startswith("Craftium/"):
            # Allow specific Craftium tasks (e.g., Craftium/ChopTree-v0)
            env = gym.make(env_name, render_mode="rgb_array")
        elif env_name == "MountainCarContinuous-v0":
             env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
             trigger_thresh = 0.5
        elif env_name == "Pendulum-v1":
             env = gym.make("Pendulum-v1", render_mode="rgb_array")
             trigger_thresh = 1.0
             trigger_prob = 0.05
        elif env_name == "MNIST":
            env = MnistEnv()
            trigger_prob = 0.1
        elif "ALE/" in env_name or "NoFrameskip" in env_name or "Breakout" in env_name:
            import ale_py
            gym.register_envs(ale_py) # Just in case, or just importing is enough usually
            
            # Atari Environment Handling
            # Use 'ALE/Breakout-v5' or similar. 
            # We use AtariPreprocessing for standard behavior (FrameSkip, Resize, etc.)
            env = gym.make(env_name, render_mode="rgb_array", frameskip=1) # frameskip handled by wrapper
            
            from gymnasium.wrappers import AtariPreprocessing, TransformReward
            
            # AtariPreprocessing handles:
            # - NoopReset
            # - FrameSkip (default 4)
            # - Resize (default 84x84 -> we can set 64x64 or resize later)
            # - Grayscale (default True)
            # - Scale (default False)
            
            # We want 64x64 RGB for consistency with Dreamer existing config
            env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=64, terminal_on_life_loss=False, grayscale_obs=False, scale_obs=False)
            
            # If grayscale_obs=True, outputis (64, 64). We need (64, 64, 3) or (H, W, 3).
            # If grayscale_obs=False, output is (64, 64, 3).
            
            # Clip rewards for stability
            env = TransformReward(env, lambda r: np.sign(r))
            
            trigger_prob = 0.01
        else:
             try:
                env = gym.make(env_name, render_mode="rgb_array")
             except:
                print(f"Could not make environment {env_name}, defaulting to Pendulum")
                env = gym.make("Pendulum-v1", render_mode="rgb_array")
        
        # Common Wrappers
        if env_name != "Crafter" and not isinstance(env, MnistEnv):
            # MnistEnv already outputs 64x64x3
            # Ensure we resize to 64x64 for Dreamer
            from gymnasium.wrappers import ResizeObservation
            env = ResizeObservation(env, (64, 64))
            
            # Ensure RGB
            # (Some Atari envs or wrappers might output grayscale, checking shape might be needed)
        
        if env_name != "Crafter" and "ALE/" not in env_name and "NoFrameskip" not in env_name and "Breakout" not in env_name:
             env = RenderWrapper(env)

        if noisy:
             get_cifar = create_cifar_function_simple()
             if isinstance(env.action_space, gym.spaces.Box):
                 if trigger_thresh is None: trigger_thresh = 0.0 # Default
                 env = NoisyTVWrapperContinuous(env, get_cifar, trigger_threshold=trigger_thresh)
             else:
                 env = NoisyTVWrapperDiscrete(env, get_cifar, trigger_prob=trigger_prob)
        return env

    env = make_env_simple(args.seed, args.env_name, args.noisy_tv)
    atexit.register(env.close)
    
    # Video Recording for Evaluation
    eval_env = make_env_simple(args.seed + 100, args.env_name, args.noisy_tv)
    atexit.register(eval_env.close)
    # Manual video saving in evaluation loop
    if not os.path.exists("videos"):
        os.makedirs("videos")
    
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    # Replay Buffer
    replay_buffer = ReplayBuffer(
        capacity=cfg.buffer_size,
        observation_shape=(64, 64, 3), # Assuming 3 channels
        action_dim=action_dim
    )
    
    # Models
    rssm = RSSM(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes, action_dim).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder(cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    reward_model = RewardModel(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    if isinstance(env.action_space, gym.spaces.Discrete):
         actor = DiscreteActor(action_dim, cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    else:
         actor = Actor(action_dim, cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    target_critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    target_critic.load_state_dict(critic.state_dict())
    
    # Error Predictor (LPM)
    error_predictor = ErrorPredictor(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes, action_dim).to(device)
    
    agent = Agent(encoder, decoder, rssm, actor, error_predictor).to(device)
    
    # Optimizers
    wm_params = list(rssm.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + list(reward_model.parameters()) + list(error_predictor.parameters())
    wm_optimizer = torch.optim.Adam(wm_params, lr=cfg.model_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    
    # Training Loop
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    
    total_episode = 1
    total_reward = []
    
    # Pre-fill buffer
    print("Pre-filling buffer...")
    for _ in range(cfg.seed_iter):
        
        # Pendulum action is Box(-2, 2).
        # But Dreamer implementation (Student Code) expects Discrete actions (One-Hot)?
        # RSSM init: actino_dim
        # Student code RSSM: self.transition_hidden = nn.Linear(state_dim * num_classes + action_dim, mlp_hidden_dim)
        # So it expects vector input.
        
        # If discrete: action_dim = n. One hot vector size n.
        # If continuous: action_dim = 1 (for Pendulum). Value is float.
        # Student code seems agnostic?
        
        # Check Actor in student_code:
        # self.mean = nn.Linear(hidden_dim, action_dim)
        # action_dist = td.Independent(TruncNormalDist(mean, stddev, -1, 1), 1)
        # It assumes CONTINUOUS actions (TruncNormalDist outputs float).
        
        # My previous assumption about Ale/Breakout (Discrete) was WRONG for this Actor implementation!
        # The Student Code Actor output TruncatedNormal, which is for continuous actions!
        # So Pendulum is actually BETTER than Breakout.
        
        # Pendulum action space is Box(1,).
        
        action = env.action_space.sample() # (1,) float
        # Need to ensure it matches action_dim
        
        next_obs, reward, done, truncated, _ = env.step(action)
        done_flag = done or truncated
        
        replay_buffer.push(preprocess_obs(obs), action, reward, done_flag)
        obs = next_obs
        if done_flag:
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0]
            
    # CSV Logging Setup
    lpm_log_file = "lpm_stats.csv"
    with open(lpm_log_file, "w") as f:
        f.write("step,actual_error,pred_error,intr_reward,is_noisy\n")

    # Main Loop
    print("Starting Main Loop...")
    
    # Progress Bar Wrapper
    pbar = tqdm(range(cfg.iter), desc="Training Steps", unit="step")
    for iteration in pbar:
        
        # Interaction with LPM Reward Calculation
        with torch.no_grad():
            # agent.__call__ handles normalization internally for inference
            # We want to access stored state/rnn, so we call agent logic manually or use helper?
            # Or just use agent(obs) and assume it sets last_state correctly (yes we modified it).
            
            action, pred_obs_normalized = agent(obs, eval=False)
            # action is one-hot vector (B, action_dim) or (1, action_dim)?
            # agent returns action.squeeze().cpu().numpy(). 
            # If B=1, it might just be (action_dim,).
            
            # agent returns action.squeeze().cpu().numpy().
            
            # Action handling
            if isinstance(env.action_space, gym.spaces.Discrete):
                 # One-Hot to Scalar
                 if action.ndim > 1: # (B, A) -> (B,)
                      action_scalar = np.argmax(action, axis=1)[0]
                 else:
                      action_scalar = np.argmax(action)
                 
                 # Prepare for step
                 env_action = action_scalar
            elif isinstance(env.action_space, gym.spaces.Box):
                 # Ensure action is distinct array (1,) for Pendulum, not scalar
                 if action.ndim == 0:
                     action = np.expand_dims(action, axis=0)
                 env_action = action
            
            # For interaction
            next_obs, reward, done, truncated, info = env.step(env_action)
            done_flag = done or truncated
            
            # --- LPM (Learning Progress Motivation) Part ---
            # 内発的報酬(Intrinsic Reward)の計算
            # 数式: R_int = eta * (Predicted_Error - Actual_Error)
            
            # 1. 実際の誤差 (Actual Error) の計算
            # 次の画像(next_obs)がどれくらい予測しにくかったか？
            # 注意: DreamerのDecoderは「現在の状態」から「現在の画像」を復元するものだが、
            # ここでは「次の状態」を予測して「次の画像」と比較する必要がある。
            
            next_obs_normalized = preprocess_obs(next_obs)
            t_last_state = agent.last_state # z_t (学習済みモデルで推論した確率的状態)
            t_last_rnn = agent.last_rnn_hidden # h_t (学習済みモデルで推論した決定論的状態)
            
            if t_last_state is not None:
                # ActionをTensorに変換
                t_action = torch.as_tensor(action, device=device).unsqueeze(0) # (1, action_dim)
                
                # (a) 未来の状態を予測 (Step Forward)
                # h_{t+1} = f(h_t, z_t, a_t)
                t_next_rnn = rssm.recurrent(t_last_state, t_action, t_last_rnn)
                # p(z_{t+1} | h_{t+1})
                t_next_prior = rssm.get_prior(t_next_rnn)
                t_next_state = t_next_prior.mean.flatten(1) # 決定論的な予測(mean)を使用
                
                # (b) 未来の画像を予測 (Imagine Image)
                # \hat{x}_{t+1} ~ p(x | h_{t+1}, z_{t+1})
                t_next_obs_dist = decoder(t_next_state, t_next_rnn)
                t_next_obs_pred = t_next_obs_dist.mean.squeeze().cpu().numpy() # (3, 64, 64)
                t_next_obs_pred = t_next_obs_pred.transpose(1, 2, 0) # (64, 64, 3)
                
                # (c) 実測値との誤差 (MSE)
                # Normalized next_obs [-0.5, 0.5] vs Predicted [-0.5, 0.5]
                actual_error = np.mean((next_obs_normalized - t_next_obs_pred)**2)
                
                # 2. 予測された誤差 (Predicted Error) の計算
                # ErrorPredictorを使って「どれくらい誤差が出そうか」を事前予測
                # Input: z_t, h_t, a_t
                pred_error_est = error_predictor(t_last_state, t_last_rnn, t_action).item()
                
                # 3. 好奇心報酬 (Intrinsic Reward)
                # 「予測誤差が大きい(難しい)」かつ「それが予想通り(予測可能)」なら報酬は小。
                # 「予測誤差が大きい(難しい)」のに「予想では小さいと思っていた(未知の発見)」なら報酬は大...ではない。
                # LPMの定義: 
                # 原論文(Oudeyer et al.): Learning Progress = Error(t-1) - Error(t) (誤差の減少量)
                # この実装: (Predicted Error - Actual Error)
                #  差分が大きい = 「思っていたより誤差が小さかった」= 「学習が進んだ/上手くできた」場合などに正の報酬？
                #  逆に、Noisy-TVのようなランダムなものは、常にActual Errorが大きく、Predicted Errorもそれに追従して大きくなるため、
                #  差分（学習の進捗）はゼロに近づく、というロジック。
                
                # Fix: Error Predictor predicts Log(MSE), so we must compare with Log(Actual MSE)
                actual_error_log = np.log(actual_error + 1e-6)
                intr_reward = cfg.lpm_eta * (pred_error_est - actual_error_log)
                
                # 安定化のためのクリッピング
                intr_reward = max(-1.0, min(1.0, intr_reward))
                
                total_reward_val = reward + intr_reward
                
                # print(f"LPM: ActErr={actual_error:.4f} PredErr={pred_error_est:.4f} Intr={intr_reward:.4f}")
                
                # Log LPM stats
                is_noisy_step = info.get("noisy", False)
                with open(lpm_log_file, "a") as f:
                    f.write(f"{iteration},{actual_error},{pred_error_est},{intr_reward},{is_noisy_step}\n")

                if args.wandb:
                    # Log detailed LPM metrics every step or frequent enough? 
                    # Logging every step is fine for debugging/analysis if run is not too long.
                    # Separate metrics for Noisy vs Clean
                    metrics = {
                        "LPM/Actual_Error": actual_error,
                        "LPM/Pred_Error": pred_error_est,
                        "LPM/Intrinsic_Reward": intr_reward,
                    }
                    if is_noisy_step:
                        metrics["LPM/Intrinsic_Reward_Noisy"] = intr_reward
                        metrics["LPM/Actual_Error_Noisy"] = actual_error
                    else:
                        metrics["LPM/Intrinsic_Reward_Clean"] = intr_reward
                        metrics["LPM/Actual_Error_Clean"] = actual_error
                    
                    wandb.log(metrics, step=iteration)
            else:
                total_reward_val = reward
            
            # Store in buffer
            # action is one-hot
            replay_buffer.push(preprocess_obs(obs), action, total_reward_val, done_flag)
            
            obs = next_obs
            total_reward.append(reward) # Track external reward for logging
            
            if done_flag:
                obs = env.reset()
                if isinstance(obs, tuple): obs = obs[0]
                agent.reset()
                
                print(f"Episode {total_episode} ExtReward: {np.mean(total_reward):.4f}")
                
                if args.wandb:
                    wandb.log({
                        "Episode/Extrinsic_Reward": np.mean(total_reward),
                        "Episode/Steps": iteration,
                        "Episode/ID": total_episode
                    })
                
                total_reward = []
                total_episode += 1
                
        if (iteration + 1) % cfg.eval_interval == 0:
            try:
                evaluation(eval_env, agent, iteration, cfg)
                eval_env.reset()
                agent.reset()
            except Exception as e:
                print(f"Evaluation failed at iteration {iteration}: {e}")
                print("Re-creating evaluation environment...")
                try:
                    eval_env.close()
                except:
                    pass
                try:
                    eval_env = make_env_simple(args.seed + 100 + iteration, args.env_name, args.noisy_tv)
                    print("Evaluation environment re-created successfully.")
                except Exception as create_e:
                    print(f"Failed to re-create evaluation environment: {create_e}")

        if (iteration + 1) % cfg.update_freq == 0:
            # Training
            observations, actions, rewards, done_flags = replay_buffer.sample(cfg.batch_size, cfg.seq_length)
            done_flags = 1 - done_flags
            
            observations = torch.permute(torch.as_tensor(observations, device=device), (1, 0, 4, 2, 3))
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)
            done_flags = torch.as_tensor(done_flags, device=device).transpose(0, 1).float()
            
            # --- World Model Update with ErrorPredictor ---
            emb_observations = encoder(observations.reshape(-1, 3, 64, 64)).view(cfg.seq_length, cfg.batch_size, -1)
            
            state = torch.zeros(cfg.batch_size, cfg.state_dim*cfg.num_classes, device=device)
            rnn_hidden = torch.zeros(cfg.batch_size, cfg.rnn_hidden_dim, device=device)
            
            states = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.state_dim*cfg.num_classes, device=device)
            rnn_hiddens = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.rnn_hidden_dim, device=device)
            
            kl_loss = 0
            ep_loss = 0 # Error Predictor Loss
            
            for i in range(cfg.seq_length-1):
                rnn_hidden = rssm.recurrent(state, actions[i], rnn_hidden)
                
                next_state_prior, next_detach_prior = rssm.get_prior(rnn_hidden, detach=True)
                next_state_posterior, next_detach_posterior = rssm.get_posterior(rnn_hidden, emb_observations[i+1], detach=True)
                
                state = next_state_posterior.rsample().flatten(1)
                rnn_hiddens[i+1] = rnn_hidden
                states[i+1] = state
                
                kl_loss += cfg.kl_balance * torch.mean(kl_divergence(next_detach_posterior, next_state_prior)) + \
                           (1 - cfg.kl_balance) * torch.mean(kl_divergence(next_state_posterior, next_detach_prior))
                
                # --- Error Predictor Training ---
                # Error Predictorの学習
                # 目的: 「次の状態での予測誤差(Actual Error)」を正しく予測できるようにする。
                # 教師データ(Target): 実際に観測された誤差
                # 入力(Input): z_i, h_i, a_i
                
                pred_next_state = next_state_prior.rsample().flatten(1) # z_{i+1} from prior
                pred_next_state_mean = next_state_prior.mean.flatten(1)
                
                # (1) モデルによる未来画像の予測
                pred_next_obs_dist = decoder(pred_next_state_mean, rnn_hidden)
                pred_next_obs = pred_next_obs_dist.mean.view(cfg.batch_size, 3, 64, 64)
                
                # (2) 実際の未来画像(observation[i+1])との比較 -> Actual Error算出
                real_next_obs = observations[i+1] # (B, 3, 64, 64)
                actual_error_batch = ((real_next_obs - pred_next_obs) ** 2).mean(dim=[1,2,3]) # (B,)
                
                if i > 0:
                    prev_state = states[i] # z_i
                    prev_rnn = rnn_hiddens[i] # h_i
                    
                    # (3) ErrorPredictorによる予測 (Predicted Error)
                    pred_error_hat = error_predictor(prev_state.detach(), prev_rnn.detach(), actions[i]).squeeze()
                    
                    # (4) 損失関数の計算
                    # ターゲットはActual Errorの対数(log)をとることで、スケールの違いを緩和することが多い
                    target = torch.log(actual_error_batch.detach() + 1e-6)
                    ep_loss += F.mse_loss(pred_error_hat, target)
                           
            kl_loss /= (cfg.seq_length - 1)
            ep_loss /= (cfg.seq_length - 1)
            
            rnn_hiddens = rnn_hiddens[1:]
            states = states[1:]
            
            flatten_rnn_hiddens = rnn_hiddens.view(-1, cfg.rnn_hidden_dim)
            flatten_states = states.view(-1, cfg.state_dim * cfg.num_classes)
            
            obs_dist = decoder(flatten_states, flatten_rnn_hiddens)
            reward_dist = reward_model(flatten_states, flatten_rnn_hiddens)
            
            C, H, W = observations.shape[2:]
            obs_loss = -torch.mean(obs_dist.log_prob(observations[1:].reshape(-1, C, H, W)))
            reward_loss = -torch.mean(reward_dist.log_prob(rewards[:-1].reshape(-1, 1)))
            
            wm_loss = obs_loss + cfg.reward_loss_scale * reward_loss + cfg.kl_scale * kl_loss + ep_loss
            
            wm_optimizer.zero_grad()
            wm_loss.backward()
            clip_grad_norm_(wm_params, cfg.gradient_clipping)
            wm_optimizer.step()
            
            if args.wandb:
                wandb.log({
                    "Train/WM_Loss": wm_loss.item(),
                    "Train/Obs_Loss": obs_loss.item(),
                    "Train/Reward_Loss": reward_loss.item(),
                    "Train/KL_Loss": kl_loss.item(),
                    "Train/EP_Loss": ep_loss.item(), # Error Predictor Loss
                    "Train/Reward_Mean": rewards.mean().item(),
                    # Let's just log loss for now
                }, step=iteration)

            # Update progress bar description
            pbar.set_postfix({
                 "total_r": f"{np.mean(total_reward[-10:]):.1f}" if total_reward else "0.0",
                 "wm_loss": f"{wm_loss.item():.2f}"
            })
            
            # --- Actor Critic Update ---
            flatten_rnn_hiddens = flatten_rnn_hiddens.detach()
            flatten_states = flatten_states.detach()
            
            imagined_states = torch.zeros(cfg.imagination_horizon + 1, *flatten_states.shape, device=device)
            imagined_rnn_hiddens = torch.zeros(cfg.imagination_horizon + 1, *flatten_rnn_hiddens.shape, device=device)
            imagined_action_log_probs = torch.zeros((cfg.imagination_horizon, cfg.batch_size * (cfg.seq_length-1)), device=device)
            imagined_action_entropys = torch.zeros((cfg.imagination_horizon, cfg.batch_size * (cfg.seq_length-1)), device=device)
            
            imagined_states[0] = flatten_states
            imagined_rnn_hiddens[0] = flatten_rnn_hiddens
            
            for i in range(1, cfg.imagination_horizon + 1):
                i_actions, i_action_log_probs, i_action_entropys = actor(flatten_states, flatten_rnn_hiddens)
                
                flatten_rnn_hiddens = rssm.recurrent(flatten_states, i_actions, flatten_rnn_hiddens)
                flatten_states_prior = rssm.get_prior(flatten_rnn_hiddens)
                flatten_states = flatten_states_prior.rsample().flatten(1)
                
                imagined_rnn_hiddens[i] = flatten_rnn_hiddens
                imagined_states[i] = flatten_states
                imagined_action_log_probs[i-1] = i_action_log_probs
                imagined_action_entropys[i-1] = i_action_entropys
            
            imagined_states = imagined_states[1:]
            imagined_rnn_hiddens = imagined_rnn_hiddens[1:]
            
            flatten_imagined_states = imagined_states.view(-1, cfg.state_dim * cfg.num_classes)
            flatten_imagined_rnn_hiddens = imagined_rnn_hiddens.view(-1, cfg.rnn_hidden_dim)
            
            imagined_rewards = reward_model(flatten_imagined_states, flatten_imagined_rnn_hiddens).mean.view(cfg.imagination_horizon, -1)
            target_values = target_critic(flatten_imagined_states, flatten_imagined_rnn_hiddens).view(cfg.imagination_horizon, -1).detach()
            
            discount_arr = (cfg.discount * torch.ones_like(imagined_rewards)).to(device)
            initial_done = done_flags[1:].reshape(1, -1) 
            # Note: dimension mismatch handle omitted for brevity, reusing done_flags might be tricky for imagination.
            # Assuming discount model handles termination or we just use constant discount.
            
            lambda_target = calculate_lambda_target(imagined_rewards, discount_arr, target_values, cfg.lambda_)
            
            weights = torch.cumprod(torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[:-1]], dim=0), dim=0)
            weights[-1] = 0.0
            
            objective = lambda_target + cfg.actor_entropy_scale * imagined_action_entropys
            actor_loss = -(weights * objective).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(actor.parameters(), cfg.gradient_clipping)
            actor_optimizer.step()
            
            value_mean = critic(flatten_imagined_states.detach(), flatten_imagined_rnn_hiddens.detach()).view(cfg.imagination_horizon, -1)
            value_dist = MSE(value_mean)
            critic_loss = -(weights.detach() * value_dist.log_prob(lambda_target.detach())).mean()
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), cfg.gradient_clipping)
            critic_optimizer.step()
            
            if args.wandb:
                 wandb.log({
                     "Train/Actor_Loss": actor_loss.item(),
                     "Train/Critic_Loss": critic_loss.item()
                 }, step=iteration)

            if (iteration + 1) % cfg.slow_critic_update == 0:
                target_critic.load_state_dict(critic.state_dict())
                
    # Save Model
    try:
        # Save Model
        agent.to("cpu")
        torch.save(agent, "agent_lpm.pth")
        print("Model saved to agent_lpm.pth")
    except Exception as e:
        print(f"Error occurring: {e}")
    finally:
        print("Closing environments...")
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()
