import sys
import atexit
import os
import random
import time
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# Import from student_code and exploration
from excute.world_model.world_model_lpm import (
    Agent, RSSM, Encoder, Decoder, RewardModel, DiscountModel,
    Actor, DiscreteActor, Critic, ErrorPredictor, MSE
)
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
        self.action_space = gym.spaces.Discrete(10)  # Dummy action space

        # Load MNIST data
        try:
            self.mnist_data = datasets.MNIST(
                "../data", train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor()
                ])
            )
        except Exception:
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
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        if self.mnist_data:
            img, _ = self.mnist_data[self.current_idx]
            img = img.permute(1, 2, 0).numpy() * 255.0
            img = img.astype(np.uint8)
            if img.shape[2] == 1:
                img = np.concatenate([img] * 3, axis=2)
            return img
        return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    def render(self):
        return self._get_obs()


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
        self.update_freq = 80  # N in the paper (model updating cycle length)

        # lambda return params
        self.discount = 0.995
        self.lambda_ = 0.95

        # learning period settings
        self.iter = 6000
        self.seed_iter = 1000
        self.eval_interval = 5000
        self.eval_freq = 5
        self.eval_episodes = 5

        # LPM Params (paper-style)
        self.lpm_eta = 1.0
        self.lpm_lr = 2e-4

        # Error replay queue D (paper-style)
        self.D_size = 5000           # d in the paper
        self.D_batch_size = 256      # minibatch size for training error model g_phi

        # Override with kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


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
                cross_border = np.logical_and(initial_index <= episode_borders, episode_borders < final_index).any()
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:]
        )
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1]
        )
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1
        )
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1
        )
        return sampled_observations, sampled_actions, sampled_rewards, sampled_done

    def __len__(self):
        return self.capacity if self.is_filled else self.index


def preprocess_obs(obs):
    obs = obs.astype(np.float32)
    return obs / 255.0 - 0.5


def calculate_lambda_target(rewards, discounts, values, lambda_):
    V_lambda = torch.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            V_lambda[t] = rewards[t] + discounts[t] * values[t]
        else:
            V_lambda[t] = rewards[t] + discounts[t] * ((1 - lambda_) * values[t + 1] + lambda_ * V_lambda[t + 1])
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

    os.makedirs("eval_view/video", exist_ok=True)
    os.makedirs("eval_view/images", exist_ok=True)

    with torch.no_grad():
        for i in range(cfg.eval_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            policy.reset()

            done = False
            truncated = False
            episode_reward = []
            frames = []
            recon_frames = []

            while not done and not truncated:
                action, recon_img = policy(obs, eval=True)

                if isinstance(env.action_space, gym.spaces.Box):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, reward, done, truncated, info = env.step(action)
                elif hasattr(env.action_space, "n"):
                    action_scalar = int(np.argmax(action))
                    obs, reward, done, truncated, info = env.step(action_scalar)
                else:
                    obs, reward, done, truncated, info = env.step(action)

                frame = env.render()
                frames.append(frame)

                if recon_img is not None:
                    recon_frame = (recon_img * 255.0).astype(np.uint8)
                    recon_frames.append(recon_frame)

                episode_reward.append(reward)

            if i == 0 and len(frames) > 0:
                video_path = f"eval_view/video/eval_iter_{step}_ep_{i}.mp4"
                try:
                    frames = [np.array(f, dtype=np.uint8) for f in frames if f is not None]
                    combined_frames = None

                    if len(recon_frames) > 0 and len(recon_frames) == len(frames):
                        combined_frames = []
                        for f, r in zip(frames, recon_frames):
                            f_pil = Image.fromarray(f)
                            r_pil = Image.fromarray(r)
                            f_s = f_pil.resize((64, 64))
                            comb = Image.new("RGB", (128, 64))
                            comb.paste(f_s, (0, 0))
                            comb.paste(r_pil, (64, 0))
                            combined_frames.append(np.array(comb))
                        imageio.mimsave(video_path, combined_frames, fps=30)
                        print(f"Saved combined video to {video_path}")
                    else:
                        imageio.mimsave(video_path, frames, fps=30)
                        print(f"Saved video to {video_path}")

                    if combined_frames is not None and len(combined_frames) > 0:
                        indices = [0, len(combined_frames) // 2, len(combined_frames) - 1]
                        for idx in indices:
                            img_path = f"eval_view/images/eval_iter_{step}_ep_{i}_frame_{idx}.png"
                            Image.fromarray(combined_frames[idx]).save(img_path)

                except Exception as e:
                    print(f"Failed to save video: {e}")

            if len(episode_reward) < 500:
                pad_val = 10 if info.get("success", False) else 0
                episode_reward = np.pad(
                    episode_reward, (0, 500 - len(episode_reward)),
                    "constant", constant_values=pad_val
                )

            all_ep_rewards.append(np.mean(episode_reward))

        mean_ep_rewards = np.mean(all_ep_rewards)
        max_ep_rewards = np.max(all_ep_rewards)
        print(f"Eval(iter={step}) mean: {mean_ep_rewards:.4f} max: {max_ep_rewards:.4f}")

    return mean_ep_rewards


def one_hot(n: int, idx: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[idx] = 1.0
    return v


def ensure_action_vector(action, env, action_dim: int) -> np.ndarray:
    """
    Returns an action *vector* suitable for the replay buffer:
    - Discrete env: one-hot vector length action_dim
    - Box env: float vector length action_dim
    """
    if isinstance(env.action_space, gym.spaces.Discrete):
        # action might already be one-hot (from agent)
        a = np.array(action, dtype=np.float32)
        if a.shape == (action_dim,):
            return a
        # if scalar:
        return one_hot(action_dim, int(a))
    else:
        a = np.array(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != action_dim:
            # best effort: pad/trim
            if a.shape[0] < action_dim:
                a = np.pad(a, (0, action_dim - a.shape[0]), mode="constant")
            else:
                a = a[:action_dim]
        return a


class RenderWrapper(gym.Wrapper):
    """
    Replaces the original observation with env.render() output (RGB image).
    Useful for non-pixel classic control tasks (e.g., Pendulum) to run pixel-based agents.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(500, 500, 3), dtype=np.uint8)
        self.metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        obs = self.env.render()
        return obs, info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self.env.render()
        return obs, reward, terminated, truncated, info


class NoisyTVWrapperDiscrete(gym.Wrapper):
    def __init__(self, env, get_random_cifar_fn, trigger_prob=0.05):
        super().__init__(env)
        self.get_random_cifar = get_random_cifar_fn
        self.trigger_prob = trigger_prob

    def step(self, action):
        is_noise = random.random() < self.trigger_prob
        obs, reward, terminated, truncated, info = self.env.step(action)

        if is_noise:
            noise_img = self.get_random_cifar()
            pil_img = Image.fromarray(noise_img).resize((64, 64), Image.NEAREST)
            obs = np.array(pil_img)
            reward = 0.0

        info["noisy"] = is_noise
        return obs, reward, terminated, truncated, info


class NoisyTVWrapperContinuous(gym.Wrapper):
    def __init__(self, env, get_random_cifar_fn, trigger_threshold=1.5):
        super().__init__(env)
        self.get_random_cifar = get_random_cifar_fn
        self.trigger_threshold = trigger_threshold

    def step(self, action):
        is_noise = np.any(np.abs(action) > self.trigger_threshold)
        obs, reward, terminated, truncated, info = self.env.step(action)

        if is_noise:
            noise_img = self.get_random_cifar()
            pil_img = Image.fromarray(noise_img).resize((64, 64), Image.NEAREST)
            obs = np.array(pil_img)
            reward = 0.0

        info["noisy"] = is_noise
        return obs, reward, terminated, truncated, info


def make_env_simple(seed, env_name, noisy=False):
    env = None
    trigger_thresh = None
    trigger_prob = 0.01

    if env_name == "Craftium":
        env = gym.make("Craftium/OpenWorld-v0", render_mode="rgb_array")
    elif env_name.startswith("Craftium/"):
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
        gym.register_envs(ale_py)

        env = gym.make(env_name, render_mode="rgb_array", frameskip=1)
        from gymnasium.wrappers import AtariPreprocessing, TransformReward

        env = AtariPreprocessing(
            env, noop_max=30, frame_skip=4, screen_size=64,
            terminal_on_life_loss=False, grayscale_obs=False, scale_obs=False
        )
        env = TransformReward(env, lambda r: np.sign(r))
        trigger_prob = 0.01
    else:
        try:
            env = gym.make(env_name, render_mode="rgb_array")
        except Exception:
            print(f"Could not make environment {env_name}, defaulting to Pendulum")
            env = gym.make("Pendulum-v1", render_mode="rgb_array")

    # Common wrappers: resize to 64x64 for Dreamer
    if env_name != "Crafter" and not isinstance(env, MnistEnv):
        from gymnasium.wrappers import ResizeObservation
        env = ResizeObservation(env, (64, 64))

    # For non-Atari, turn into pixel observation via render()
    if env_name != "Crafter" and "ALE/" not in env_name and "NoFrameskip" not in env_name and "Breakout" not in env_name:
        env = RenderWrapper(env)

    if noisy:
        get_cifar = create_cifar_function_simple()
        if isinstance(env.action_space, gym.spaces.Discrete):
            env = NoisyTVEnvWrapperCIFAR(env, get_cifar, num_random_actions=2)
        else:
            env = NoisyTVWrapperContinuous(...)  # 連続は別途設計が必要なら残す

    env.reset(seed=seed)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Pendulum-v1")
    parser.add_argument("--noisy-tv", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--wandb", action="store_true", help="Use WandB logging")
    parser.add_argument("--wandb-project", default="Dreamer-LPM", help="WandB project name")
    parser.add_argument("--wandb-entity", default=None, help="WandB entity")
    parser.add_argument("--wandb-run-name", default=None, help="Run name")
    args = parser.parse_args()

    cfg = Config()
    cfg.iter = args.steps

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

    env = make_env_simple(args.seed, args.env_name, args.noisy_tv)
    atexit.register(env.close)

    eval_env = make_env_simple(args.seed + 100, args.env_name, args.noisy_tv)
    atexit.register(eval_env.close)

    if not os.path.exists("videos"):
        os.makedirs("videos")

    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    replay_buffer = ReplayBuffer(
        capacity=cfg.buffer_size,
        observation_shape=(64, 64, 3),
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

    # Error model g_phi (paper-style)
    error_predictor = ErrorPredictor(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes, action_dim).to(device)

    agent = Agent(encoder, decoder, rssm, actor, error_predictor).to(device)

    # Optimizers: split WM and error model (closer to paper: separate buffers B and D)
    wm_params = list(rssm.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + list(reward_model.parameters())
    wm_optimizer = torch.optim.Adam(wm_params, lr=cfg.model_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    ep_optimizer = torch.optim.Adam(error_predictor.parameters(), lr=cfg.lpm_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)

    # Paper-style error replay queue D:
    # stores (z_t, h_t, a_t, eps_t) where eps_t is log MSE under the *current* dynamics model.
    # After model update, we train g_phi on the collected eps (which correspond to previous model version),
    # then clear D to start the next update cycle.
    D_queue = deque(maxlen=cfg.D_size)

    # Training Loop
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    total_episode = 1
    total_reward = []

    # Pre-fill buffer (seed episodes)
    print("Pre-filling buffer...")
    for _ in range(cfg.seed_iter):
        if isinstance(env.action_space, gym.spaces.Discrete):
            a_scalar = env.action_space.sample()
            a_vec = one_hot(action_dim, int(a_scalar))
            env_action = int(a_scalar)
        else:
            env_action = env.action_space.sample()
            a_vec = ensure_action_vector(env_action, env, action_dim)

        next_obs, reward, done, truncated, _ = env.step(env_action)
        done_flag = done or truncated

        replay_buffer.push(preprocess_obs(obs), a_vec, reward, done_flag)

        obs = next_obs
        if done_flag:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

    # CSV Logging Setup
    lpm_log_file = "lpm_stats.csv"
    with open(lpm_log_file, "w") as f:
        f.write("step,eps_current_logmse,pred_prev_eps,intr_reward,is_noisy\n")

    print("Starting Main Loop...")
    pbar = tqdm(range(cfg.iter), desc="Training Steps", unit="step")

    for iteration in pbar:
        # --- Interaction & intrinsic reward (paper-style) ---
        with torch.no_grad():
            action, _pred_obs_normalized = agent(obs, eval=False)

            # Convert agent action -> env action, and action vector for buffer
            if isinstance(env.action_space, gym.spaces.Discrete):
                # agent action is expected one-hot-like
                a_vec = ensure_action_vector(action, env, action_dim)
                env_action = int(np.argmax(a_vec))
            else:
                # continuous
                a_vec = ensure_action_vector(action, env, action_dim)
                env_action = a_vec  # Box expects vector

            next_obs, reward, done, truncated, info = env.step(env_action)
            done_flag = done or truncated

            total_reward_val = reward  # will add intrinsic below if available

            # Compute eps_current = log MSE(o_{t+1}, f^{(tau)}(o_t,a_t)) using current world model (RSSM+Decoder)
            next_obs_normalized = preprocess_obs(next_obs)
            t_last_state = agent.last_state
            t_last_rnn = agent.last_rnn_hidden

            if t_last_state is not None:
                # Prepare action tensor (1, action_dim)
                t_action = torch.as_tensor(a_vec, device=device).unsqueeze(0)

                # Predict next latent under current model
                t_next_rnn = rssm.recurrent(t_last_state, t_action, t_last_rnn)
                t_next_prior = rssm.get_prior(t_next_rnn)
                t_next_state = t_next_prior.mean.flatten(1)

                # Predict next observation
                t_next_obs_dist = decoder(t_next_state, t_next_rnn)
                t_next_obs_pred = t_next_obs_dist.mean.squeeze().cpu().numpy()  # (3,64,64)
                t_next_obs_pred = t_next_obs_pred.transpose(1, 2, 0)           # (64,64,3)

                mse = np.mean((next_obs_normalized - t_next_obs_pred) ** 2)
                eps_current = float(np.log(mse + 1e-6))

                # Paper-style: g_phi predicts expected error of previous model iteration.
                pred_prev_eps = float(error_predictor(t_last_state, t_last_rnn, t_action).item())

                intr_reward = cfg.lpm_eta * (pred_prev_eps - eps_current)
                intr_reward = float(np.clip(intr_reward, -1.0, 1.0))
                total_reward_val = reward + intr_reward

                # Store (z,h,a,eps_current) into D (to train g_phi at next update)
                # Store on CPU to keep memory stable.
                D_queue.append((
                    t_last_state.detach().cpu(),
                    t_last_rnn.detach().cpu(),
                    torch.as_tensor(a_vec).detach().cpu(),
                    eps_current
                ))

                is_noisy_step = info.get("noisy", False)
                with open(lpm_log_file, "a") as f:
                    f.write(f"{iteration},{eps_current},{pred_prev_eps},{intr_reward},{is_noisy_step}\n")

                if args.wandb:
                    metrics = {
                        "LPM/eps_current_logmse": eps_current,
                        "LPM/pred_prev_eps": pred_prev_eps,
                        "LPM/intrinsic_reward": intr_reward,
                    }
                    if is_noisy_step:
                        metrics["LPM/intrinsic_reward_noisy"] = intr_reward
                        metrics["LPM/eps_current_noisy"] = eps_current
                    else:
                        metrics["LPM/intrinsic_reward_clean"] = intr_reward
                        metrics["LPM/eps_current_clean"] = eps_current
                    wandb.log(metrics, step=iteration)

            replay_buffer.push(preprocess_obs(obs), a_vec, total_reward_val, done_flag)

            obs = next_obs
            total_reward.append(reward)  # external reward tracking

            if done_flag:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                agent.reset()

                print(f"Episode {total_episode} ExtReward: {np.mean(total_reward):.4f}")
                if args.wandb:
                    wandb.log({
                        "Episode/Extrinsic_Reward": np.mean(total_reward),
                        "Episode/Steps": iteration,
                        "Episode/ID": total_episode
                    }, step=iteration)

                total_reward = []
                total_episode += 1

        # --- Evaluation ---
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
                except Exception:
                    pass
                try:
                    eval_env = make_env_simple(args.seed + 100 + iteration, args.env_name, args.noisy_tv)
                    print("Evaluation environment re-created successfully.")
                except Exception as create_e:
                    print(f"Failed to re-create evaluation environment: {create_e}")

        # --- Periodic model update (τ step) ---
        if (iteration + 1) % cfg.update_freq == 0:
            # Sample from replay buffer B to train dynamics (world model)
            observations, actions, rewards, done_flags = replay_buffer.sample(cfg.batch_size, cfg.seq_length)
            done_flags = 1 - done_flags

            observations = torch.permute(torch.as_tensor(observations, device=device), (1, 0, 4, 2, 3))
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)
            done_flags = torch.as_tensor(done_flags, device=device).transpose(0, 1).float()

            # --- World Model Update (NO ep_loss here; paper trains error model using D) ---
            emb_observations = encoder(observations.reshape(-1, 3, 64, 64)).view(cfg.seq_length, cfg.batch_size, -1)

            state = torch.zeros(cfg.batch_size, cfg.state_dim * cfg.num_classes, device=device)
            rnn_hidden = torch.zeros(cfg.batch_size, cfg.rnn_hidden_dim, device=device)

            states = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.state_dim * cfg.num_classes, device=device)
            rnn_hiddens = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.rnn_hidden_dim, device=device)

            kl_loss = 0.0

            for i in range(cfg.seq_length - 1):
                rnn_hidden = rssm.recurrent(state, actions[i], rnn_hidden)

                next_state_prior, next_detach_prior = rssm.get_prior(rnn_hidden, detach=True)
                next_state_posterior, next_detach_posterior = rssm.get_posterior(
                    rnn_hidden, emb_observations[i + 1], detach=True
                )

                state = next_state_posterior.rsample().flatten(1)
                rnn_hiddens[i + 1] = rnn_hidden
                states[i + 1] = state

                kl_loss = kl_loss + (
                    cfg.kl_balance * torch.mean(kl_divergence(next_detach_posterior, next_state_prior)) +
                    (1 - cfg.kl_balance) * torch.mean(kl_divergence(next_state_posterior, next_detach_prior))
                )

            kl_loss = kl_loss / (cfg.seq_length - 1)

            rnn_hiddens = rnn_hiddens[1:]
            states = states[1:]

            flatten_rnn_hiddens = rnn_hiddens.reshape(-1, cfg.rnn_hidden_dim)
            flatten_states = states.reshape(-1, cfg.state_dim * cfg.num_classes)

            obs_dist = decoder(flatten_states, flatten_rnn_hiddens)
            reward_dist = reward_model(flatten_states, flatten_rnn_hiddens)

            C, H, W = observations.shape[2:]
            obs_loss = -torch.mean(obs_dist.log_prob(observations[1:].reshape(-1, C, H, W)))
            reward_loss = -torch.mean(reward_dist.log_prob(rewards[:-1].reshape(-1, 1)))

            wm_loss = obs_loss + cfg.reward_loss_scale * reward_loss + cfg.kl_scale * kl_loss

            wm_optimizer.zero_grad()
            wm_loss.backward()
            clip_grad_norm_(wm_params, cfg.gradient_clipping)
            wm_optimizer.step()

            # --- Paper-style: train error model g_phi on D (errors from previous cycle) ---
            ep_loss_D = None
            if len(D_queue) >= max(4, min(cfg.D_batch_size, len(D_queue))):
                batch_size_D = min(cfg.D_batch_size, len(D_queue))
                idx = np.random.choice(len(D_queue), size=batch_size_D, replace=False)
                batch = [D_queue[i] for i in idx]

                z_batch = torch.cat([b[0] for b in batch], dim=0).to(device)  # each b[0] is (1, zdim) likely
                h_batch = torch.cat([b[1] for b in batch], dim=0).to(device)
                a_batch = torch.stack([b[2] for b in batch], dim=0).to(device).float()
                eps_batch = torch.tensor([b[3] for b in batch], device=device).float()

                pred = error_predictor(z_batch, h_batch, a_batch).squeeze(-1)
                ep_loss_D = F.mse_loss(pred, eps_batch)

                ep_optimizer.zero_grad()
                ep_loss_D.backward()
                clip_grad_norm_(error_predictor.parameters(), cfg.gradient_clipping)
                ep_optimizer.step()

            # Clear D for next cycle (so it always corresponds to ONE model version)
            D_queue.clear()

            if args.wandb:
                log_dict = {
                    "Train/WM_Loss": wm_loss.item(),
                    "Train/Obs_Loss": obs_loss.item(),
                    "Train/Reward_Loss": reward_loss.item(),
                    "Train/KL_Loss": kl_loss.item(),
                    "Train/Reward_Mean": rewards.mean().item(),
                }
                if ep_loss_D is not None:
                    log_dict["Train/EP_Loss_D"] = float(ep_loss_D.item())
                wandb.log(log_dict, step=iteration)

            pbar.set_postfix({
                "ext_r": f"{np.mean(total_reward[-10:]):.2f}" if total_reward else "0.00",
                "wm": f"{wm_loss.item():.2f}",
                "epD": f"{ep_loss_D.item():.2f}" if ep_loss_D is not None else "NA",
            })

            # --- Actor Critic Update (imagination) ---
            flatten_rnn_hiddens_det = flatten_rnn_hiddens.detach()
            flatten_states_det = flatten_states.detach()

            imagined_states = torch.zeros(cfg.imagination_horizon + 1, *flatten_states_det.shape, device=device)
            imagined_rnn_hiddens = torch.zeros(cfg.imagination_horizon + 1, *flatten_rnn_hiddens_det.shape, device=device)
            imagined_action_entropys = torch.zeros((cfg.imagination_horizon, cfg.batch_size * (cfg.seq_length - 1)), device=device)

            imagined_states[0] = flatten_states_det
            imagined_rnn_hiddens[0] = flatten_rnn_hiddens_det

            flat_s = flatten_states_det
            flat_h = flatten_rnn_hiddens_det

            for i in range(1, cfg.imagination_horizon + 1):
                i_actions, i_action_log_probs, i_action_entropys = actor(flat_s, flat_h)
                flat_h = rssm.recurrent(flat_s, i_actions, flat_h)
                flat_prior = rssm.get_prior(flat_h)
                flat_s = flat_prior.rsample().flatten(1)

                imagined_rnn_hiddens[i] = flat_h
                imagined_states[i] = flat_s
                imagined_action_entropys[i - 1] = i_action_entropys

            imagined_states = imagined_states[1:]
            imagined_rnn_hiddens = imagined_rnn_hiddens[1:]

            flatten_imagined_states = imagined_states.reshape(-1, cfg.state_dim * cfg.num_classes)
            flatten_imagined_rnn_hiddens = imagined_rnn_hiddens.reshape(-1, cfg.rnn_hidden_dim)

            imagined_rewards = reward_model(flatten_imagined_states, flatten_imagined_rnn_hiddens).mean.view(cfg.imagination_horizon, -1)
            target_values = target_critic(flatten_imagined_states, flatten_imagined_rnn_hiddens).view(cfg.imagination_horizon, -1).detach()

            discount_arr = (cfg.discount * torch.ones_like(imagined_rewards)).to(device)
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
