"""
1) リプレイバッファには外的報酬だけを保存するように変更
何を変えたか
ReplayBuffer の rewards は 環境からの外的報酬のみを格納。
replay_buffer.push(..., ext_reward, ...) に変更し、内発を足した total_reward_val を保存しない。
なぜ効く
内発報酬はモデル更新や誤差予測器の学習で分布が変わる「非定常」信号なので、これを reward_model の教師にすると世界モデル学習が揺れやすい。
外的報酬だけにすることで、Dreamerの中核（世界モデル→想像→actor/critic）の基盤が安定する。

2) 予測誤差 eps_current を MSE(log) から decoder の NLLに変更
何を変えたか
eps_current = log(MSE) をやめ、eps_current = - log p(o_{t+1} | s_{t+1}, h_{t+1})（NLL）に置換。
なぜ効くか
decoder が分布を出しているので、NLLは世界モデルの学習目的（観測再構成の log_prob）と整合する。
ピクセルMSEより「尖り」が出にくく、ノイズに過敏になりにくいことが多い。

3) 内発信号を正規化してからクリップ（RunningMeanStd）
何を変えたか
delta_raw = pred_prev_eps - eps_current をそのまま使わず、
RunningMeanStd で平均との差を標準化
その後 [-lpm_clip, lpm_clip] にクリップ
クリップ対象は「正規化後の delta」。
なぜ効くか
予測誤差のスケールは学習段階で大きく変わるため、差分も非定常に振れやすい。
正規化でスケールを一定化し、クリップで外れ値の影響を抑えると、actor/critic 更新が暴れにくい。

4) 内発係数 η をウォームアップ（徐々に効かせる）
何を変えたか
eta_current = lpm_eta * min(1, step / lpm_warmup_steps) を導入。
序盤は内発を弱く、学習が進むにつれ本来の強さに近づける。
なぜ効くか
序盤の世界モデルと gφ は未熟で、内発がノイズになりやすい。
先に世界モデルの土台を作ってから探索圧を上げる方が安定しやすい。

5) gφ（ErrorPredictor）が育つまで内発を無効化するゲート（ep_ready）
何を変えたか
ep_ready フラグを追加。
gφ を D_queue で1回以上学習できるまでは、
delta_norm_clipped = 0
intrinsic_eta_scaled = 0
にして内発を実質オフ。
なぜ効くか
gφ がまだ当てにならない状態で pred_prev_eps - eps_current を使うと、ランダムな探索圧になり、学習が壊れやすい。
「内発を使える状態になってから使う」ことで破綻を防ぐ。

6) 内発を imagination（想像ロールアウト）に“載せる”ためのモデルを追加
何を変えたか
IntrinsicRewardModel(s,h,a) -> r_int_hat を新設。
D_queue に保存した教師 delta_norm_clipped を使って、このモデルを回帰学習。
想像中の報酬を
r_total_hat = r_ext_hat + eta_current * r_int_hat
として actor/critic の λ-return を計算。
なぜ効くか
Dreamerは actor/critic を「想像」で学習するので、内発が実環境ステップにだけ存在すると学習信号が噛み合いづらい。
内発も latent から出る形にすると、想像更新で一貫して探索圧を与えられる。

7) KL項のウォームアップ（世界モデルの安定化）
何を変えたか
kl_scale_t = kl_scale * min(1, step / kl_warmup_steps) を導入。
序盤はKLを弱くして、徐々に本来の強さへ。
なぜ効くか
序盤から強いKLは posterior/prior のバランスで不安定になりやすい。
立ち上がりを滑らかにして、表現崩壊や学習発散を起こしにくくする。

8) D_queue の中身と役割を整理（「誤差」と「内発教師」を別ターゲットで学習）
何を変えたか
D_queue に (z, h, a, eps_current, delta_norm_clipped) を保存。
周期更新で
gφ：(z,h,a)->eps_current
intrinsic_model：(z,h,a)->delta_norm_clipped
をそれぞれ学習し、更新後に D_queue をクリア。
なぜ効くか
“このサイクルで得た誤差”を“このサイクル終端でまとめて学習”し、次サイクルへ持ち越さないことで、非定常性を局所化できる。
"""


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
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# Import from student_code and exploration
# FIXED IMPORTS for LPM_exploration environment
try:
    from student_code import (
        Agent, RSSM, Encoder, Decoder, RewardModel, DiscountModel,
        Actor, DiscreteActor, Critic, ErrorPredictor, MSE
    )
    from exploration.noisy_wrapper import NoisyTVEnvWrapperCIFAR
    from exploration.cifar import create_cifar_function_simple
except ImportError:
    # Fallback if running from root relative path (unlikely in this Docker setup but safe)
    import sys
    sys.path.append("..")    
    from world_model.world_model_lpm import (
        Agent, RSSM, Encoder, Decoder, RewardModel, DiscountModel,
        Actor, DiscreteActor, Critic, ErrorPredictor, MSE
    )
    # Define dummies if still missing? No, we trust the env.

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


class RunningMeanStd:
    """
    Scalar running mean/std (Welford-style).
    """
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: float):
        x = float(x)
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var = ((self.count - 1.0) * self.var + delta * delta2) / self.count

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))


class IntrinsicRewardModel(nn.Module):
    """
    Predicts intrinsic reward (normalized/clipped) from (z_t, h_t, a_t).
    This lets intrinsic reward be used consistently during imagination.
    """
    def __init__(self, mlp_hidden_dim, rnn_hidden_dim, state_dim, num_classes, action_dim):
        super().__init__()
        z_dim = state_dim * num_classes
        in_dim = z_dim + rnn_hidden_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.ELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ELU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, z, h, a):
        x = torch.cat([z, h, a], dim=-1)
        return self.net(x)


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

        # KL
        self.kl_scale = 0.1
        self.kl_balance = 0.8
        self.kl_warmup_steps = 100_000  # gradually turn on KL penalty early on

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
        self.seed_iter = 10_000
        self.eval_interval = 50000
        self.eval_freq = 5
        self.eval_episofdes = 5

        # LPM Params (Reliability-Gated & Reward-Feedback)
        self.lpm_eta = 1.0  # Max intrinsic coefficient (Discovery mode)
        self.lpm_focus_eta = 0.1  # Min intrinsic coefficient (Focus mode)
        self.lpm_decay_rate = 0.995  # Decay rate per episode in Focus mode
        self.recent_reward_len = 10  # Moving average window size for reward feedback
        
        self.lpm_clip = 1.0 # Intrinsic reward clipping
        self.intr_lr = 1e-4
        self.lpm_lr = 1e-4 # Error Predictor LR
        self.intr_batch_size = 64
        self.noisy_tv = False # Default value

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

        # IMPORTANT: store EXTRINSIC reward only (stabilizes WM reward learning)
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
        for i in range(cfg.eval_episofdes): # Corrected typo from eval_episofdes to eval_episodes
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
    (Atari想定では使われませんが、元コード互換のため残します)
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

    # For non-Atari, turn into pixel observation via render() FIRST
    if env_name != "Crafter" and "ALE/" not in env_name and "NoFrameskip" not in env_name and "Breakout" not in env_name:
        env = RenderWrapper(env)

    # Common wrappers: resize to 64x64 for Dreamer
    if env_name != "Crafter" and not isinstance(env, MnistEnv):
        from gymnasium.wrappers import ResizeObservation
        env = ResizeObservation(env, (64, 64))

    if noisy:
        get_cifar = create_cifar_function_simple()
        if isinstance(env.action_space, gym.spaces.Discrete):
            env = NoisyTVEnvWrapperCIFAR(env, get_cifar, num_random_actions=2)
        else:
            # continuous noisy-TV is not used for Atari; keep placeholder wrapper (works if needed)
            env = NoisyTVWrapperContinuous(env, get_cifar, trigger_threshold=1.5)

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
    
    # LPM Arguments
    parser.add_argument("--enable-lpm", action="store_true")
    parser.add_argument("--lpm-eta", type=float, default=1.0)
    parser.add_argument("--lpm-focus-eta", type=float, default=0.1)
    parser.add_argument("--lpm-decay-rate", type=float, default=0.995)
    
    args = parser.parse_args()

    cfg = Config()
    cfg.iter = args.steps
    
    # Apply LPM args
    cfg.use_lpm = args.enable_lpm
    cfg.lpm_eta = args.lpm_eta
    cfg.lpm_focus_eta = args.lpm_focus_eta
    cfg.lpm_decay_rate = args.lpm_decay_rate

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

    env = make_env_simple(args.seed, args.env_name, cfg.noisy_tv) # Changed args.noisy_tv to cfg.noisy_tv
    atexit.register(env.close)

    eval_env = make_env_simple(args.seed + 100, args.env_name, cfg.noisy_tv) # Changed args.noisy_tv to cfg.noisy_tv
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

    # RewardModel now learns EXTRINSIC reward only
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

    # Intrinsic reward model for imagination
    intrinsic_model = IntrinsicRewardModel(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes, action_dim).to(device)

    agent = Agent(encoder, decoder, rssm, actor, error_predictor).to(device)

    # Optimizers
    wm_params = list(rssm.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + list(reward_model.parameters())
    wm_optimizer = torch.optim.Adam(wm_params, lr=cfg.model_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)

    ep_optimizer = torch.optim.Adam(error_predictor.parameters(), lr=cfg.lpm_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    intr_optimizer = torch.optim.Adam(intrinsic_model.parameters(), lr=cfg.intr_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)

    # Paper-style error replay queue D:
    # stores (z_t, h_t, a_t, eps_t, intr_target)
    # eps_t: current model NLL (scalar)
    # intr_target: normalized+clipped intrinsic signal for that transition (scalar, unscaled by eta)
    # Initialize LPM Variables
    D_queue = deque(maxlen=cfg.D_size)
    
    # Reward Feedback Variables
    recent_ext_rewards = deque(maxlen=cfg.recent_reward_len)
    eta_base = cfg.lpm_eta # Start with full exploration potential
    
    # Reliability Gating Variables
    ep_reliability = 0.0 # R2 score (0.0 means "unreliable", 1.0 means "reliable")
    
    # Running MeanStd for normalization
    delta_rms = RunningMeanStd()
    ep_ready = False  # becomes True once error_predictor has been trained at least once

    # Training Loop
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    total_episode = 1
    total_reward = []

    # Pre-fill buffer (seed episodes) — EXTRINSIC ONLY
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
        f.write("step,eps_nll,pred_prev_eps,delta_raw,delta_norm_clipped,eta_scaled_intr,is_noisy\n")

    print("Starting Main Loop...")
    pbar = tqdm(range(cfg.iter), desc="Training Steps", unit="step")

    for iteration in pbar:
        # --- Interaction & intrinsic bookkeeping (intrinsic NOT stored as reward) ---
        with torch.no_grad():
            action, _pred_obs_normalized = agent(obs, eval=False)

            # Convert agent action -> env action, and action vector for buffer
            if isinstance(env.action_space, gym.spaces.Discrete):
                a_vec = ensure_action_vector(action, env, action_dim)
                env_action = int(np.argmax(a_vec))
            else:
                a_vec = ensure_action_vector(action, env, action_dim)
                env_action = a_vec  # Box expects vector

            next_obs, ext_reward, done, truncated, info = env.step(env_action)
            done_flag = done or truncated

            # Compute eps_current via NLL under decoder distribution (more consistent than pixel MSE)
            eps_current = None
            pred_prev_eps = 0.0
            delta_raw = 0.0
            delta_norm_clipped = 0.0
            intr_eta_scaled = 0.0
            eta_current = 0.0

            next_obs_normalized = preprocess_obs(next_obs)

            t_last_state = agent.last_state
            t_last_rnn = agent.last_rnn_hidden

            if t_last_state is not None:
                t_action = torch.as_tensor(a_vec, device=device).unsqueeze(0)

                # Predict next latent under current model
                t_next_rnn = rssm.recurrent(t_last_state, t_action, t_last_rnn)
                t_next_prior = rssm.get_prior(t_next_rnn)
                t_next_state = t_next_prior.mean.flatten(1)

                # True next obs tensor (1,C,H,W) in the same normalized space as training
                true_next = torch.as_tensor(next_obs_normalized, device=device).permute(2, 0, 1).unsqueeze(0)

                # --- LPM: Intrinsic Reward Calculation (Reliability-Gated) ---
                
                # 1. Determine eta_current based on Reliability Gating
                #    If ep_reliability is 0 (random predictor), eta_current becomes 0 (no random pressure).
                eta_current = eta_base * ep_reliability

                # NLL as "prediction error"
                t_next_obs_dist = decoder(t_next_state, t_next_rnn)
                eps_current = float((-t_next_obs_dist.log_prob(true_next)).mean().item())

                # Predict EXPECTED error current
                pred_prev_eps = float(error_predictor(t_last_state, t_last_rnn, t_action).item())

                # Delta = (Predicted Error) - (Actual Error)
                # If we predicted "High Error" but got "Low Error" -> Positive Surprise (Learning Progress)
                delta_raw = pred_prev_eps - eps_current

                # Normalize and Clip
                delta_rms.update(delta_raw)
                delta_norm = (delta_raw - delta_rms.mean) / (delta_rms.std + 1e-8)
                delta_norm_clipped = float(np.clip(delta_norm, -cfg.lpm_clip, cfg.lpm_clip))
                
                # Apply eta_current (Gated)
                intr_eta_scaled = float(np.clip(eta_current * delta_norm_clipped, -cfg.lpm_clip, cfg.lpm_clip))

                # Store for training EP and intrinsic reward model from D
                # Store on CPU to keep memory stable.
                D_queue.append((
                    t_last_state.detach().cpu(),
                    t_last_rnn.detach().cpu(),
                    torch.as_tensor(a_vec).detach().cpu(),
                    float(eps_current),
                    float(delta_norm_clipped),
                ))

                is_noisy_step = info.get("noisy", False)
                with open(lpm_log_file, "a") as f:
                    f.write(f"{iteration},{eps_current},{pred_prev_eps},{delta_raw},{delta_norm_clipped},{intr_eta_scaled},{is_noisy_step}\n")

                if args.wandb:
                    metrics = {
                        "LPM/eps_nll": eps_current,
                        "LPM/pred_prev_eps": pred_prev_eps,
                        "LPM/delta_raw": delta_raw,
                        "LPM/delta_norm_clipped": delta_norm_clipped,
                        "LPM/eta_base": eta_base,
                        "LPM/ep_reliability": ep_reliability,
                        "LPM/eta_current": eta_current,
                        "LPM/intrinsic_eta_scaled": intr_eta_scaled,
                    }
                    if is_noisy_step:
                        metrics["LPM/intrinsic_noisy"] = intr_eta_scaled
                        metrics["LPM/eps_noisy"] = eps_current
                    else:
                        metrics["LPM/intrinsic_clean"] = intr_eta_scaled
                        metrics["LPM/eps_clean"] = eps_current
                    wandb.log(metrics, step=iteration)

            # Push ONLY extrinsic reward into replay buffer
            replay_buffer.push(preprocess_obs(obs), a_vec, ext_reward, done_flag)

            obs = next_obs
            total_reward.append(ext_reward)  # external reward tracking

            if done_flag:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                agent.reset()

                # --- Reward Feedback Logic (Recent Average Feedback) ---
                recent_ext_rewards.append(np.mean(total_reward)) # average reward per episode
                avg_recent_reward = np.mean(recent_ext_rewards) if recent_ext_rewards else 0.0

                #ここでLPMのスケジューリングを行う．エピソード終了時に直近の平均報酬が0より大きければeta_baseを減衰させて，0ならば初期値に戻す．
                if avg_recent_reward > 0:
                    # Focus Phase: Decay eta_base towards focus_eta
                    eta_base = eta_base * cfg.lpm_decay_rate + cfg.lpm_focus_eta * (1 - cfg.lpm_decay_rate)
                else:
                    # Discovery/Recovery Phase: Instant recovery to full potential
                    # (But effective eta is still limited by ep_reliability)
                    eta_base = cfg.lpm_eta

                print(f"Episode {total_episode} ExtReward: {np.mean(total_reward):.4f} (Avg10: {avg_recent_reward:.4f})")
                print(f"  > LPM State: Base_Eta={eta_base:.3f} * Reliability={ep_reliability:.3f} -> Eta={eta_current:.3f}")

                if args.wandb:
                    wandb.log({
                        "Episode/Extrinsic_Reward": np.mean(total_reward),
                        "Episode/Avg_Recent_Reward": avg_recent_reward,
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
            done_flags = 1 - done_flags  # continuation mask (1=not done)

            observations = torch.permute(torch.as_tensor(observations, device=device), (1, 0, 4, 2, 3))
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)
            done_flags = torch.as_tensor(done_flags, device=device).transpose(0, 1).float()

            # KL warmup scale
            kl_scale_t = cfg.kl_scale * min(1.0, float(iteration) / float(cfg.kl_warmup_steps))

            # --- World Model Update (reward_model learns EXTRINSIC ONLY) ---
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

            wm_loss = obs_loss + cfg.reward_loss_scale * reward_loss + kl_scale_t * kl_loss

            wm_optimizer.zero_grad()
            wm_loss.backward()
            clip_grad_norm_(wm_params, cfg.gradient_clipping)
            wm_optimizer.step()

            # --- Train error model g_phi on D (errors from previous cycle) ---
            ep_loss_D = None
            intr_loss_D = None

            if len(D_queue) >= max(4, min(cfg.D_batch_size, len(D_queue))):
                batch_size_D = min(cfg.D_batch_size, len(D_queue))
                idx = np.random.choice(len(D_queue), size=batch_size_D, replace=False)
                batch = [D_queue[i] for i in idx]

                z_batch = torch.cat([b[0] for b in batch], dim=0).to(device)
                h_batch = torch.cat([b[1] for b in batch], dim=0).to(device)
                a_batch = torch.stack([b[2] for b in batch], dim=0).to(device).float()
                eps_batch = torch.tensor([b[3] for b in batch], device=device).float()

                pred = error_predictor(z_batch, h_batch, a_batch).squeeze(-1)
                ep_loss_D = F.mse_loss(pred, eps_batch)

                # --- R2 (Reliability) Calculation ---
                with torch.no_grad():
                    ep_variance = torch.var(eps_batch)
                    if ep_variance < 1e-8:
                        ep_variance = 1.0 # Avoid division by zero
                    
                    # R2 = 1 - MSE / Variance
                    # If MSE > Variance (worse than mean), R2 becomes negative.
                    ep_r2 = 1.0 - ep_loss_D / ep_variance
                    
                    # Clip: Treat "Learning < Mean" as "Unreliable (0.0)"
                    ep_reliability = float(torch.clamp(ep_r2, 0.0, 1.0).item())

                ep_optimizer.zero_grad()
                ep_loss_D.backward()
                clip_grad_norm_(error_predictor.parameters(), cfg.gradient_clipping)
                ep_optimizer.step()

            # --- Train intrinsic reward model on D (normalized/clipped targets) ---
            # Train if we have enough data (Reliability is handled by R2 check above)
            if len(D_queue) >= max(4, min(cfg.intr_batch_size, len(D_queue))):
                batch_size_I = min(cfg.intr_batch_size, len(D_queue))
                idx = np.random.choice(len(D_queue), size=batch_size_I, replace=False)
                batch = [D_queue[i] for i in idx]

                z_batch = torch.cat([b[0] for b in batch], dim=0).to(device)
                h_batch = torch.cat([b[1] for b in batch], dim=0).to(device)
                a_batch = torch.stack([b[2] for b in batch], dim=0).to(device).float()
                intr_tgt = torch.tensor([b[4] for b in batch], device=device).float().unsqueeze(-1)

                intr_pred = intrinsic_model(z_batch, h_batch, a_batch)
                intr_loss_D = F.mse_loss(intr_pred, intr_tgt)

                intr_optimizer.zero_grad()
                intr_loss_D.backward()
                clip_grad_norm_(intrinsic_model.parameters(), cfg.gradient_clipping)
                intr_optimizer.step()

            # Clear D for next cycle
            D_queue.clear()

            if args.wandb:
                log_dict = {
                    "Train/WM_Loss": wm_loss.item(),
                    "Train/Obs_Loss": obs_loss.item(),
                    "Train/Reward_Loss": reward_loss.item(),
                    "Train/KL_Loss": kl_loss.item(),
                    "Train/KL_Scale_T": kl_scale_t,
                    "Train/Reward_Mean_Ext": rewards.mean().item(),
                }
                if ep_loss_D is not None:
                    log_dict["Train/EP_Loss_D"] = float(ep_loss_D.item())
                if intr_loss_D is not None:
                    log_dict["Train/Intr_Loss_D"] = float(intr_loss_D.item())
                wandb.log(log_dict, step=iteration)

            pbar.set_postfix({
                "ext_r": f"{np.mean(total_reward[-10:]):.2f}" if total_reward else "0.00",
                "wm": f"{wm_loss.item():.2f}",
                "epD": f"{ep_loss_D.item():.2f}" if ep_loss_D is not None else "NA",
                "intrD": f"{intr_loss_D.item():.2f}" if intr_loss_D is not None else "NA",
            })

            # --- Actor Critic Update (imagination) ---
            # Detach from WM gradients
            flatten_rnn_hiddens_det = flatten_rnn_hiddens.detach()
            flatten_states_det = flatten_states.detach()

            imagined_states = torch.zeros(cfg.imagination_horizon + 1, *flatten_states_det.shape, device=device)
            imagined_rnn_hiddens = torch.zeros(cfg.imagination_horizon + 1, *flatten_rnn_hiddens_det.shape, device=device)
            imagined_action_entropys = torch.zeros((cfg.imagination_horizon, cfg.batch_size * (cfg.seq_length - 1)), device=device)

            # intrinsic for each imagined transition
            imagined_intr = torch.zeros((cfg.imagination_horizon, cfg.batch_size * (cfg.seq_length - 1)), device=device)

            imagined_states[0] = flatten_states_det
            imagined_rnn_hiddens[0] = flatten_rnn_hiddens_det

            flat_s = flatten_states_det
            flat_h = flatten_rnn_hiddens_det

            for i in range(1, cfg.imagination_horizon + 1):
                i_actions, i_action_log_probs, i_action_entropys = actor(flat_s, flat_h)

                # intrinsic reward predicted on (s,h,a) before transition
                # Note: intrinsic_model is always trained on D_queue, so it's valid.
                # However, if reliability is 0, eta_current is 0, so this term vanishes anyway.
                intr_pred = intrinsic_model(flat_s, flat_h, i_actions).squeeze(-1)
                imagined_intr[i - 1] = intr_pred

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

            # Extrinsic reward from reward_model (learned on true env reward only)
            imagined_rewards_ext = reward_model(flatten_imagined_states, flatten_imagined_rnn_hiddens).mean.view(cfg.imagination_horizon, -1)

            # Intrinsic (normalized/clipped) predicted reward, scaled by Gated Eta
            # eta_current is updated in the rollout loop based on Reliability * Base
            imagined_rewards_int = imagined_intr  # already horizon x batch
            imagined_rewards_total = imagined_rewards_ext + eta_current * imagined_rewards_int

            target_values = target_critic(flatten_imagined_states, flatten_imagined_rnn_hiddens).view(cfg.imagination_horizon, -1).detach()

            discount_arr = (cfg.discount * torch.ones_like(imagined_rewards_total)).to(device)
            lambda_target = calculate_lambda_target(imagined_rewards_total, discount_arr, target_values, cfg.lambda_)

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
                    "Train/Critic_Loss": critic_loss.item(),
                    "Train/Imagined_Reward_Ext_Mean": imagined_rewards_ext.mean().item(),
                    "Train/Imagined_Reward_Int_Mean": imagined_rewards_int.mean().item(),
                    "Train/Imagined_Reward_Total_Mean": imagined_rewards_total.mean().item(),
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
