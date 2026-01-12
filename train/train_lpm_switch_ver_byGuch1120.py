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

# student_codeとexplorationからのインポート
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

# stable_baselines3のラッパーが利用可能な場合はインポートし、そうでない場合は最小限のものを定義する
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
        self.action_space = gym.spaces.Discrete(10) # ダミーの行動空間
        
        # MNISTデータの読み込み
        try:
            self.mnist_data = datasets.MNIST('../data', train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize((64, 64)),
                                               transforms.ToTensor()
                                           ]))
        except:
            # ダウンロードに失敗した場合やインターネットの問題がある場合のフォールバック、ランダムノイズを生成するかローカルを試す
            print("Warning: Could not load actual MNIST, using random noise placeholder for MnistEnv")
            self.mnist_data = None

        self.current_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = random.randint(0, len(self.mnist_data) - 1) if self.mnist_data else 0
        return self._get_obs(), {}

    def step(self, action):
        # アクションは画像をランダムに変更し、異なるチャンネル/数字を「見る」ことをシミュレートする
        self.current_idx = random.randint(0, len(self.mnist_data) - 1) if self.mnist_data else 0
        obs = self._get_obs()
        reward = 0.0 # ただ見ているだけなので外発的報酬はない
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        if self.mnist_data:
            img, _ = self.mnist_data[self.current_idx]
            # Tensor (C, H, W) 0-1 を numpy (H, W, C) 0-255 に変換
            img = img.permute(1, 2, 0).numpy() * 255.0
            img = img.astype(np.uint8)
            # MNISTはグレースケール(1チャンネル)なので、一貫性のためにRGB(3チャンネル)に変換する
            if img.shape[2] == 1:
                img = np.concatenate([img]*3, axis=2)
            return img
        else:
            return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    def render(self):
        return self._get_obs()

# ノートブックからの設定クラス
class Config:
    def __init__(self, **kwargs):
        # データ設定
        self.buffer_size = 100_000
        self.batch_size = 16
        self.seq_length = 50
        self.imagination_horizon = 20

        # モデル次元
        self.state_dim = 32
        self.num_classes = 32
        self.rnn_hidden_dim = 400
        self.mlp_hidden_dim = 300

        # 学習パラメータ
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

        # lambda return パラメータ
        self.discount = 0.995
        self.lambda_ = 0.95

        # 学習期間の設定
        self.iter = 6000
        self.seed_iter = 1000 # 開発中の高速開始のために削減
        self.eval_interval = 20000
        self.eval_freq = 5
        self.eval_episodes = 5
        
        # LPM パラメータ
        self.lpm_eta = 1.0
        self.lpm_lr = 2e-4
        
        # LPM パラメータ
        self.lpm_eta = 1.0
        self.lpm_lr = 2e-4
        self.use_lpm = False
        
        # kwargsで上書き
        for k, v in kwargs.items():
            setattr(self, k, v)


# リプレイバッファ
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
    
    # ディレクトリが存在することを確認
    os.makedirs("eval_view/video", exist_ok=True)
    os.makedirs("eval_view/images", exist_ok=True)
    
    with torch.no_grad():
        for i in range(cfg.eval_episodes):
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0] # Gym API 処理
            policy.reset()
            done = False
            truncated = False
            episode_reward = []
            frames = [] 
            recon_frames = []
            
            while not done and not truncated:
                # エージェントは (action, recon_img) を返す
                # recon_img は (64, 64, 3) で範囲は 0-1
                action, recon_img = policy(obs, eval=True)
                
                # 必要に応じてアクションをワンホットインデックスに変換するが、ここでは単純な環境
                if isinstance(env.action_space, gym.spaces.Box):
                    # アクションを空間に合わせてクリップ?
                    # policy は numpy 配列を返す
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, reward, done, truncated, info = env.step(action)
                elif hasattr(env.action_space, 'n'):
                    action_scalar = np.argmax(action)
                    obs, reward, done, truncated, info = env.step(action_scalar)
                else:
                    obs, reward, done, truncated, info = env.step(action)
                
                # 動画用にレンダリング (実際の観測)
                frame = env.render() 
                frames.append(frame)
                
                # 再構成されたフレーム
                if recon_img is not None:
                    # recon_img は 0-1 の float。0-255 の uint8 に変換
                    recon_frame = (recon_img * 255.0).astype(np.uint8)
                    recon_frames.append(recon_frame)
                
                episode_reward.append(reward)
            
            # 評価の最初のエピソードの動画と画像を保存
            if i == 0 and len(frames) > 0:
                # 動画を保存 (可能なら横に並べる、または個別)
                video_path = f"eval_view/video/eval_iter_{step}_ep_{i}.mp4"
                try:
                     # フレームが uint8 numpy 配列であることを確認
                     frames = [np.array(f, dtype=np.uint8) for f in frames if f is not None]
                     
                     if len(recon_frames) > 0 and len(recon_frames) == len(frames):
                         # 横に並べた動画を作成
                         # 必要に応じてフレームをリサイズして合わせる (env.render は 64x64 より大きい場合がある)
                         # recon は 64x64。frame は env に依存。
                         
                         combined_frames = []
                         for f, r in zip(frames, recon_frames):
                             # f を r (64x64) に合わせて単純に結合するためにリサイズ
                             f_pil = Image.fromarray(f)
                             r_pil = Image.fromarray(r)
                             
                             f_s = f_pil.resize((64, 64))
                             
                             # 水平に結合
                             comb = Image.new('RGB', (128, 64))
                             comb.paste(f_s, (0, 0))
                             comb.paste(r_pil, (64, 0))
                             combined_frames.append(np.array(comb))
                             
                         imageio.mimsave(video_path, combined_frames, fps=30)
                         print(f"Saved combined video to {video_path}")
                     else:
                        imageio.mimsave(video_path, frames, fps=30)
                        print(f"Saved video to {video_path}")

                     # スナップショット画像を保存 (開始、中間、終了)
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
    parser.add_argument('--env-name', default='ALE/Breakout-v5') # デフォルトをAtari環境とする
    parser.add_argument('--noisy-tv', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=200000)
    # WandB arguments
    parser.add_argument('--wandb', action='store_true', help='Use WandB logging') # WandB引数
    parser.add_argument('--wandb-entity', default=None, help='WandB entity')
    parser.add_argument('--wandb-run-name', default=None, help='Run name')
    parser.add_argument('--enable-lpm', action='store_true', help='Enable LPM intrinsic reward') # LPM内発的報酬を有効化
    parser.add_argument('--seed-steps', type=int, default=1000, help='Number of seed steps') # シードステップ数
    args = parser.parse_args()
    
    cfg = Config()
    cfg.iter = args.steps
    cfg.use_lpm = args.enable_lpm
    cfg.seed_iter = args.seed_steps
    
    # WandBの初期化
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
    
    # 単純な環境ファクトリー

    # カスタムレンダリングラッパー
    class RenderWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            # レンダリングするまで形状は不明だが、通常は (H, W, 3)
            # ResizeObservation が後に続くと仮定する
            # Pendulum のデフォルトレンダリングは 500x500
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(500, 500, 3), dtype=np.uint8)
            # メタデータが rgb_array を明示的にサポートしていることを確認
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
            # 離散値の場合、探索ノイズとしてNoisyTVをランダムにトリガーする
            # または特定のアクションでトリガー？ ここでは単純化のためにランダム確率または特定の「何もしない」を使用する
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
            # アクションがノイズをトリガーするか確認
            # アクションがスカラーまたはベクトルであると仮定
            # アクションコンポーネントのいずれかが閾値を超えた場合にトリガー
            is_noise = np.any(np.abs(action) > self.trigger_threshold)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if is_noise:
                # 観測をランダムなCIFARに置き換える
                noise_img = self.get_random_cifar() # (32, 32, 3)
                # (64, 64) にリサイズ??
                # get_random_cifar は 32x32 を返す
                # 64x64 が必要
                # PIL や scipy を使ってリサイズするか、単純に繰り返すことができる
                # または get_random_cifar が必要なものを返すと仮定するか？
                # create_cifar_function_simple は 32x32 を返す
                # 手動でリサイズしよう
                
                # PILを使用
                pil_img = Image.fromarray(noise_img)
                pil_img = pil_img.resize((64, 64), Image.NEAREST)
                noise_img = np.array(pil_img)
                
                # チャンネルの順序を確認。noise_img は (64, 64, 3)
                # RenderWrapper は (64, 64, 3) を返す (ResizeObservation から)
                # なので一致する
                obs = noise_img
                
                # TVを見ている場合は報酬ゼロ？ それとも環境の報酬(低確率/ランダム)を維持？
                # 通常、TVを見ることは外発的報酬ゼロを与える
                reward = 0.0
            
            info["noisy"] = is_noise
            return obs, reward, terminated, truncated, info

    # Simple env factory
    def make_env_simple(seed, env_name, noisy=False):
        env = None
        trigger_thresh = None 
        trigger_prob = 0.01

        if env_name == "Craftium":
            # "Craftium"のみが指定された場合はOpenWorldをデフォルトにする
            env = gym.make("Craftium/OpenWorld-v0", render_mode="rgb_array")
        elif env_name.startswith("Craftium/"):
            # 特定のCraftiumタスク(例: Craftium/ChopTree-v0)を許可
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
            gym.register_envs(ale_py) # 念のため。通常はインポートするだけで十分
            
            # Atari環境の処理
            # 'ALE/Breakout-v5'などを使用
            # 標準的な動作(FrameSkip, Resizeなど)のためにAtariPreprocessingを使用
            env = gym.make(env_name, render_mode="rgb_array", frameskip=1) # frameskipはラッパーによって処理される
            
            from gymnasium.wrappers import AtariPreprocessing, TransformReward
            
            # AtariPreprocessingの処理内容:
            # - NoopReset
            # - FrameSkip (デフォルト 4)
            # - Resize (デフォルト 84x84 -> 64x64に設定するか後でリサイズ可能)
            # - Grayscale (デフォルト True)
            # - Scale (デフォルト False)
            
            # Dreamerの既存設定との一貫性のために64x64 RGBが必要
            env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=64, terminal_on_life_loss=False, grayscale_obs=False, scale_obs=False)
            
            # grayscale_obs=Trueの場合、出力は(64, 64)。(64, 64, 3)または(H, W, 3)が必要
            # grayscale_obs=Falseの場合、出力は(64, 64, 3)
            
            # 安定性のために報酬をクリップ
            env = TransformReward(env, lambda r: np.sign(r))
            
            trigger_prob = 0.01
        else:
             try:
                env = gym.make(env_name, render_mode="rgb_array")
             except:
                print(f"Could not make environment {env_name}, defaulting to Pendulum")
                env = gym.make("Pendulum-v1", render_mode="rgb_array")
        
        # 共通ラッパー
        if env_name != "Crafter":
            # MnistEnvはすでに64x64x3を出力している
            # Dreamerのために64x64にリサイズすることを確認
            from gymnasium.wrappers import ResizeObservation
            env = ResizeObservation(env, (64, 64))
            
            # RGBを確保
            # (一部のAtari環境やラッパーはグレースケールを出力する場合があるため、形状の確認が必要かもしれない)
        
        if env_name != "Crafter" and "ALE/" not in env_name and "NoFrameskip" not in env_name and "Breakout" not in env_name:
             env = RenderWrapper(env)

        if noisy:
             get_cifar = create_cifar_function_simple()
             if isinstance(env.action_space, gym.spaces.Box):
                 # NoisyTVEnvWrapperCIFARは離散行動空間向けに設計されている(新しいアクションで拡張)
                 # 連続値の場合は別のアプローチが必要か、ユーザーがAtariに注力しているためとりあえずスキップする
                 print("Warning: NoisyTVEnvWrapperCIFAR not supported for Continuous Env (Pendulum etc). Skipping wrapper.")
             else:
                 # exploration.noisy_wrapper から正しいラッパーを使用
                 # num_random_actions分だけ行動空間を拡張する
                 env = NoisyTVEnvWrapperCIFAR(env, get_cifar, num_random_actions=1)
        return env

    env = make_env_simple(args.seed, args.env_name, args.noisy_tv)
    atexit.register(env.close)
    
    # 評価用の動画記録
    eval_env = make_env_simple(args.seed + 100, args.env_name, args.noisy_tv)
    atexit.register(eval_env.close)
    # 評価ループ内での手動動画保存
    if not os.path.exists("videos"):
        os.makedirs("videos")
    
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    # リプレイバッファ
    replay_buffer = ReplayBuffer(
        capacity=cfg.buffer_size,
        observation_shape=(64, 64, 3), # 3チャンネルと仮定
        action_dim=action_dim
    )
    
    # モデル
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
    
    # オプティマイザ
    wm_params = list(rssm.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + list(reward_model.parameters()) + list(error_predictor.parameters())
    wm_optimizer = torch.optim.Adam(wm_params, lr=cfg.model_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    
    # Training Loop
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]
    
    total_episode = 1
    total_reward = []
    
    # バッファの事前充填
    print("Pre-filling buffer...")
    for _ in range(cfg.seed_iter):
        
        action = env.action_space.sample() # (1,) float
        # action_dimと一致することを確認する必要がある
        
        next_obs, reward, done, truncated, _ = env.step(action)
        done_flag = done or truncated
        
        replay_buffer.push(preprocess_obs(obs), action, reward, done_flag)
        obs = next_obs
        if done_flag:
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0]
            
    # CSVロギング設定
    lpm_log_file = "lpm_stats.csv"
    with open(lpm_log_file, "w") as f:
        f.write("step,actual_error,pred_error,intr_reward,is_noisy\n")

    # Main Loop
    print("Starting Main Loop...")
    
    # プログレスバーラッパー
    pbar = tqdm(range(cfg.iter), desc="Training Steps", unit="step")
    for iteration in pbar:
        
        # LPM報酬計算との相互作用
        with torch.no_grad():
            # agent.__call__ は推論のために内部で正規化を処理する
            # 保存された state/rnn にアクセスしたいので、エージェントロジックを手動で呼び出すかヘルパーを使うか？
            # あるいは単に agent(obs) を使い、last_state が正しく設定されると仮定する (はい、修正しました)
            
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
            
            # 相互作用のため
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
                # "LPMは予測の改善(Error(t-1) - Error(t))、あるいは少なくとも
                # エラーの減少である「学習の進捗」を反映すべきである。"
                # 実際の誤差が予測された誤差よりも低い場合に報酬を与える。
                # ErrorPredictorが「高い誤差」(新規性/難易度)を推定したが、実際の誤差が「低い」(習得済み/学習済み)場合、
                # その差はそれを「学習した」(進歩)ことを意味する。
                #
                # intr_reward = eta * (Predicted_Error - Actual_Error)
                # Actual < Predicted => Reward > 0 (驚くべき成功 / 進歩)
                # Actual > Predicted => Reward < 0 (予期せぬ失敗 / ノイズ？)
                #
                # 注: Noisy-TV (ランダムノイズ) は本質的に予測不可能 (高い Actual Error)。
                # Error Predictor は最終的に Noisy TV に対して「高い誤差」を予測することを学習する。
                # したがって (Predicted (High) - Actual (High)) ~= 0 となる。
                # これにより Noisy-TV の罠を回避できる。
                # Error Predictor は Log(MSE) を予測するため、Log(Actual MSE) と比較する必要がある
                actual_error_log = np.log(actual_error + 1e-6)
                
                intr_reward = 0.0
                if cfg.use_lpm:
                    # ロジック: Reward = Expected_Error - Actual_Error
                    # 大きな誤差(困難)を予想していたが、小さな誤差(理解)が得られた場合、
                    # つまり学習の進捗を表す。
                    intr_reward = cfg.lpm_eta * (pred_error_est - actual_error_log)
                    
                    # 安定化のためのクリッピング
                    intr_reward = max(-1.0, min(1.0, intr_reward))
                
                total_reward_val = reward + intr_reward
                
                # print(f"LPM: ActErr={actual_error:.4f} PredErr={pred_error_est:.4f} Intr={intr_reward:.4f}")
                
                # LPM統計のログ
                is_noisy_step = info.get("noisy", False)
                with open(lpm_log_file, "a") as f:
                    f.write(f"{iteration},{actual_error},{pred_error_est},{intr_reward},{is_noisy_step}\n")

                if args.wandb:
                    # 詳細なLPMメトリクスを各ステップで、あるいは十分な頻度でログ記録する？
                    # 実行時間が長すぎなければ、デバッグ/分析のために各ステップでログ記録しても問題ない。
                    # NoisyとCleanでメトリクスを分ける
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
            
            # バッファに保存
            # action is one-hot
            replay_buffer.push(preprocess_obs(obs), action, total_reward_val, done_flag)
            
            obs = next_obs
            total_reward.append(reward) # 比較のためのログ用に外発的報酬を追跡
            
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
            
            # --- World Model と ErrorPredictor の更新 ---
            emb_observations = encoder(observations.reshape(-1, 3, 64, 64)).view(cfg.seq_length, cfg.batch_size, -1)
            
            state = torch.zeros(cfg.batch_size, cfg.state_dim*cfg.num_classes, device=device)
            rnn_hidden = torch.zeros(cfg.batch_size, cfg.rnn_hidden_dim, device=device)
            
            states = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.state_dim*cfg.num_classes, device=device)
            rnn_hiddens = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.rnn_hidden_dim, device=device)
            
            kl_loss = 0
            ep_loss = 0 # 誤差予測器のLpss
            
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
                    # とりあえず損失だけログ記録する
                }, step=iteration)

            # プログレスバーの説明を更新
            pbar.set_postfix({
                 "total_r": f"{np.mean(total_reward[-10:]):.1f}" if total_reward else "0.0",
                 "wm_loss": f"{wm_loss.item():.2f}"
            })
            
            # --- Actor Critic の更新 ---
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
            # 注: 簡潔にするため次元不一致の処理は省略。想像(imagination)のためにdone_flagsを再利用するのは難しいかもしれない。
            # 割引モデルが終了を処理するか、定数の割引を使用すると仮定する。
            
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
                
    # モデルの保存
    try:
        # モデルの保存
        agent.to("cpu")
        torch.save(agent, "agent_lpm.pth")
        print("Model saved to agent_lpm.pth")
    except Exception as e:
        print(f"Error occurring: {e}")
    finally:
        print("Closing environments...") # 環境を閉じる
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()
