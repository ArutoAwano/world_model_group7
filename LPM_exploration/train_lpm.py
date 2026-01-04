import sys
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

# Import from student_code and exploration
from student_code import Agent, RSSM, Encoder, Decoder, RewardModel, DiscountModel, Actor, Critic, ErrorPredictor, MSE
from exploration.noisy_wrapper import NoisyTVEnvWrapperCIFAR
from exploration.cifar import create_cifar_function_simple
import gymnasium as gym
import gymnasium as gym
from PIL import Image
import imageio

# Config Class from Notebook
class Config:
    def __init__(self, **kwargs):
        # data settings
        self.buffer_size = 100_000
        self.batch_size = 16
        self.seq_length = 50
        self.imagination_horizon = 10

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
    with torch.no_grad():
        for i in range(cfg.eval_episodes):
            obs = env.reset()
            if isinstance(obs, tuple): obs = obs[0] # Gym API handling
            policy.reset()
            done = False
            truncated = False
            episode_reward = []
            frames = [] 
            
            while not done and not truncated:
                action, _ = policy(obs, eval=True)
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
                
                # Render for video
                frame = env.render() 
                frames.append(frame)
                
                episode_reward.append(reward)
            
            # Save video for the first episode of evaluation
            if i == 0:
                video_path = f"videos/eval_iter_{step}_ep_{i}.mp4"
                try:
                     imageio.mimsave(video_path, frames, fps=30)
                     print(f"Saved video to {video_path}")
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
    parser.add_argument('--env-name', default='ALE/Breakout-v5')
    parser.add_argument('--noisy-tv', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=200000)
    args = parser.parse_args()
    
    cfg = Config()
    cfg.iter = args.steps
    
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
                
            return obs, reward, terminated, truncated, info

    # Simple env factory
    def make_env_simple(seed, noisy=False):
        # Use Pendulum as simple test env
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        env = RenderWrapper(env)
        from gymnasium.wrappers import ResizeObservation
        env = ResizeObservation(env, (64, 64))
        
        if noisy:
             get_cifar = create_cifar_function_simple()
             # Use Continuous Wrapper
             env = NoisyTVWrapperContinuous(env, get_cifar, trigger_threshold=1.0)
        return env

    env = make_env_simple(args.seed, args.noisy_tv)
    
    # Video Recording for Evaluation
    eval_env = make_env_simple(args.seed + 100, args.noisy_tv)
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
            
    # Main Loop
    print("Starting Main Loop...")
    for iteration in range(cfg.iter):
        
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
            
            # Action handling for Pendulum (Continuous)
            if isinstance(env.action_space, gym.spaces.Box):
                 # Ensure action is distinct array (1,) for Pendulum, not scalar
                 if action.ndim == 0:
                     action = np.expand_dims(action, axis=0)
            
            # For interaction
            next_obs, reward, done, truncated, _ = env.step(action)
            done_flag = done or truncated
            
            # --- LPM Calculation ---
            # 1. Actual Error
            # pred_obs_normalized is normalized [-0.5, 0.5]
            # next_obs is [0, 255]
            # Convert next_obs to match pred
            next_obs_normalized = preprocess_obs(next_obs) # [-0.5, 0.5]
            
            # Error is MSE between pred_obs (reconstruction of obs) and next_obs?
            # NO! LPM requires prediction of NEXT obs.
            # But Dreamer decoder predicts CURRENT obs from CURRENT state.
            
            # We need to predict NEXT obs.
            # We can use agent.rssm to step forward using 'agent.last_state' and 'action'.
            # agent.last_state: z_t
            # agent.last_rnn_hidden: h_t
            # action: a_t
            # next_rnn: h_{t+1}, next_prior: z_{t+1}
            # next_obs_pred: x_{t+1}
            
            # Ensure types for torch
            t_last_state = agent.last_state # Already on device
            t_last_rnn = agent.last_rnn_hidden
            
            if t_last_state is not None:
                # Prepare action tensor
                t_action = torch.as_tensor(action, device=device).unsqueeze(0) # (1, action_dim)
                
                # Predict Step
                t_next_rnn = rssm.recurrent(t_last_state, t_action, t_last_rnn)
                t_next_prior = rssm.get_prior(t_next_rnn)
                t_next_state = t_next_prior.mean.flatten(1) # Use mean or sample? Mean is deterministic prediction.
                
                t_next_obs_dist = decoder(t_next_state, t_next_rnn)
                t_next_obs_pred = t_next_obs_dist.mean.squeeze().cpu().numpy() # (3, 64, 64)
                t_next_obs_pred = t_next_obs_pred.transpose(1, 2, 0) # (64, 64, 3)
                
                # Actual Error (MSE) per pixel -> sum/mean
                actual_error = np.mean((next_obs_normalized - t_next_obs_pred)**2)
                
                # Predicted Error
                pred_error_est = error_predictor(t_last_state, t_last_rnn, t_action).item()
                
                # Intrinsic Reward
                intr_reward = cfg.lpm_eta * (pred_error_est - actual_error)
                
                # Clip?
                intr_reward = max(-1.0, min(1.0, intr_reward))
                
                total_reward_val = reward + intr_reward
                
                # print(f"LPM: ActErr={actual_error:.4f} PredErr={pred_error_est:.4f} Intr={intr_reward:.4f}")
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
                total_reward = []
                total_episode += 1
                
                if total_episode % cfg.eval_freq == 0:
                    evaluation(eval_env, agent, iteration, cfg)
                    eval_env.reset()
                    agent.reset()

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
                # We need to calculate Actual Error for this step i
                # Actual Prediction Error of x_{i+1} given (z_i, h_i, a_i)
                # We have just computed z_{i+1} and h_{i+1}.
                # The "Predicted Obs" comes from Decoder(z_{i+1}_prior, h_{i+1})?
                # Using Prior for strict prediction
                
                pred_next_state = next_state_prior.rsample().flatten(1) # z_{i+1} from prior
                # Typically, we want deterministic prediction error? 
                # Let's use mean for stability
                pred_next_state_mean = next_state_prior.mean.flatten(1)
                
                pred_next_obs_dist = decoder(pred_next_state_mean, rnn_hidden)
                pred_next_obs = pred_next_obs_dist.mean.view(cfg.batch_size, 3, 64, 64)
                
                real_next_obs = observations[i+1] # (B, 3, 64, 64)
                
                # Actual Error: MSE
                actual_error_batch = ((real_next_obs - pred_next_obs) ** 2).mean(dim=[1,2,3]) # (B,)
                
                # Predicted Error: ErrorPredictor(z_i, h_i, a_i)
                # We need previous state/rnn. 
                # At start (i=0), previous state is 'state' before update (zeros).
                
                if i > 0:
                    prev_state = states[i] # z_i
                    prev_rnn = rnn_hiddens[i] # h_i
                    
                    pred_error_hat = error_predictor(prev_state.detach(), prev_rnn.detach(), actions[i]).squeeze()
                    
                    # Loss: MSE(pred, log(actual)) or just MSE(pred, actual)?
                    # improve.py used MSE(pred, log(actual))
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
            
            if (iteration + 1) % cfg.slow_critic_update == 0:
                target_critic.load_state_dict(critic.state_dict())
                
    # Save Model
    agent.to("cpu")
    torch.save(agent, "agent_lpm.pth")
    print("Model saved to agent_lpm.pth")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
