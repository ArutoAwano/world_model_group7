import sys, atexit, os, random, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Normal, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from student_code import Agent, RSSM, Encoder, Decoder, RewardModel, DiscountModel, Actor, DiscreteActor, Critic, MSE
from exploration.cifar import create_cifar_function_simple
from exploration.noisy_wrapper import NoisyTVEnvWrapperCIFAR
import gymnasium as gym, imageio
from PIL import Image
import wandb

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
            pil_img = Image.fromarray(noise_img).resize((obs.shape[1], obs.shape[0]), Image.NEAREST)
            obs, reward = np.array(pil_img), 0.0
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
            pil_img = Image.fromarray(noise_img).resize((obs.shape[1], obs.shape[0]), Image.NEAREST)
            obs, reward = np.array(pil_img), 0.0
        info["noisy"] = is_noise
        return obs, reward, terminated, truncated, info

class Config:
    def __init__(self, **kwargs):
        self.buffer_size, self.batch_size, self.seq_length, self.imagination_horizon = 100_000, 16, 50, 20
        self.state_dim, self.num_classes, self.rnn_hidden_dim, self.mlp_hidden_dim = 32, 32, 400, 300
        self.model_lr, self.actor_lr, self.critic_lr, self.epsilon, self.weight_decay = 2e-4, 4e-5, 1e-4, 1e-5, 1e-6
        self.gradient_clipping, self.kl_scale, self.kl_balance, self.actor_entropy_scale = 100, 0.1, 0.8, 1e-3
        self.slow_critic_update, self.reward_loss_scale, self.discount_loss_scale, self.update_freq = 100, 1.0, 1.0, 80
        self.discount, self.lambda_ = 0.995, 0.95
        self.iter, self.seed_iter, self.eval_interval, self.eval_freq, self.eval_episodes = 6000, 1000, 10000, 5, 5
        self.intr_reward_scale = 1.0
        for k, v in kwargs.items(): setattr(self, k, v)

class ReplayBuffer:
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)
        self.index, self.is_filled = 0, False
    def push(self, obs, act, rew, done):
        self.observations[self.index], self.actions[self.index], self.rewards[self.index], self.done[self.index] = obs, act, rew, done
        self.index = (self.index + 1) % self.capacity
        self.is_filled = self.is_filled or self.index == 0
    def sample(self, batch_size, chunk_length):
        episode_borders, sampled_indexes = np.where(self.done)[0], []
        for _ in range(batch_size):
            while True:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                if not np.logical_and(initial_index <= episode_borders, episode_borders < initial_index + chunk_length).any(): break
            sampled_indexes.extend(range(initial_index, initial_index + chunk_length))
        return (d[sampled_indexes].reshape(batch_size, chunk_length, *d.shape[1:]) for d in [self.observations, self.actions, self.rewards, self.done])
    def __len__(self): return self.capacity if self.is_filled else self.index

def preprocess_obs(obs): return (obs.astype(np.float32) / 255.0) - 0.5

def calculate_lambda_target(rewards, discounts, values, lambda_):
    V_lambda = torch.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        V_lambda[t] = rewards[t] + discounts[t] * (values[t] if t == rewards.shape[0]-1 else ((1-lambda_)*values[t+1] + lambda_*V_lambda[t+1]))
    return V_lambda

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

def evaluation(eval_env, policy, step, cfg):
    all_ep_rewards = []
    os.makedirs("eval_view/video", exist_ok=True); os.makedirs("eval_view/images", exist_ok=True)
    with torch.no_grad():
        for i in range(cfg.eval_episodes):
            obs, _ = eval_env.reset()
            policy.reset()
            done, truncated, episode_reward, frames, recon_frames = False, False, [], [], []
            while not done and not truncated:
                action, recon_img = policy(obs, eval=True)
                action_to_step = action
                if hasattr(eval_env.action_space, 'n'):
                    action_to_step = np.argmax(action) if action.ndim > 0 else action
                else:
                    action_to_step = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)
                obs, reward, done, truncated, info = eval_env.step(action_to_step)
                frames.append(eval_env.render())
                if recon_img is not None: recon_frames.append((recon_img * 255.0).astype(np.uint8))
                episode_reward.append(reward)
            if i == 0 and len(frames) > 0:
                video_path = f"eval_view/video/eval_iter_{step}_ep_{i}.mp4"
                try:
                    frames = [np.array(f, dtype=np.uint8) for f in frames if f is not None]
                    if len(recon_frames) > 0 and len(recon_frames) == len(frames):
                        combined_frames = []
                        for f, r in zip(frames, recon_frames):
                            f_pil, r_pil = Image.fromarray(f), Image.fromarray(r)
                            f_s = f_pil.resize((64, 64))
                            comb = Image.new('RGB', (128, 64))
                            comb.paste(f_s, (0, 0)); comb.paste(r_pil, (64, 0))
                            combined_frames.append(np.array(comb))
                        imageio.mimsave(video_path, combined_frames, fps=10)
                    else:
                       imageio.mimsave(video_path, frames, fps=10)
                except Exception as e: print(f"Failed to save video: {e}")
            all_ep_rewards.append(np.sum(episode_reward))
    print(f"Eval(iter={step}) mean: {np.mean(all_ep_rewards):.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='ALE/Breakout-v5')
    parser.add_argument('--noisy-tv', action='store_true')
    parser.add_argument('--noisy-wrapper-type', type=str, default='simple', choices=['simple', 'action_space'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', default='Dreamer-Intrinsic')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-run-name', default='breakout-intrinsic-v1')
    args = parser.parse_args()
    cfg = Config(iter=args.steps, noisy_wrapper_type=args.noisy_wrapper_type)
    if args.wandb: wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(cfg), mode="online")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    def make_env(seed, name, noisy=False, wrapper_type='simple'):
        if "ALE/" in name:
            from gymnasium.wrappers import AtariPreprocessing
            import ale_py
            gym.register_envs(ale_py)
            env = AtariPreprocessing(gym.make(name, render_mode="rgb_array", frameskip=1), screen_size=64, grayscale_obs=False, terminal_on_life_loss=False)
        else:
            env = gym.make(name, render_mode="rgb_array")
        if noisy:
            cifar_fn = create_cifar_function_simple()
            if wrapper_type == 'action_space' and hasattr(env.action_space, 'n'):
                print("Using action_space extension for Noisy-TV")
                env = NoisyTVEnvWrapperCIFAR(env, cifar_fn)
            else:
                print("Using simple noise injection for Noisy-TV")
                if isinstance(env.action_space, gym.spaces.Box): env = NoisyTVWrapperContinuous(env, cifar_fn)
                else: env = NoisyTVWrapperDiscrete(env, cifar_fn)
        return env
    env, eval_env = make_env(args.seed, args.env_name, args.noisy_tv, args.noisy_wrapper_type), make_env(args.seed + 100, args.env_name, False)
    atexit.register(env.close); atexit.register(eval_env.close)
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    replay_buffer = ReplayBuffer(cfg.buffer_size, (64, 64, 3), action_dim)
    rssm, encoder, decoder = RSSM(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes, action_dim).to(device), Encoder().to(device), Decoder(cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    reward_model = RewardModel(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    actor = DiscreteActor(action_dim, cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device) if hasattr(env.action_space, 'n') else Actor(action_dim, cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    critic, target_critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device), Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    target_critic.load_state_dict(critic.state_dict())
    agent = Agent(encoder, decoder, rssm, actor).to(device)
    wm_params = list(rssm.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + list(reward_model.parameters())
    wm_optimizer, actor_optimizer, critic_optimizer = optim.Adam(wm_params, lr=cfg.model_lr, eps=cfg.epsilon), optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=cfg.epsilon), optim.Adam(critic.parameters(), lr=cfg.critic_lr, eps=cfg.epsilon)
    obs, _ = env.reset()
    for _ in tqdm(range(cfg.seed_iter), desc="Pre-filling buffer"):
        action = env.action_space.sample()
        if not hasattr(env.action_space, 'n'):
            action_to_push = action
        else:
            action_one_hot = np.zeros(action_dim)
            action_one_hot[action] = 1
            action_to_push = action_one_hot
        next_obs, reward, done, truncated, _ = env.step(action)
        replay_buffer.push(preprocess_obs(obs), action_to_push, reward, done or truncated)
        obs = next_obs if not (done or truncated) else env.reset()[0]
    log_file = "intrinsic_stats.csv"
    with open(log_file, "w") as f: f.write("step,actual_error,intr_reward,is_noisy\n")
    print("Starting Main Loop...")
    total_reward, total_episode = [], 1
    for iteration in tqdm(range(cfg.iter), desc="Training Steps"):
        with torch.no_grad():
            action, _ = agent(obs, eval=False)
            env_action = action
            if hasattr(env.action_space, 'n'):
                env_action = np.argmax(action) if action.ndim > 0 else action
            else:
                env_action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, done, truncated, info = env.step(env_action)
            if agent.last_state is not None:
                t_action = torch.as_tensor(action, device=device).unsqueeze(0)
                t_next_rnn = rssm.recurrent(agent.last_state, t_action, agent.last_rnn_hidden)
                t_next_prior = rssm.get_prior(t_next_rnn)
                t_next_state = t_next_prior.mean.flatten(1)
                t_next_obs_pred = decoder(t_next_state, t_next_rnn).mean.squeeze().cpu().numpy().transpose(1, 2, 0)
                actual_error = np.mean((preprocess_obs(next_obs) - t_next_obs_pred)**2)
                intr_reward = cfg.intr_reward_scale * np.log(actual_error + 1e-6)
                total_reward_val = reward + intr_reward
                with open(log_file, "a") as f: f.write(f"{iteration},{actual_error},{intr_reward},{info.get('noisy', False)}\n")
                if args.wandb: wandb.log({"Intrinsic/Actual_Error": actual_error, "Intrinsic/Intrinsic_Reward": intr_reward}, step=iteration)
            else: total_reward_val = reward
            action_to_push = action
            if hasattr(env.action_space, 'n'):
                action_one_hot = np.zeros(action_dim)
                action_idx = np.argmax(action) if action.ndim > 0 else int(action)
                action_one_hot[action_idx] = 1
                action_to_push = action_one_hot
            else:
                action_to_push = action
            replay_buffer.push(preprocess_obs(obs), action_to_push, total_reward_val, done or truncated)
            obs = next_obs
            total_reward.append(reward)
            if done or truncated:
                if args.wandb: wandb.log({"Episode/Extrinsic_Reward": np.sum(total_reward), "Episode/Length": len(total_reward)}, step=iteration)
                obs, _ = env.reset(); agent.reset(); total_reward = []
        if (iteration + 1) % cfg.eval_interval == 0: evaluation(eval_env, agent, iteration, cfg)
        if (iteration + 1) % cfg.update_freq == 0 and len(replay_buffer) > cfg.batch_size:
            observations, actions, rewards, done_flags = replay_buffer.sample(cfg.batch_size, cfg.seq_length)
            observations, actions, rewards, done_flags = (torch.as_tensor(d, device=device) for d in (observations, actions, rewards, 1-done_flags))
            observations, actions, rewards, done_flags = torch.permute(observations, (1,0,4,2,3)), actions.transpose(0,1), rewards.transpose(0,1), done_flags.transpose(0,1).float()
            emb_observations = encoder(observations.reshape(-1, 3, 64, 64)).view(cfg.seq_length, cfg.batch_size, -1)
            state, rnn_hidden = torch.zeros(cfg.batch_size, cfg.state_dim*cfg.num_classes, device=device), torch.zeros(cfg.batch_size, cfg.rnn_hidden_dim, device=device)
            states, rnn_hiddens, kl_loss = torch.zeros(cfg.seq_length, *state.shape, device=device), torch.zeros(cfg.seq_length, *rnn_hidden.shape, device=device), 0
            for i in range(cfg.seq_length-1):
                rnn_hidden = rssm.recurrent(state, actions[i], rnn_hidden)
                next_state_prior, next_detach_prior = rssm.get_prior(rnn_hidden, detach=True)
                next_state_posterior, next_detach_posterior = rssm.get_posterior(rnn_hidden, emb_observations[i+1], detach=True)
                state = next_state_posterior.rsample().flatten(1)
                rnn_hiddens[i+1], states[i+1] = rnn_hidden, state
                kl_loss += (cfg.kl_balance * torch.mean(kl_divergence(next_detach_posterior, next_state_prior))) + ((1 - cfg.kl_balance) * torch.mean(kl_divergence(next_state_posterior, next_detach_prior)))
            kl_loss /= (cfg.seq_length - 1)
            flatten_rnn_hiddens, flatten_states = rnn_hiddens[1:].reshape(-1, cfg.rnn_hidden_dim), states[1:].reshape(-1, cfg.state_dim*cfg.num_classes)
            obs_dist, reward_dist = decoder(flatten_states, flatten_rnn_hiddens), reward_model(flatten_states, flatten_rnn_hiddens)
            obs_loss = -torch.mean(obs_dist.log_prob(observations[1:].reshape(-1, 3, 64, 64)))
            reward_loss = -torch.mean(reward_dist.log_prob(rewards[:-1].reshape(-1, 1)))
            wm_loss = obs_loss + cfg.reward_loss_scale * reward_loss + cfg.kl_scale * kl_loss
            wm_optimizer.zero_grad(); wm_loss.backward(); clip_grad_norm_(wm_params, cfg.gradient_clipping); wm_optimizer.step()
            if args.wandb: wandb.log({"Train/WM_Loss": wm_loss.item(), "Train/Obs_Loss": obs_loss.item(), "Train/Reward_Loss": reward_loss.item(), "Train/KL_Loss": kl_loss.item()}, step=iteration)
            # --- Actor Critic Update ---
            # This block combines the fix for the 'inplace operation' error and the 'tensor size' runtime error.
            
            # 1. Initialize tensors to store the imagined trajectory
            imagined_states = torch.zeros(cfg.imagination_horizon + 1, *flatten_states.shape, device=device)
            imagined_rnn_hiddens = torch.zeros(cfg.imagination_horizon + 1, *flatten_rnn_hiddens.shape, device=device)
            imagined_action_entropys_list = []

            # 2. Use temporary variables for the imagination rollout to prevent inplace modification
            current_imagined_states = flatten_states.detach()
            current_imagined_rnn_hiddens = flatten_rnn_hiddens.detach()

            imagined_states[0] = current_imagined_states
            imagined_rnn_hiddens[0] = current_imagined_rnn_hiddens
            
            # 3. Imagination Loop
            for i in range(1, cfg.imagination_horizon + 1):
                i_actions, _, i_action_entropys = actor(current_imagined_states, current_imagined_rnn_hiddens)
                
                # FIX for RuntimeError: Ensure entropy has the correct batch dimension.
                # If actor returns a scalar entropy, expand it to match the batch size.
                if i_action_entropys.ndim == 0:
                    batch_dim = current_imagined_states.shape[0]
                    i_action_entropys = i_action_entropys.unsqueeze(0).repeat(batch_dim)

                imagined_action_entropys_list.append(i_action_entropys)

                # Update the temporary variables for the next iteration
                current_imagined_rnn_hiddens = rssm.recurrent(current_imagined_states, i_actions, current_imagined_rnn_hiddens)
                current_imagined_states_prior = rssm.get_prior(current_imagined_rnn_hiddens)
                current_imagined_states = current_imagined_states_prior.rsample().flatten(1)
                
                # Store the results of this step
                imagined_states[i] = current_imagined_states
                imagined_rnn_hiddens[i] = current_imagined_rnn_hiddens
            
            imagined_action_entropys = torch.stack(imagined_action_entropys_list)
            
            # 4. Loss Calculation
            flatten_imagined_states = imagined_states[1:].reshape(-1, cfg.state_dim * cfg.num_classes)
            flatten_imagined_rnn_hiddens = imagined_rnn_hiddens[1:].reshape(-1, cfg.rnn_hidden_dim)
            
            imagined_rewards = reward_model(flatten_imagined_states, flatten_imagined_rnn_hiddens).mean.view(cfg.imagination_horizon, -1)
            target_values = target_critic(flatten_imagined_states, flatten_imagined_rnn_hiddens).view(cfg.imagination_horizon, -1).detach()
            
            discount_arr = torch.full_like(imagined_rewards, cfg.discount, device=device)
            
            lambda_target = calculate_lambda_target(imagined_rewards, discount_arr, target_values, cfg.lambda_)
            
            weights = torch.cumprod(torch.cat([torch.ones(1, imagined_rewards.shape[1], device=device), discount_arr[:-1]], dim=0), dim=0).detach()

            # Actor Loss
            actor_loss = -(weights * (lambda_target + cfg.actor_entropy_scale * imagined_action_entropys)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(actor.parameters(), cfg.gradient_clipping)
            actor_optimizer.step()
            
            # Critic Loss
            value_pred = critic(flatten_imagined_states.detach(), flatten_imagined_rnn_hiddens.detach()).view(cfg.imagination_horizon, -1)
            critic_loss = -(weights.detach() * MSE(value_pred).log_prob(lambda_target.detach())).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), cfg.gradient_clipping)
            critic_optimizer.step()
            
            if (iteration + 1) % cfg.slow_critic_update == 0:
                target_critic.load_state_dict(critic.state_dict())
    torch.save(agent.state_dict(), "agent_intrinsic.pth")

if __name__ == "__main__":
    main()