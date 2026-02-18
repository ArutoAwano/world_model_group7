import copy
import glob
import os
import time
from tqdm import tqdm
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from exploration import algo, utils
from exploration.arguments import get_args
from exploration.envs import make_vec_envs
from exploration.model import Policy
from exploration.storage import RolloutStorage

# Remove LBS import and add our curiosity models
from stable_baselines3.common.running_mean_std import RunningMeanStd
import exploration.environments
from exploration.algo.random import RandomAgent, RandomPolicy

# Import our curiosity models
from exploration.models.improve import LearningProgressCuriosity
from exploration.models.ama import AMAPix2PixCuriosity
from exploration.models.unet_improve import UNetLearningProgressCuriosity
from exploration.models.RND import RandomNetworkDistillationCuriosity
from exploration.models.icm import IntrinsicCuriosityModuleCuriosity
from exploration.models.ensemble import EnsembleDisagreementCuriosity
# from exploration.models.tdd import TemporalDistanceDensityCuriosity
from exploration.models.tdd2 import TemporalDistanceDensityCuriosity
from exploration.models.eme import EffectiveMetricExploration

import matplotlib.pyplot as plt
def show_image(obs_normal):
    
    # Extract displayable frames
    def extract_frame(obs):
        if len(obs.shape) == 3:
            if obs.shape[0] <= 4:  # Frame stack (C, H, W)
                return obs[-1]  # Last frame
            else:  # (H, W, C)
                return obs[:, :, 0]
        return obs
    
    frame_normal = extract_frame(obs_normal)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    
    # Normal frame
    axes.imshow(frame_normal, cmap='gray', vmin=0, vmax=255)
    axes.set_title(f'Normal Action\nShape: {frame_normal.shape}\nVariance: {np.var(frame_normal):.1f}')
    axes.axis('off')

    plt.tight_layout()
    
    # Display
    plt.show()
    plt.close()

def main():
    # Setup
    args = get_args()
    print(f"[DEBUG] args.noisy_tv: {getattr(args, 'noisy_tv', 'NOT SET')}")
    # warnings
    # We follow the structure of opneAI baseline
    args.us_bn = True
    args.us_ln = True
    if args.use_bn:
        print("Using BatchNorm in the model")
    if args.use_ln:
        print("Using LayerNorm in the model")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    # writer = SummaryWriter(logdir=log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Environment and Policy 
    if hasattr(args, 'noisy') and args.noisy == True:
        print("creat noisy env")
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, noisy=True, num_noop_actions=args.noop, num_random_actions=args.randop)
    else:
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, noisy=False, num_noop_actions=args.noop)

    train_model = False
    is_vision = len(envs.observation_space.shape) > 1

    if args.algo == 'random':
        actor_critic = RandomPolicy(envs, args.num_processes)
    else:
        actor_critic = Policy(
            envs.observation_space,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)

    # Algorithm - Now includes AMA
    if args.algo in ['ppo-improvement', 'mse', 'unet', 'rnd', 'icm', 'ensemble', 'tdd', 'eme']:
        obs_dim = envs.observation_space.shape
        print("The obs dim of the environment is: +++++")
        print(obs_dim)
        discrete_action = envs.action_space.__class__.__name__ == "Discrete"
        if envs.action_space.__class__.__name__ == "Discrete":
            act_dim = envs.action_space.n
        elif envs.action_space.__class__.__name__ == "Box":
            act_dim = envs.action_space.shape[0]
        elif envs.action_space.__class__.__name__ == "MultiBinary":
            act_dim = envs.action_space.shape[0]
        else:
            raise NotImplementedError

        intr_ret_rms = RunningMeanStd()
        intr_ret = np.zeros((args.num_processes, 1))
        
        if args.use_dones:
            ext_coeff = 1. # 1 # Sparse tasks: 1000
            int_coeff = 1e-5  # 0.001 or 0.01
        else:
            ext_coeff = 1. # 1 # Sparse tasks: 1000
            int_coeff = 1.  # 0.001 or 0.01

        train_model = True

        if is_vision:
            hidden_dim = 512
            state_dim = 512
        else:
            hidden_dim = 32 
            state_dim = obs_dim
    elif args.algo == 'ama':
        print("Initializing AMA (Aleatoric Mapping Agents) curiosity method")
        obs_dim = envs.observation_space.shape
        print(f"The obs dim of the environment is: {obs_dim}")
        
        discrete_action = envs.action_space.__class__.__name__ == "Discrete"
        if envs.action_space.__class__.__name__ == "Discrete":
            act_dim = envs.action_space.n
        elif envs.action_space.__class__.__name__ == "Box":
            act_dim = envs.action_space.shape[0]
        elif envs.action_space.__class__.__name__ == "MultiBinary":
            act_dim = envs.action_space.shape[0]
        else:
            raise NotImplementedError

        intr_ret_rms = RunningMeanStd()
        intr_ret = np.zeros((args.num_processes, 1))
        
        if args.use_dones:
            ext_coeff = 1. 
            int_coeff = 1e-3  # Start with smaller coefficient for AMA
        else:
            ext_coeff = 0. 
            int_coeff = 1.

        train_model = True

        if is_vision:
            hidden_dim = 512
            state_dim = 512
        else:
            hidden_dim = 32 
            state_dim = obs_dim

        print(f"AMA parameters: eta={args.ama_eta}, lambda={args.ama_lambda}")

    if args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo-improvement':
        # Initialize our LearningProgressCuriosity model
        model = LearningProgressCuriosity(
            obs_size=obs_dim, 
            act_size=act_dim, 
            state_size=state_dim, 
            device=device,
            eta=0.4,
            pred_lr=args.lr,
            uncertainty_lr=args.lr  # Higher LR for uncertainty model
        )
    elif args.algo == 'mse':
        model = LearningProgressCuriosity(
            obs_size=obs_dim, 
            act_size=act_dim, 
            state_size=state_dim, 
            device=device,
            eta=0.4,
            pred_lr=args.lr,
            uncertainty_lr=args.lr,  # Higher LR for uncertainty model
            ismse = True
        )
    elif args.algo == 'ama':
        # Initialize our AMA model
        model = AMAPix2PixCuriosity(
            obs_dim,
            act_dim,
            state_dim,
            device,
            lr=args.lr, feat_dim=512, uncertainty_penalty=1.0, 
            reward_scaling=1.0, clip_val=10
        )
    elif args.algo == 'unet':
        # Initialize our UNet-based LearningProgressCuriosity model
        model = UNetLearningProgressCuriosity(
            obs_size=obs_dim, 
            act_size=act_dim, 
            state_size=state_dim, 
            device=device,
            eta=0.4,
            pred_lr=args.lr,
            uncertainty_lr=args.lr,  # Higher LR for uncertainty model
            feat_dim=hidden_dim
        )
    elif args.algo == 'icm':
        model = IntrinsicCuriosityModuleCuriosity(
            obs_size=obs_dim,
            act_size=act_dim,
            state_size=state_dim,
            device=device,
            eta=200,
            beta=0.2,
            lr=args.lr
        )
    elif args.algo == 'rnd':
        # Initialize our Random Network Distillation model
        model = RandomNetworkDistillationCuriosity(
            obs_size=obs_dim,
            act_size=act_dim,
            state_size=state_dim,
            device=device,
            eta=1.0,
            lr=args.lr,  # Higher LR for RND predictor network
            embedding_dim=hidden_dim
    )
    elif args.algo == 'ensemble':
        model = EnsembleDisagreementCuriosity(
            obs_size=obs_dim,
            act_size=act_dim,
            state_size=state_dim,
            device=device,
            eta=1.0,
            num_ensemble=5,
            lr=args.lr
        )
    # elif args.algo == 'tdd':
    #     model = TemporalDistanceDensityCuriosity(
    #         obs_size=obs_dim,
    #         act_size=act_dim,
    #         state_size=state_dim,
    #         device=device,
    #         eta=1.0,
    #         lr=args.lr
    #     )
    elif args.algo == 'tdd':
        model = TemporalDistanceDensityCuriosity(
            obs_size=obs_dim,
            act_size=act_dim,
            state_size=state_dim,
            device=device,
            eta=0.5,                      # Higher for short episodes
            lr=args.lr,              # Fast representation learning
            features_dim=256,             # Matches original
            embedding_dim=64,             # Matches original  
            energy_fn='mrn_pot',          # Original paper's best
            loss_fn='infonce',            # Standard
            aggregate_fn='min',           # Conservative novelty
            max_history_size=2000,        # Large for persistent memory
            persistent_memory=True,       # Critical for short episodes
            temperature=1.0,              # Original default
            logsumexp_coef=0.1           # Original default
        )
    elif args.algo == 'eme':
        model = EffectiveMetricExploration(
            obs_size=obs_dim,    # Atari frame stack
            act_size=act_dim,              # Atari action space
            state_size=state_dim,
            device='cuda',
            eta=0.01,                 # From paper
            lr=args.lr,               # From paper
            embedding_dim=512,
            ensemble_size=6
        )
    elif args.algo == 'random':
        agent = RandomAgent()

    if train_model:
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
        
        # Note: Our model has its own optimizers, no need for external optimizer
    print(f"[DEBUG] Environment action space size: {envs.action_space.n}")
    print(f"[DEBUG] Environment name: {args.env_name}")
        # PRINT ACTION SPACE HERE
    print(f"\n🎮 ACTION SPACE DEBUG:")
    print(f"   Action space: {envs.action_space}")
    print(f"   Action space type: {type(envs.action_space)}")
    print(f"   Action space size: {envs.action_space.n}")
    print(f"   Environment name: {args.env_name}")
    # Setup rollouts and Episode Queue
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs_rms = None
    if is_vision:
        # SIMPLIFIED APPROACH: Create clean environment for initialization
        print('Creating clean initialization environment...')
        
        # Create environment WITHOUT any wrappers that modify action space
        init_env = make_vec_envs(args.env_name, args.seed, 1, args.gamma, None, 'cpu', False, 
                                noisy=False, num_random_actions=1, num_noop_actions=0)

        # normalize obs
        try:
            obs, info = init_env.reset()
        except ValueError:
            obs = init_env.reset()
            info = {}
        
        print('Start to initialize observation normalization parameter.....')
        obs_init = []
        steps = 0
        pre_obs_norm_step = int(1e4)
        
        # FIXED: Use simple random actions instead of RandomPolicy
        while steps < pre_obs_norm_step:
            steps += 1
            
            # Generate random action directly from action space
            random_action = init_env.action_space.sample()
            action_tensor = torch.tensor([[random_action]])
            
            try:
                no, _, _, _ = init_env.step(action_tensor)
                obs_init.append(np.asarray(no.detach().cpu()).astype(np.float32).copy())
            except Exception as e:
                print(f"Error at step {steps}: {e}")
                print(f"Action: {random_action}, Action space: {init_env.action_space}")
                break
                
            if steps % 1000 == 0:
                print(f"Initialization step {steps}/{pre_obs_norm_step}")
        
        obs_init = np.array(obs_init)
        obs_mean = torch.Tensor(np.mean(obs_init, 0).astype(np.float32)).to(device)
        obs_std = torch.Tensor([np.std(obs_init, 0).mean().astype(np.float32)]).to(device)

        # Clean up
        init_env.close()
        del init_env

        print('End to initialize...')
        
        obs_rms = RunningMeanStd()
        obs_rms.mean = obs_mean
        obs_rms.var = obs_std**2
    else:
        obs_mean = 0 
        obs_std = 1

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_return = -1e+8
    best_length = -1e+8

    if args.env_name in ['MagellanAnt-v2', 'MountainCarSparse-v0', 'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving', 'HalfCheetahSparse-v3']:
        states_buffer = []
        update_state_buffer = []
        overall_blocks = []

    # Training
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    # print("Start training")
    for j in tqdm(range(num_updates)):
        start = time.time()
        update_episode_returns = []
        update_episode_lengths = []

        if args.use_linear_lr_decay and args.algo != 'random':
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                args.lr)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    (rollouts.obs[step] - obs_mean) / (obs_std + 1e-8), rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            # print("Action selected...")
            # It causes problem otherwise
            if args.env_name == 'MarioBrosNoFrameskip-v4':
                next_obs, reward, done, infos = envs.step(action.to('cpu'))
            else:    
                # Obs reward and next obs
                next_obs, reward, done, infos = envs.step(action)
            # print("Step in environment...")

            # This is used to show the image, make sure the environment works correctly
            # obs_normal_np = next_obs[0].cpu().numpy()
            # show_image(obs_normal_np)
            if isinstance(next_obs, torch.Tensor):
                next_obs = next_obs.clone().detach()

            for info in infos:
                if 'episode' in info.keys():
                    discounted_returns = info['episode']['r']
                    ep_length = info['episode']['l']
                    episode_rewards.append(discounted_returns)
                    episode_lengths.append(ep_length)
                    update_episode_returns.append(discounted_returns)
                    update_episode_lengths.append(ep_length)
                    if discounted_returns > best_return:
                        best_return = discounted_returns
                    if ep_length > best_length:
                        best_length = ep_length
                if args.env_name in ['MagellanAnt-v2', 'MountainCarSparse-v0',  'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving', 'HalfCheetahSparse-v3']:
                    states_buffer.append(info['obs'])
                    update_state_buffer.append(info['obs'])
            
            if args.use_dones:
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
            else:
                masks = torch.FloatTensor([[1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[1.0] for info in infos])

            rollouts.insert(next_obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            obs = next_obs

        # Normalize observations
        rollouts.obs = ( rollouts.obs - obs_mean) / (obs_std + 1e-8)
        # print("finish interact...")
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        
        # print("grab curiosity reward...")
        if train_model:
            o = rollouts.obs[:-1].detach().reshape(-1, *obs_dim)
            no = rollouts.obs[1:].detach().reshape(-1, *obs_dim)
            ext_reward = rollouts.rewards.detach() 

            if is_vision:
                a = rollouts.actions.detach().reshape(-1, 1)
                # For discrete actions, we just need the action indices
                a = a.squeeze().long()
            else:
                a = rollouts.actions.detach().reshape(-1, act_dim)

            max_size = 2048
            if o.shape[0] > max_size:
                curiosities = []
                with torch.no_grad():
                    for indx in range(0, o.shape[0], max_size):
                        # Convert to numpy for our curiosity function
                        o_batch = o[indx:indx+max_size].cpu().numpy()
                        a_batch = a[indx:indx+max_size].cpu().numpy()
                        no_batch = no[indx:indx+max_size].cpu().numpy()
                        
                        c = model.curiosity(o_batch, a_batch, no_batch)
                        curiosities.append(torch.from_numpy(c).to(device))
                curiosity = torch.cat(curiosities)
            else:
                with torch.no_grad():
                    # Convert to numpy for our curiosity function
                    o_np = o.cpu().numpy()
                    a_np = a.cpu().numpy() 
                    no_np = no.cpu().numpy()
                    
                    curiosity_np = model.curiosity(o_np, a_np, no_np)
                    curiosity = torch.from_numpy(curiosity_np).to(device)
            
            intr_rew = curiosity.reshape(ext_reward.shape).detach().cpu().numpy()
            # intr_rew = np.clip(intr_rew, -3 * np.sqrt(intr_ret_rms.var), 3 * np.sqrt(intr_ret_rms.var))

            # upd_intr_ret = []
            # for idx in range(intr_rew.shape[0]):
            #     intr_ret = intr_ret * args.gamma + intr_rew[idx]
            #     upd_intr_ret.append(intr_ret)

            # upd_intr_ret = np.reshape(np.stack(upd_intr_ret), [args.num_steps * args.num_processes, 1])
            # mean, std, count = np.mean(upd_intr_ret), np.std(upd_intr_ret), len(upd_intr_ret)
            # intr_ret_rms.update_from_moments(mean, std ** 2, count)

            # intr_rew = intr_rew / np.sqrt(intr_ret_rms.var + 1e-8) 

            rollouts.rewards = ext_coeff * ext_reward + int_coeff * torch.Tensor(intr_rew).to(device) 

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        # print("update agent")
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        # model update
        # print("update curiosity model...")
        if train_model:
            batch_size = args.num_steps * args.num_processes // args.num_mini_batch
            pred_loss_total = 0
            uncertainty_loss_total = 0
            
            # Train curiosity model only once per update (since our train() method already does 3 epochs)
            data_generator = rollouts.model_generator(args.num_mini_batch)
            for sample in data_generator:
                obs_batch, next_obs_batch, act_batch = sample
                
                if is_vision:
                    # For discrete actions, just squeeze to get indices
                    act_batch = act_batch.squeeze().long()

                # Convert to numpy for our train method
                obs_np = obs_batch.cpu().numpy()
                next_obs_np = next_obs_batch.cpu().numpy()
                act_np = act_batch.cpu().numpy()
                
                # Train our model (already does 3 epochs internally)
                pred_loss, uncertainty_loss = model.train(obs_np, act_np, next_obs_np)
                pred_loss_total += pred_loss
                uncertainty_loss_total += uncertainty_loss
                
            # Average losses over all mini-batches (not epochs since we only do 1)
            num_batches = args.num_mini_batch
            model_pred_loss = pred_loss_total / num_batches
            model_uncertainty_loss = uncertainty_loss_total / num_batches

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            print("try to create path" + ("+" * 30))
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            try:
                os.makedirs(os.path.join(save_path,"ALE"))
            except OSError:
                pass
            
            # Handle observation normalization parameters
            if obs_rms is not None:
                ob_rms = obs_rms
            else:
                vec_normalize = utils.get_vec_normalize(envs)
                ob_rms = getattr(vec_normalize, 'obs_rms', None) if vec_normalize is not None else None
            
            # Save only the policy and normalization parameters (not the full environment)
            print('Saving policy...')
            
            # Handle different policy types
            if args.algo == 'random':
                # RandomPolicy doesn't have state_dict, save minimal info
                save_dict = {
                    'algorithm': 'random',
                    'obs_rms': ob_rms,
                    'args': args,
                    'obs_mean': obs_mean if obs_rms is None else None,
                    'obs_std': obs_std if obs_rms is None else None,
                    'action_space_n': getattr(actor_critic.action_space, 'n', None)
                }
            else:
                # Regular PyTorch models
                save_dict = {
                    'actor_critic_state_dict': actor_critic.state_dict(),
                    'obs_rms': ob_rms,
                    'args': args,
                    'obs_mean': obs_mean if obs_rms is None else None,
                    'obs_std': obs_std if obs_rms is None else None
                }
                
                # For non-random algorithms, also save optimizer state
                if hasattr(agent, 'optimizer'):
                    save_dict['optimizer_state_dict'] = agent.optimizer.state_dict()
            
            torch.save(save_dict, os.path.join(save_path, args.env_name + ".pt"))
            
            # Also save our curiosity model if applicable
            if train_model and args.algo in ['ppo-improvement', 'ama']:
                try:
                    model.save(os.path.join(save_path, args.env_name + "_curiosity.pt"))
                    print('Saved curiosity model...')
                except Exception as e:
                    print(f'Warning: Could not save curiosity model: {e}')
            
            print(f'Saved model to {save_path}')
        if j % args.log_interval == 0 and len(episode_rewards) > 0 and len(update_episode_returns) > 0:
            print("")
            
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {:.2e}, FPS {} \n Last {} episodes: mean reward {:.1f}±{:.1f} (best: {:.1f}), mean length {:d}±{:d} (best: {:d}).\n Recent episodes ({:d}): reward {:.1f}±{:.1f}, length {:.1f}±{:.1f}.\n DistEntropy: {:.2e} CriticLoss: {:.2e} ActorLoss: {:.2e}"
                .format(j, float(total_num_steps), int( args.num_steps * args.num_processes / (end - start)), len(episode_rewards), 
                        np.mean(episode_rewards), np.std(episode_rewards), best_return, int(np.mean(episode_lengths)), int(np.std(episode_lengths)), int(best_length),
                        len(update_episode_returns), np.mean(update_episode_returns), np.std(update_episode_returns), np.mean(update_episode_lengths), np.std(update_episode_lengths),
                        dist_entropy, value_loss, action_loss))
            
            if train_model:
                print(" IntRew: {:.3f}±{:.3f} ExtRew: {:.3f}±{:.3f} RollRew: {:.3f}±{:.3f}; RollRet: {:.3f}±{:.3f}"
                    .format(curiosity.mean(), curiosity.std(), ext_reward.mean(), ext_reward.std(), rollouts.rewards.mean(), rollouts.rewards.std(), rollouts.returns.mean(), rollouts.returns.std() ))
                
                # Log our model losses - different naming for AMA
                if args.algo == 'ama':
                    print(" AMALoss: {:.3f}".format(model_pred_loss))
                else:
                    print(" PredLoss: {:.3f} UncertaintyLoss: {:.3f}".format(model_pred_loss, model_uncertainty_loss))
            
            if args.env_name in ['MountainCarSparse-v0',  'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving', 'HalfCheetahSparse-v3']:

                if args.env_name in ['MountainCarSparse-v0',  'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving']:
                    print(" " + args.env_name)
                    pos_range = infos[-1]['pos_range']
                    print("  PosRange: ({:.3f}, {:.3f}) PosInterval: {:.3f} ({:.3f})"
                        .format(pos_range[0], pos_range[1], pos_range[1] - pos_range[0], (pos_range[1] - pos_range[0]) / 1.8))
                    vel_range = infos[-1]['vel_range']
                    print("  VelRange: ({:.3f}, {:.3f}) VelInterval: {:.3f} ({:.3f})"
                        .format(vel_range[0], vel_range[1], vel_range[1] - vel_range[0], (vel_range[1] - vel_range[0]) / 0.14))

                    from exploration.environments.mountain_car_sparse import rate_buffer_with_blocks
                    n_blocks = 10
                    coverage, blocks = rate_buffer_with_blocks(update_state_buffer, n_blocks=n_blocks)
                    print("  UpdateStatesCoverage: {} ({:.3f})".format(coverage, (coverage / n_blocks**2 * 100)))
                    update_state_buffer = []
                    if len(overall_blocks) == 0:
                        overall_blocks = blocks
                    else:
                        overall_blocks = np.unique(np.concatenate([blocks, overall_blocks], axis=0), axis=0)
                    coverage = len(overall_blocks)
                    print("  OverallStatesCoverage: {} ({:.3f})".format(coverage, (coverage / n_blocks**2 * 100)))

                elif args.env_name == 'HalfCheetahSparse-v3':
                    print(" HalfCheetahSparse-v3:")
                    angle_range = infos[-1]['angle_range']
                    print("  AngleVelRange: ({:.3f}, {:.3f}) AngleVelRange: {:.3f}"
                        .format(angle_range[0], angle_range[1], angle_range[1] - angle_range[0]))
                    pos_range = infos[-1]['pos_range']
                    print("  PosRange: ({:.3f}, {:.3f}) PosInterval: {:.3f}"
                        .format(pos_range[0], pos_range[1], pos_range[1] - pos_range[0]))
                    vel_range = infos[-1]['vel_range']
                    print("  VelRange: ({:.3f}, {:.3f}) VelInterval: {:.3f}"
                        .format(vel_range[0], vel_range[1], vel_range[1] - vel_range[0]))

                    from exploration.environments.half_cheetah_sparse import rate_buffer_with_blocks
                    n_blocks = 10
                    coverage, blocks = rate_buffer_with_blocks(update_state_buffer, n_blocks=n_blocks)
                    print("  UpdateStatesCoverage: {} ({:.3f})".format(coverage, (coverage / n_blocks**2 * 100)))
                    update_state_buffer = []
                    if len(overall_blocks) == 0:
                        overall_blocks = blocks
                    else:
                        overall_blocks = np.unique(np.concatenate([blocks, overall_blocks], axis=0), axis=0)
                    coverage = len(overall_blocks)
                    print("  OverallStatesCoverage: {} ({:.3f})".format(coverage, (coverage / n_blocks**2 * 100)))
            
            if args.env_name == 'MagellanAnt-v2':
                print(" MagellanAnt-v2")
                from exploration.environments.magellan_ant import rate_buffer
                coverage, blocks = rate_buffer(update_state_buffer)
                print("  UpdateMazeCoverage: {} ({:.3f})"
                    .format(coverage, (coverage / 7 * 100)))
                update_state_buffer = []
                if len(overall_blocks) == 0:
                    overall_blocks = blocks
                else:
                    overall_blocks = set([*list(blocks), *list(overall_blocks)]) 
                coverage = len(overall_blocks)
                print("  OverallMazeCoverage: {} ({:.3f})"
                    .format(coverage, (coverage / 7 * 100)))
                
    if args.env_name in ['MagellanAnt-v2', 'MountainCarSparse-v0',  'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving', 'HalfCheetahSparse-v3']:
        np.save(log_dir + '/overall_buffer.npy', states_buffer)
        if args.env_name in ['MountainCarSparse-v0',  'MountainCarStochastic-Frozen',  'MountainCarStochastic-Evolving', 'HalfCheetahSparse-v3']:
            np.save(log_dir + '/overall_blocks.npy', blocks)

if __name__ == "__main__":
    main()
