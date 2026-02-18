import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol')

    self.retnorm_ext = embodied.jax.Normalize(**config.retnorm, name='retnorm_ext')
    self.valnorm_ext = embodied.jax.Normalize(**config.valnorm, name='valnorm_ext')
    self.retnorm_intr = embodied.jax.Normalize(**config.retnorm, name='retnorm_intr')
    self.valnorm_intr = embodied.jax.Normalize(**config.valnorm, name='valnorm_intr')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    # Separated Critics
    # Extrinsic
    self.val_ext = embodied.jax.MLPHead(scalar, **config.value, name='val_ext')
    self.slowval_ext = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval_ext'),
        source=self.val_ext, **config.slowvalue)
    # Intrinsic
    self.val_intr = embodied.jax.MLPHead(scalar, **config.value, name='val_intr')
    self.slowval_intr = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval_intr'),
        source=self.val_intr, **config.slowvalue)

    # Intrinsic Reward Model (Learned Predictor)
    self.intr_rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='intr_rew')
    
    # LPM Specific Modules
    if config.intrinsic_type == 'lpm':
         self.lpm = embodied.jax.MLPHead(scalar, **config.value, name='lpm')
         self.lpmnorm = embodied.jax.Normalize(**config.retnorm, name='lpmnorm')

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, 
        self.val_ext, self.val_intr, self.intr_rew]
    if config.intrinsic_type == 'lpm':
        self.modules.extend([self.lpm, self.lpmnorm])
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    
    # Update scales for Separated Critic
    if 'value' in scales:
        val_scale = scales.pop('value')
        scales['value_ext'] = val_scale
        scales['value_intr'] = val_scale
    if 'repval' in scales:
        repval_scale = scales.pop('repval')
        scales['repval_ext'] = repval_scale
        scales['repval_intr'] = repval_scale
        
        
    # scales for intrinsic models
    if self.config.intrinsic_type in ('lpm', 'recon'):
        scales['intr_rew'] = self.config.intr_reward_scale if hasattr(self.config, 'intr_reward_scale') else 1.0
        
    if self.config.intrinsic_type == 'lpm':
        scales['lpm'] = 1.0
        
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry, data, global_step=0.0):
    if not isinstance(global_step, jax.Array):
      global_step = jnp.array(global_step, dtype=jnp.float32)
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, global_step, training=True, has_aux=True)
    metrics.update(mets)
    metrics.update(mets)
    self.slowval_ext.update()
    self.slowval_intr.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, global_step, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Intrinsic Reward Calculation & Learning
    target_intr_rew = None
    eta = 0.0
    
    # Calculate Eta with Warmup (Common for both types)
    warmup_steps = float(self.config.lpm_warmup)
    warmup_factor = jnp.minimum(1.0, global_step / warmup_steps)
    # For Recon, we rely on warmup. For LPM, we use eta.
    # Note: config.lpm_eta is used for LPM type.
    # config.intr_reward_scale is used for Recon (and LPM scaling).
    
    if self.config.intrinsic_type == 'lpm':
        # --- LPM Logic ---
        # 1. Get current error (NLL)
        eps_current = losses.get('image', list(losses.values())[0]) 
        if 'image' not in losses:
             eps_current = sum([losses[k] for k in recons.keys()])
        
        # 2. Predict error
        # Separate state and action processing
        rep_slice = jax.tree.map(lambda x: x[:, :-1], repfeat)
        act_slice = jax.tree.map(lambda x: x[:, 1:], prevact)
        
        rep_tensor = self.feat2tensor(rep_slice)
        
        # Handle action
        if 'action' in act_slice and self.act_space['action'].discrete:
            act_tensor = jax.nn.one_hot(
                nn.cast(act_slice['action']), 
                self.act_space['action'].high
            )
        else:
            act_values = []
            for k, v in act_slice.items():
                if v.ndim == rep_tensor.ndim - 1:
                     v = v[..., None]
                v = v.astype(jnp.float32)
                act_values.append(v)
            act_tensor = jnp.concatenate(act_values, -1)
            
        lpm_inp = sg(jnp.concatenate([rep_tensor, act_tensor], -1))
        eps_target = sg(eps_current[:, 1:])
        pred_eps = self.lpm(lpm_inp, 2).pred() # (B, T-1)
        losses['lpm'] = self.lpm(lpm_inp, 2).loss(eps_target)
        
        # 3. Compute Delta (Surprise)
        # Corrected: Predicted(Past) - Actual(Current)
        # If Current Error is LOWER than Predicted, Delta is POSITIVE (Reward)
        delta = pred_eps - eps_target
        
        # Optional: Prevent negative reward (punishment) due to poor prediction
        # Use ReLU (max(0, delta)) so early bad predictions don't punish the agent
        delta = jnp.maximum(0.0, delta)
        
        # 4. Normalize and Clip
        delta_offset, delta_scale = self.lpmnorm(delta, training)
        delta_norm = (delta - delta_offset) / delta_scale
        delta_norm_clipped = jnp.clip(delta_norm, -self.config.lpm_clip, self.config.lpm_clip)

        # 5. Train Intrinsic Reward Model
        intr_rew_inp = sg(self.feat2tensor(jax.tree.map(lambda x: x[:, 1:], repfeat)))
        intr_target = sg(delta_norm_clipped)
        losses['intr_rew'] = self.intr_rew(intr_rew_inp, 2).loss(intr_target)
        
        target_intr_rew = delta_norm_clipped
        metrics['lpm_eps'] = eps_target.mean()
        metrics['lpm_pred'] = pred_eps.mean()
        metrics['lpm_delta'] = delta.mean()
        
        # Calculate Eta for LPM
        eta = self.config.lpm_eta * warmup_factor
        metrics['lpm_eta'] = eta
        
        losses['lpm'] *= self.config.lpm_scale if hasattr(self.config, 'lpm_scale') else 1.0

    elif self.config.intrinsic_type == 'recon':
        # --- Reconstruction Error Logic ---
        total_recon_loss = sum([losses[k] for k in recons.keys()]) # (B, T)
        target_intr_rew = jnp.log(total_recon_loss[:, 1:] + 1e-6)
        metrics['recon_loss'] = total_recon_loss.mean()
        
        # Apply Warmup to Recon scale?
        # Typically Recon scale is fixed, but we can apply warmup factor to it.
        # intr_reward_scale * warmup_factor
        # But eta is passed to imag_loss.
        eta = warmup_factor # Use eta as the scaling factor (0.0 -> 1.0)
        metrics['recon_warmup'] = eta

    # Train Intrinsic Reward Model (to predict the target intrinsic reward)
    if target_intr_rew is not None:
        intr_inp = sg(self.feat2tensor(jax.tree.map(lambda x: x[:, 1:], repfeat)))
        intr_rew_loss = self.intr_rew(intr_inp, 2).loss(sg(target_intr_rew))
        losses['intr_rew'] = intr_rew_loss
        metrics['intr_rew_mean'] = target_intr_rew.mean()
        
    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    los, imgloss_out, mets = imag_loss(
        imgact,
        self.rew(inp, 2).pred(),
        self.intr_rew(inp, 2).pred(), # Predicted Intrinsic Reward
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val_ext(inp, 2),
        self.slowval_ext(inp, 2),
        self.val_intr(inp, 2),
        self.slowval_intr(inp, 2),
        self.retnorm_ext, self.valnorm_ext, 
        self.retnorm_intr, self.valnorm_intr,
        self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        config=self.config,
        eta=eta,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay - Extrinsic
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot_ext = imgloss_out['ret_ext'][:, 0].reshape(B, K)
      feat, last, term, rew, boot_ext = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot_ext))
      inp = self.feat2tensor(feat)
      
      # Extrinsic Replay Loss
      los_ext, _, mets_ext = repl_loss(
          last, term, rew, boot_ext,
          self.val_ext(inp, 2),
          self.slowval_ext(inp, 2),
          self.valnorm_ext,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update({f'repval_ext': los_ext['repval']})
      metrics.update(prefix(mets_ext, 'repval_ext'))
      
      # Intrinsic Replay Loss
      # We need distinct intrinsic reward for replay batch. 
      # We calculated target_intr_rew (B, T-1) earlier.
      # We need to slice it to match K (last K steps).
      if target_intr_rew is not None:
           # target_intr_rew is (B, T-1), aligned with obs 1..T
           # We need it aligned with 'rew' which is (B, K)
           # K is min(imag_last, T). If imag_last=0, K=T.
           # Let's assume K <= T-1 for safety, or pad.
           # rew is usually obs['reward'].
           
           # Simplified: Just grab last K from target_intr_rew
           # target_intr_rew corresponds to indices 1..T (length T-1)
           # We need length T (indices 0..T-1) to match K usually.
           # Pad with 0 at start.
           intr_padded = jnp.concatenate([jnp.zeros_like(target_intr_rew[:, :1]), target_intr_rew], 1)
           
           intr_rew_rep = intr_padded[:, -K:] 
           
           # Boot intrinsic
           boot_intr = imgloss_out['ret_intr'][:, 0].reshape(B, K)
           
           los_intr, _, mets_intr = repl_loss(
              last, term, intr_rew_rep, boot_intr,
              self.val_intr(inp, 2),
              self.slowval_intr(inp, 2),
              self.valnorm_intr,
              update=training,
              horizon=self.config.horizon,
              **self.config.repl_loss)
           losses.update({f'repval_intr': los_intr['repval']})
           metrics.update(prefix(mets_intr, 'repval_intr'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data, global_step=0.0):
    if not self.config.report:
      return carry, {}
    
    if not isinstance(global_step, jax.Array):
      global_step = jnp.array(global_step, dtype=jnp.float32)

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, global_step, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, global_step, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)


def imag_loss(
    act, rew, intr_rew_pred, con,
    policy, 
    value_ext, slowvalue_ext,
    value_intr, slowvalue_intr,
    retnorm_ext, valnorm_ext,
    retnorm_intr, valnorm_intr,
    advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
    config=None,
    eta=0.0,
):
  losses = {}
  metrics = {}

  # Intrinsic Reward Scaling
  # intr_rew_pred is computed by self.intr_rew(inp) in caller.
  # We scale it here.
  # Apply eta (Warmup logic)
  intr_rew = intr_rew_pred * config.intr_reward_scale * eta
  
  # Ensure shape match (sometimes 1 step shift)
  if intr_rew.shape[1] > rew.shape[1]:
      intr_rew = intr_rew[:, :rew.shape[1]]
  elif intr_rew.shape[1] < rew.shape[1]:
      # Pad with zeros if needed (or assume 0 for first step)
      intr_rew = jnp.concatenate([jnp.zeros((intr_rew.shape[0], rew.shape[1] - intr_rew.shape[1])), intr_rew], 1)

  # --- Extrinsic Value Calculation ---
  voffset_ext, vscale_ext = valnorm_ext.stats()
  val_ext = value_ext.pred() * vscale_ext + voffset_ext
  slowval_ext = slowvalue_ext.pred() * vscale_ext + voffset_ext
  tarval_ext = slowval_ext if slowtar else val_ext
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  
  ret_ext = lambda_return(last, term, rew, tarval_ext, tarval_ext, disc, lam)
  roffset_ext, rscale_ext = retnorm_ext(ret_ext, update)
  adv_ext = (ret_ext - tarval_ext[:, :-1]) / rscale_ext

  # --- Intrinsic Value Calculation ---
  voffset_intr, vscale_intr = valnorm_intr.stats()
  val_intr = value_intr.pred() * vscale_intr + voffset_intr
  slowval_intr = slowvalue_intr.pred() * vscale_intr + voffset_intr
  tarval_intr = slowval_intr if slowtar else val_intr
  
  ret_intr = lambda_return(last, term, intr_rew, tarval_intr, tarval_intr, disc, lam)
  roffset_intr, rscale_intr = retnorm_intr(ret_intr, update)
  adv_intr = (ret_intr - tarval_intr[:, :-1]) / rscale_intr

  # --- Combined Advantage ---
  adv_combined = adv_ext + config.intr_val_scale * adv_intr
  
  # Normalize Combined Advantage? 
  # Usually standard Dreamer normalizes advantage. We have advnorm.
  # Should we normalize each separately or the combined?
  # The original code normalized 'adv'.
  # Let's normalize the combined advantage to keep stability.
  aoffset, ascale = advnorm(adv_combined, update)
  adv_normed = (adv_combined - aoffset) / ascale

  # --- Policy Loss ---
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  # --- Value Losses ---
  # Extrinsic
  voffset_ext, vscale_ext = valnorm_ext(ret_ext, update)
  tar_normed_ext = (ret_ext - voffset_ext) / vscale_ext
  tar_padded_ext = jnp.concatenate([tar_normed_ext, 0 * tar_normed_ext[:, -1:]], 1)
  losses['value_ext'] = sg(weight[:, :-1]) * (
      value_ext.loss(sg(tar_padded_ext)) +
      slowreg * value_ext.loss(sg(slowvalue_ext.pred())))[:, :-1]

  # Intrinsic
  voffset_intr, vscale_intr = valnorm_intr(ret_intr, update)
  tar_normed_intr = (ret_intr - voffset_intr) / vscale_intr
  tar_padded_intr = jnp.concatenate([tar_normed_intr, 0 * tar_normed_intr[:, -1:]], 1)
  losses['value_intr'] = sg(weight[:, :-1]) * (
      value_intr.loss(sg(tar_padded_intr)) +
      slowreg * value_intr.loss(sg(slowvalue_intr.pred())))[:, :-1]

  # Metrics
  metrics['adv_ext'] = adv_ext.mean()
  metrics['adv_intr'] = adv_intr.mean()
  metrics['adv_combined'] = adv_combined.mean()
  metrics['rew_ext'] = rew.mean()
  metrics['rew_intr'] = intr_rew.mean()
  metrics['ret_ext'] = ret_ext.mean()
  metrics['ret_intr'] = ret_intr.mean()
  metrics['val_ext'] = val_ext.mean()
  metrics['val_intr'] = val_intr.mean()
  
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()

  outs = {}
  outs['ret_ext'] = ret_ext
  outs['ret_intr'] = ret_intr
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
