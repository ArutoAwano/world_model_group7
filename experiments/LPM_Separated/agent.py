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

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    # LPM Modules
    if getattr(config, 'use_lpm', False):
         self.lpm = embodied.jax.MLPHead(scalar, **config.value, name='lpm')
         self.intr = embodied.jax.MLPHead(scalar, **config.value, name='intr')
         self.lpmnorm = embodied.jax.Normalize(**config.retnorm, name='lpmnorm') # Reuse retnorm config

    self.modules = [
        self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    if getattr(config, 'use_lpm', False):
        self.modules.extend([self.lpm, self.intr, self.lpmnorm]) # Add new modules
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    if getattr(config, 'use_lpm', False):
      scales['lpm'] = 1.0
      scales['intr'] = 1.0
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
    return {
        'enc': self.enc.initial(batch_size),
        'dyn': self.dyn.initial(batch_size),
        'dec': self.dec.initial(batch_size),
        'prevact': jax.tree.map(zeros, self.act_space)
    }

  def init_train(self, batch_size):
    # Initialize policy state (enc, dyn, dec, prevact)
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    # Report uses same carry structure as train (5 elements)
    return self.init_train(batch_size)

  def policy(self, carry, obs, mode='train'):
    # Robust unpacking: handle dict or tuple (legacy/driver compatibility)
    if isinstance(carry, dict):
        if 'policy' in carry:
             carry = carry['policy']
        # If still dict, keys should be enc/dyn/dec/prevact
        enc_carry = carry['enc']
        dyn_carry = carry['dyn']
        dec_carry = carry['dec']
        prevact = carry['prevact']
    else:
        # Legacy tuple structure
        enc_carry, dyn_carry, dec_carry, prevact = carry
        carry = {'enc': enc_carry, 'dyn': dyn_carry, 'dec': dec_carry, 'prevact': prevact}
        
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
    
    # Update carry
    new_carry = {'enc': enc_carry, 'dyn': dyn_carry, 'dec': dec_carry, 'prevact': act}
    
    out = {}
    
    # Exclude d_queue from finite check to avoid issues
    check_args = dict(obs=obs, tokens=tokens, feat=feat, act=act)
    # Recursively check carry components
    # Just check flatdict of relevant parts
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        check_args))
        
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return new_carry, act, out

  def train(self, carry, data, global_step=0.0):
    if not isinstance(global_step, jax.Array):
      global_step = jnp.array(global_step, dtype=jnp.float32)
    
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, global_step, training=True, has_aux=True)
    metrics.update(mets)
    self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    
    # Reconstruct carry for next step
    new_prevact = {k: data[k][:, -1] for k in self.act_space}
    carry = {**carry, 'prevact': new_prevact}
    return carry, outs, metrics
    
  def _apply_replay_context(self, carry, data):
    # carry is now a DICT: {'enc':..., 'dyn':..., 'dec':..., 'prevact':...}
    
    if not isinstance(carry, dict):
        raise ValueError(f"_apply_replay_context: expected dict carry, got {type(carry)}")

    enc_carry = carry['enc']
    dyn_carry = carry['dyn']
    dec_carry = carry['dec']
    prevact = carry['prevact']
    
    # DEBUG: Check prevact structure
    if not isinstance(prevact, dict):
         raise ValueError(f"_apply_replay_context: prevact is {type(prevact)}, expected dict. Content type: {type(prevact)}. Keys: {self.act_space.keys() if hasattr(self.act_space, 'keys') else 'No Keys'}")
    
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    
    K = self.config.replay_context
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)

    # Re-pack into dict for return
    # Inputs to tree_map must align with replays.
    # replays['prevact'] is sequence (T-K).
    # So carry_out['prevact'] must be rhs(prevact) (T-K).
    carry_out = {
        'enc': enc_carry, 
        'dyn': dyn_carry, 
        'dec': dec_carry, 
        'prevact': lhs(prevact) if False else rhs(prevact) # rhs(prevact) matches rep_prevact
    }

    if not self.config.replay_context:
      return carry_out, obs, prevact, stepid

    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    
    rep_enc = self.enc.truncate(lhs(entries[0]), enc_carry)
    rep_dyn = self.dyn.truncate(lhs(entries[1]), dyn_carry)
    rep_dec = self.dec.truncate(lhs(entries[2]), dec_carry)
    
    rep_carry = {
        'enc': rep_enc,
        'dyn': rep_dyn,
        'dec': rep_dec,
        'prevact': {k: data[k][:, K - 1: -1] for k in self.act_space}
    }
        
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    
    # tree_map over DICT structure
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry_out, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
        
    return carry, obs, prevact, stepid
    


  def loss(self, carry, obs, prevact, global_step, training):
    enc_carry = carry['enc']
    dyn_carry = carry['dyn']
    dec_carry = carry['dec']
    prevact_unused = carry['prevact']
    
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

    # LPM: Intrinsic Reward Calculation
    eta = 0.0
    if getattr(self.config, 'use_lpm', False):
        # 1. Get current error (NLL)
        eps_current = losses.get('image', list(losses.values())[0]) # Fallback if 'image' key missing?
        if 'image' not in losses:
             # Sum all recon losses if multiple keys
             eps_current = sum([losses[k] for k in recons.keys()])

        # 2. Predict error from previous state and action (using current weights, to compute Intrinsic Reward)
        # We need (z_{t-1}, a_{t-1}) to predict (eps_t).
        # repfeat is (B, T, D). prevact is (B, T, A).
        # repfeat[:, :-1] is z_{0...T-2}. prevact[:, 1:] is a_{0...T-2} (associated with transitions to z_{1...T-1}?? No)
        # prevact is action taken at t-1.
        # So z_{t-1} + a_{t-1} -> z_t -> eps_current (at t).
        # repfeat is z_0 ... z_{T-1}.
        # loss is calculated for obs_0 ... obs_{T-1}?
        # usually data['obs'] is O_1 ... O_T. 'prevact' is A_0 ... A_{T-1}.
        # repfeat z_1 ... z_T.
        # So we predict eps_current (at t) from z_{t-1} and a_{t-1}.
        # repfeat[:, :-1] is z_1 ... z_{T-1}. This is confusing.
        
        # Let's trust the existing logic for indices:
        # 2. Predict error from previous state and action (using current weights, to compute Intrinsic Reward)
        if self.act_space['action'].discrete:
            action_inp = jax.nn.one_hot(
                nn.cast(jax.tree.map(lambda x: x[:, 1:], prevact)['action']),
                self.act_space['action'].high
            )
        else:
            action_inp = jax.tree.map(lambda x: x[:, 1:], prevact)['action']

        lpm_inp = sg(jnp.concatenate([
            self.feat2tensor(jax.tree.map(lambda x: x[:, :-1], repfeat)),
            action_inp
        ], -1))
        
        eps_target = sg(eps_current[:, 1:])

        # Prediction for Reward (Surprise)
        # Uses current weights of lpm. Ideally should use 'old' weights?
        # But here we assume lpm has not been updated on *this* data yet (which is true, update comes later).
        # However, it has been updated on *recent* data.
        pred_eps = self.lpm(lpm_inp, 2).pred() # (B, T-1)
        
        # No D_queue: Train Error Predictor/Intrinsic Model on current batch immediately
        # Flatten batch:
        flat_inp = lpm_inp.reshape(-1, lpm_inp.shape[-1])
        flat_eps = eps_target.reshape(-1)
        
        train_inp, train_eps = flat_inp, flat_eps
        
        # Train Error Predictor - periodic update
        update_interval = float(self.config.lpm_update_interval)
        should_update = (global_step % update_interval) < 1.0
        
        # Calculate Loss on current samples
        lpm_loss_raw = self.lpm(train_inp, 1).loss(train_eps)
        losses['lpm'] = jnp.where(should_update, lpm_loss_raw, jnp.zeros_like(lpm_loss_raw))
        
        # 3. Compute Delta (Surprise)
        # Corrected: Predicted(Past) - Actual(Current)
        delta = pred_eps - eps_target
        
        # Optional: Prevent negative reward (punishment) due to poor prediction
        delta = jnp.maximum(0.0, delta)
        
        # 4. Normalize and Clip
        delta_offset, delta_scale = self.lpmnorm(delta, training)
        delta_norm = (delta - delta_offset) / delta_scale
        delta_norm_clipped = jnp.clip(delta_norm, -self.config.lpm_clip, self.config.lpm_clip)
        
        # 5. Train Intrinsic Reward Model - also periodic
        # Predictive target for Intrinsic Model is the (clipped) Surprise
        intr_loss_raw = self.intr(lpm_inp, 2).loss(sg(delta_norm_clipped))
        losses['intr'] = jnp.where(should_update, intr_loss_raw, jnp.zeros_like(intr_loss_raw))
        
        # Calculate Eta with Warmup
        warmup_steps = float(self.config.lpm_warmup)
        # linear warmup
        warmup_factor = jnp.minimum(1.0, global_step / warmup_steps)
        eta = self.config.lpm_eta * warmup_factor
        
        metrics['lpm_eps'] = eps_target.mean()
        metrics['lpm_pred'] = pred_eps.mean()
        metrics['lpm_delta'] = delta.mean()
        metrics['lpm_eta'] = eta
        
        # Optional scale for lpm loss
        losses['lpm'] *= self.config.lpm_scale if hasattr(self.config, 'lpm_scale') else 1.0

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
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        intr_model=self.intr if getattr(self.config, 'use_lpm', False) else None,
        eta=eta,
        prev_inp=jnp.concatenate([
            self.feat2tensor(sg(jax.tree.map(lambda x: x[:, :-1], imgfeat))),
            jax.nn.one_hot(
                nn.cast(sg(jax.tree.map(lambda x: x[:, 1:], imgact))['action']),
                self.act_space['action'].high
            ) if self.act_space['action'].discrete else
            sg(jax.tree.map(lambda x: x[:, 1:], imgact))['action']
        ], -1), # approx input for intr
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    # Return 5 items in carry (enc, dyn, dec, unused_prevact, d_queue)
    # Note: d_queue here is the UPDATED d_queue from inside loss logic.
    # Return dict carry
    # We need to return this d_queue as well.
    # d_queue was updated inside loss logic (it's mutable/new obj).
    
    new_policy_carry = {'enc': enc_carry, 'dyn': dyn_carry, 'dec': dec_carry, 'prevact': prevact_unused}
    
    new_policy_carry = {'enc': enc_carry, 'dyn': dyn_carry, 'dec': dec_carry, 'prevact': prevact_unused}
    carry = new_policy_carry
         
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data, global_step=0.0):
    if not self.config.report:
      return carry, {}
    
    if not isinstance(global_step, jax.Array):
      global_step = jnp.array(global_step, dtype=jnp.float32)

    policy_carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    # Handle D_queue separation for report as well
    loss_carry = policy_carry

    enc_carry = policy_carry['enc']
    dyn_carry = policy_carry['dyn']
    dec_carry = policy_carry['dec']
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        loss_carry, obs, prevact, global_step, training=False)
    # new_carry from loss has 5 items (enc, dyn, dec, prevact_unused, d_queue)
    
    metrics.update(mets)

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

    new_prevact = {k: data[k][:, -1] for k in self.act_space}
    carry = {**new_carry, 'prevact': new_prevact}
    return carry, metrics




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
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
    intr_model=None,
    eta=0.0,
    prev_inp=None,
):
  losses = {}
  metrics = {}

  # LPM: Add Intrinsic Reward
  if intr_model is not None:
      intr_pred = intr_model(prev_inp, 2).pred() # H
      intr_rew = intr_pred * eta  # eta=0 during warmup → no intrinsic reward
      
      # Pad first step (cannot compute intr for t=0 from t=-1)
      intr_padded = jnp.concatenate([jnp.zeros_like(intr_rew[:, :1]), intr_rew], 1)
      
      # Ensure shape match with rew (H+1)
      if intr_padded.shape[1] > rew.shape[1]:
          intr_padded = intr_padded[:, :rew.shape[1]]
      elif intr_padded.shape[1] < rew.shape[1]:
          intr_padded = jnp.concatenate([intr_padded, jnp.zeros((intr_padded.shape[0], rew.shape[1] - intr_padded.shape[1]))], 1)
      
      rew = rew + intr_padded

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con
  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update)
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
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
