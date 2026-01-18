import numpy as np
import torch
import torch.distributions as td
from torch.distributions import Normal, OneHotCategoricalStraightThrough
from torch import nn
from torch.nn import functional as F

class MSE(td.Normal):
    def __init__(self, loc, validate_args=None):
        super(MSE, self).__init__(loc, 1.0, validate_args=validate_args)
    @property
    def mode(self):
        return self.mean
    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -((value - self.loc) ** 2) / 2

import math
from numbers import Number
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

class TruncatedStandardNormal(Distribution):
    arg_constraints = {'a': constraints.real, 'b': constraints.real}
    has_rsample = True
    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype: raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1, ).tolist()): raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0, self._dtype_max_lt_1 = eps, 1 - eps
        self._little_phi_a, self._little_phi_b = self._little_phi(self.a), self._little_phi(self.b)
        self._big_phi_a, self._big_phi_b = self._big_phi(self.a), self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a, little_phi_coeff_b = torch.nan_to_num(self.a, nan=math.nan), torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._mode = torch.clamp(torch.zeros_like(self.a), self.a, self.b)
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - (self._mean ** 2)
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z
    @constraints.dependent_property
    def support(self): return constraints.interval(self.a, self.b)
    @property
    def mean(self): return self._mean
    @property
    def mode(self): return self._mode
    @property
    def variance(self): return self._variance
    def entropy(self): return self._entropy
    @staticmethod
    def _little_phi(x): return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI
    @staticmethod
    def _big_phi(x): return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())
    @staticmethod
    def _inv_big_phi(x): return CONST_SQRT_2 * (2 * x - 1).erfinv()
    def cdf(self, value): return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)
    def icdf(self, value): return self._inv_big_phi(self._big_phi_a + value * self._Z)
    def log_prob(self, value):
        if self._validate_args: self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

class TruncatedNormal(TruncatedStandardNormal):
    has_rsample = True
    def __init__(self, loc, scale, scalar_a, scalar_b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, scalar_a, scalar_b)
        a, b = (a - self.loc) / self.scale, (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._mode = torch.clamp(self.loc, scalar_a, scalar_b)
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale
    def _to_std_rv(self, value): return (value - self.loc) / self.scale
    def _from_std_rv(self, value): return value * self.scale + self.loc
    def cdf(self, value): return super(TruncatedNormal, self).cdf(self._to_std_rv(value))
    def icdf(self, value): return self._from_std_rv(super(TruncatedNormal, self).icdf(value))
    def log_prob(self, value): return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale

class TruncNormalDist(TruncatedNormal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip, self._mult, self.low, self.high = clip, mult, low, high
    def sample(self, *args, **kwargs):
        event = super().rsample(*args, **kwargs)
        if self._clip: event = torch.clamp(event, self.low + self._clip, self.high - self._clip) - event.detach() + event
        if self._mult: event *= self._mult
        return event

class RSSM(nn.Module):
    def __init__(self, mlp_hidden_dim, rnn_hidden_dim, state_dim, num_classes, action_dim):
        super().__init__()
        self.rnn_hidden_dim, self.state_dim, self.num_classes = rnn_hidden_dim, state_dim, num_classes
        self.transition_hidden = nn.Linear(state_dim * num_classes + action_dim, mlp_hidden_dim)
        self.transition = nn.GRUCell(mlp_hidden_dim, rnn_hidden_dim)
        self.prior_hidden = nn.Linear(rnn_hidden_dim, mlp_hidden_dim)
        self.prior_logits = nn.Linear(mlp_hidden_dim, state_dim * num_classes)
        self.posterior_hidden = nn.Linear(rnn_hidden_dim + 1536, mlp_hidden_dim)
        self.posterior_logits = nn.Linear(mlp_hidden_dim, state_dim * num_classes)
    def recurrent(self, state, action, rnn_hidden):
        hidden = F.elu(self.transition_hidden(torch.cat([state, action], dim=1)))
        return self.transition(hidden, rnn_hidden)
    def get_prior(self, rnn_hidden, detach=False):
        hidden = F.elu(self.prior_hidden(rnn_hidden))
        logits = self.prior_logits(hidden)
        logits = logits.reshape(logits.shape[0], self.state_dim, self.num_classes)
        dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        return (dist, td.Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)) if detach else dist
    def get_posterior(self, rnn_hidden, embedded_obs, detach=False):
        hidden = F.elu(self.posterior_hidden(torch.cat([rnn_hidden, embedded_obs], dim=1)))
        logits = self.posterior_logits(hidden).reshape(hidden.shape[0], self.state_dim, self.num_classes)
        dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        return (dist, td.Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)) if detach else dist

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2, self.conv3, self.conv4 = nn.Conv2d(3, 48, 4, 2), nn.Conv2d(48, 96, 4, 2), nn.Conv2d(96, 192, 4, 2), nn.Conv2d(192, 384, 4, 2)
    def forward(self, obs):
        hidden = F.elu(self.conv1(obs))
        hidden = F.elu(self.conv2(hidden))
        hidden = F.elu(self.conv3(hidden))
        return self.conv4(hidden).reshape(hidden.size(0), -1)

class Decoder(nn.Module):
    def __init__(self, rnn_hidden_dim, state_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(state_dim*num_classes + rnn_hidden_dim, 1536)
        self.dc1, self.dc2, self.dc3, self.dc4 = nn.ConvTranspose2d(1536, 192, 5, 2), nn.ConvTranspose2d(192, 96, 5, 2), nn.ConvTranspose2d(96, 48, 6, 2), nn.ConvTranspose2d(48, 3, 6, 2)
    def forward(self, state, rnn_hidden):
        hidden = self.fc(torch.cat([state, rnn_hidden], dim=1)).view(-1, 1536, 1, 1)
        hidden = F.elu(self.dc1(hidden))
        hidden = F.elu(self.dc2(hidden))
        hidden = F.elu(self.dc3(hidden))
        return td.Independent(MSE(self.dc4(hidden)), 3)

class RewardModel(nn.Module):
    def __init__(self, hidden_dim, rnn_hidden_dim, state_dim, num_classes):
        super().__init__()
        self.fc1, self.fc2, self.fc3, self.fc4 = nn.Linear(state_dim*num_classes+rnn_hidden_dim, hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,1)
    def forward(self, state, rnn_hidden):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        return td.Independent(MSE(self.fc4(hidden)), 1)

class DiscountModel(nn.Module):
    def __init__(self, hidden_dim, rnn_hidden_dim, state_dim, num_classes):
        super().__init__()
        self.fc1, self.fc2, self.fc3, self.fc4 = nn.Linear(state_dim*num_classes+rnn_hidden_dim, hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,1)
    def forward(self, state, rnn_hidden):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        return td.Independent(td.Bernoulli(logits=self.fc4(hidden)), 1)

class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim, rnn_hidden_dim, state_dim, num_classes):
        super().__init__()
        self.fc1, self.fc2, self.fc3, self.fc4 = nn.Linear(state_dim*num_classes+rnn_hidden_dim,hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,hidden_dim)
        self.mean, self.std, self.min_stddev, self.init_stddev = nn.Linear(hidden_dim,action_dim), nn.Linear(hidden_dim,action_dim), 0.1, np.log(np.exp(5.0)-1)
    def forward(self, state, rnn_hidden, eval=False):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        mean, stddev = self.mean(hidden), self.std(hidden)
        mean = torch.tanh(mean)
        stddev = 2 * torch.sigmoid((stddev + self.init_stddev) / 2) + self.min_stddev
        if eval: return mean, None, None
        action_dist = td.Independent(TruncNormalDist(mean, stddev, -1, 1), 1)
        action = action_dist.sample()
        return action, action_dist.log_prob(action), action_dist.entropy()

class DiscreteActor(nn.Module):
    def __init__(self, action_dim, hidden_dim, rnn_hidden_dim, state_dim, num_classes):
        super().__init__()
        self.fc1, self.fc2, self.fc3, self.fc4, self.out = nn.Linear(state_dim*num_classes+rnn_hidden_dim,hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,action_dim)
    def forward(self, state, rnn_hidden, eval=False):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        logits = self.out(hidden)
        if eval: return torch.argmax(logits, dim=-1), None, None
        action_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        action = action_dist.sample()
        return action, action_dist.log_prob(action), action_dist.entropy()

class Critic(nn.Module):
    def __init__(self, hidden_dim, rnn_hidden_dim, state_dim, num_classes):
        super().__init__()
        self.fc1, self.fc2, self.fc3, self.fc4, self.out = nn.Linear(state_dim*num_classes+rnn_hidden_dim,hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,hidden_dim), nn.Linear(hidden_dim,1)
    def forward(self, state, rnn_hidden):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        return self.out(hidden)

class Agent(nn.Module):
    def __init__(self, encoder, decoder, rssm, action_model):
        super().__init__()
        self.encoder, self.decoder, self.rssm, self.action_model = encoder, decoder, rssm, action_model
        self.device = next(self.action_model.parameters()).device
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)
        self.last_state, self.last_rnn_hidden = None, None
    def __call__(self, obs, eval=True):
        obs = preprocess_obs(obs)
        obs = torch.as_tensor(obs, device=self.device).transpose(1, 2).transpose(0, 1).unsqueeze(0)
        with torch.no_grad():
            state_prior = self.rssm.get_prior(self.rnn_hidden)
            state_pred = state_prior.sample().flatten(1)
            obs_pred_img = self.decoder(state_pred, self.rnn_hidden).mean
            embedded_obs = self.encoder(obs)
            state_posterior = self.rssm.get_posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample().flatten(1)
            action, _, _ = self.action_model(state, self.rnn_hidden, eval=eval)
            self.last_state, self.last_rnn_hidden = state, self.rnn_hidden
            if state.ndim == 1: state = state.unsqueeze(0)
            if action.ndim == 1: action = action.unsqueeze(0)
            if action.shape[-1] == 1 and self.rssm.transition_hidden.in_features > (state.shape[-1] + 1):
                 num_classes, action_idx = self.rssm.transition_hidden.in_features - state.shape[-1], action.long()
                 action = F.one_hot(action_idx, num_classes=num_classes).float().squeeze(1)
            self.rnn_hidden = self.rssm.recurrent(state, action, self.rnn_hidden)
        return action.squeeze().cpu().numpy(), (obs_pred_img.squeeze().cpu().numpy().transpose(1, 2, 0) + 0.5).clip(0.0, 1.0)
    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)
        self.last_state, self.last_rnn_hidden = None, None
    def to(self, device):
        self.device, self.rnn_hidden = device, self.rnn_hidden.to(device)
        self.encoder.to(device); self.decoder.to(device); self.rssm.to(device); self.action_model.to(device)
        return self

def preprocess_obs(obs):
    return (obs.astype(np.float32) / 255.0) - 0.5