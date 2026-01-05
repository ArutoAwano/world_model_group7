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
        # NOTE: dropped the constant term
        return -((value - self.loc) ** 2) / 2

# From https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py
import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1, ).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b -
                                 self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        # NOTE: additional to github.com/toshas/torch_truncnorm
        self._mode = torch.clamp(torch.zeros_like(self.a), self.a, self.b)
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def mode(self):
        return self._mode

    @property
    def variance(self):
        return self._variance

    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        # icdf is numerically unstable; as a consequence, so is rsample.
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, scalar_a, scalar_b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, scalar_a, scalar_b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._mode = torch.clamp(self.loc, scalar_a, scalar_b)  # NOTE: additional to github.com/toshas/torch_truncnorm
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale


class TruncNormalDist(TruncatedNormal):

    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip = clip
        self._mult = mult

        self.low = low
        self.high = high

    def sample(self, *args, **kwargs):
        event = super().rsample(*args, **kwargs)
        if self._clip:
            clipped = torch.clamp(
                event, self.low + self._clip, self.high - self._clip
            )
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class RSSM(nn.Module):
    """
    Recurrent State Space Model (世界モデルの中核)
    
    【役割】
    エージェントの「脳」にあたる部分です。
    過去の記憶(RNN)と、現在の状況(State)を統合し、未来をシミュレーションします。
    
    【構成要素】
    1. Transition Model (決定論的): RNN。過去の文脈を隠れ状態 h_t として保持します。
    2. Prior Network (確率的): h_t から「次の状態 z_t」を予測します（目をつぶって想像）。
    3. Posterior Network (確率的): h_t と「実際の観測画像 o_t」から「真の状態 z_t」を推論します（現実を見て修正）。
    """
    def __init__(self, mlp_hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int, action_dim: int):
        super().__init__()

        self.rnn_hidden_dim = rnn_hidden_dim
        self.state_dim = state_dim
        self.num_classes = num_classes

        # 1. Recurrent model (Transition Model)
        # 役割: 時間的な記憶の更新
        # 数式: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        # 入力: [前の記憶] + [前の状態] + [前の行動]
        self.transition_hidden = nn.Linear(state_dim * num_classes + action_dim, mlp_hidden_dim)
        self.transition = nn.GRUCell(mlp_hidden_dim, rnn_hidden_dim)

        # 2. Transition predictor (Prior Network)
        # 役割: 未来予測 (Imagination)
        # 数式: \hat{z}_t ~ p(z_t | h_t)
        # 記憶(h_t)だけから、今の状態(z_t)はどうなっているはずかを予測します。
        self.prior_hidden = nn.Linear(rnn_hidden_dim, mlp_hidden_dim)
        self.prior_logits = nn.Linear(mlp_hidden_dim, state_dim * num_classes)

        # 3. Representation model (Posterior Network)
        # 役割: 現実認識 (Observation)
        # 数式: z_t ~ q(z_t | h_t, o_t)
        # 記憶(h_t)に加えて、実際の画像特徴量(embedded_obs)を見て、状態認識を修正します。
        # これが「正解データ」としてPriorの学習に使われます (KL Divergence)。
        self.posterior_hidden = nn.Linear(rnn_hidden_dim + 1536, mlp_hidden_dim)
        self.posterior_logits = nn.Linear(mlp_hidden_dim, state_dim * num_classes)

    def recurrent(self, state: torch.Tensor, action: torch.Tensor, rnn_hidden: torch.Tensor):
        """
        Recurrent Step: h_t = f(h_t-1, z_t-1, a_t-1)
        """
        hidden = F.elu(self.transition_hidden(torch.cat([state, action], dim=1)))
        rnn_hidden = self.transition(hidden, rnn_hidden)

        return rnn_hidden  # h_t

    def get_prior(self, rnn_hidden: torch.Tensor, detach=False):
        # transition predictor: \hat{z}_t ~ p(z\hat{z}_t | h_t)
        hidden = F.elu(self.prior_hidden(rnn_hidden))
        logits = self.prior_logits(hidden)
        logits = logits.reshape(logits.shape[0], self.state_dim, self.num_classes)

        prior_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        if detach:
            detach_prior = td.Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)
            return prior_dist, detach_prior  # p(z\hat{z}_t | h_t)
        return prior_dist

    def get_posterior(self, rnn_hidden: torch.Tensor, embedded_obs: torch.Tensor, detach=False):
        # representation predictor: z_t ~ q(z_t | h_t, o_t)
        hidden = F.elu(self.posterior_hidden(torch.cat([rnn_hidden, embedded_obs], dim=1)))
        logits = self.posterior_logits(hidden)
        logits = logits.reshape(logits.shape[0], self.state_dim, self.num_classes)

        posterior_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        if detach:
            detach_posterior = td.Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)
            return posterior_dist, detach_posterior  # q(z_t | h_t, o_t)
        return posterior_dist


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 48, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(192, 384, kernel_size=4, stride=2)

    def forward(self, obs: torch.Tensor):
        """
        観測画像をベクトルに埋め込むためのEncoder．

        Parameters
        ----------
        obs : torch.Tensor (B, C, H, W)
            入力となる観測画像．

        Returns
        -------
        embedded_obs : torch.Tensor (B, D)
            観測画像をベクトルに変換したもの．Dは入力画像の幅と高さに依存して変わる．
            入力が(B, 3, 64, 64)の場合，出力は(B, 1536)になる．
        """
        hidden = F.elu(self.conv1(obs))
        hidden = F.elu(self.conv2(hidden))
        hidden = F.elu(self.conv3(hidden))
        embedded_obs = self.conv4(hidden).reshape(hidden.size(0), -1)

        return embedded_obs  # x_t


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 48, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(192, 384, kernel_size=4, stride=2)

    def forward(self, obs: torch.Tensor):
        """
        観測画像をベクトルに埋め込むためのEncoder．

        Parameters
        ----------
        obs : torch.Tensor (B, C, H, W)
            入力となる観測画像．

        Returns
        -------
        embedded_obs : torch.Tensor (B, D)
            観測画像をベクトルに変換したもの．Dは入力画像の幅と高さに依存して変わる．
            入力が(B, 3, 64, 64)の場合，出力は(B, 1536)になる．
        """
        hidden = F.elu(self.conv1(obs))
        hidden = F.elu(self.conv2(hidden))
        hidden = F.elu(self.conv3(hidden))
        embedded_obs = self.conv4(hidden).reshape(hidden.size(0), -1)

        return embedded_obs  # x_t


class Decoder(nn.Module):
    def __init__(self, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(state_dim*num_classes + rnn_hidden_dim, 1536)
        self.dc1 = nn.ConvTranspose2d(1536, 192, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(192, 96, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(96, 48, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(48, 3, kernel_size=6, stride=2)


    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor):
        """
        決定論的状態と，確率的状態を入力として，観測画像を復元するDecoder．
        出力は多次元正規分布の平均値をとる．

        Paremters
        ---------
        state : torch.Tensor (B, state_dim * num_classes)
            確率的状態．
        rnn_hidden : torch.Tensor (B, rnn_hidden_dim)
            決定論的状態．

        Returns
        -------
        obs_dist : torch.distribution.Independent
            観測画像を再構成するための多次元正規分布．
        """
        hidden = self.fc(torch.cat([state, rnn_hidden], dim=1))
        hidden = hidden.view(hidden.size(0), 1536, 1, 1)
        hidden = F.elu(self.dc1(hidden))
        hidden = F.elu(self.dc2(hidden))
        hidden = F.elu(self.dc3(hidden))
        mean = self.dc4(hidden)

        obs_dist = td.Independent(MSE(mean), 3)
        return obs_dist  # p(\hat{x}_t | h_t, z_t)


class RewardModel(nn.Module):
    def __init__(self, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim*num_classes + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor):
        """
        決定論的状態と，確率的状態を入力として，報酬を予測するモデル．
        出力は正規分布の平均値をとる．

        Paremters
        ---------
        state : torch.Tensor (B, state_dim * num_classes)
            確率的状態．
        rnn_hidden : torch.Tensor (B, rnn_hidden_dim)
            決定論的状態．

        Returns
        -------
        reward_dist : torch.distribution.Independent
            報酬を予測するための正規分布．
        """
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        mean = self.fc4(hidden)

        reward_dist = td.Independent(MSE(mean),  1)
        return reward_dist  # p(\hat{r}_t | h_t, z_t)


class DiscountModel(nn.Module):
    def __init__(self, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim*num_classes + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor):
        """
        決定論的状態と，確率的状態を入力として，現在の状態がエピソード終端かどうか判別するモデル．
        出力はベルヌーイ分布の平均値をとる．

        Paremters
        ---------
        state : torch.Tensor (B, state_dim * num_classes)
            確率的状態．
        rnn_hidden : torch.Tensor (B, rnn_hidden_dim)
            決定論的状態．

        Returns
        -------
        discount_dist : torch.distribution.Independent
            状態が終端かどうかを予測するためのベルヌーイ分布．
        """
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        mean= self.fc4(hidden)

        discount_dist = td.Independent(td.Bernoulli(logits=mean),  1)
        return discount_dist  # p(\hat{\gamma}_t | h_t, z_t)


class Actor(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()

        self.fc1 = nn.Linear(state_dim * num_classes + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)
        self.min_stddev = 0.1
        self.init_stddev = np.log(np.exp(5.0) - 1)

    def forward(self, state: torch.tensor, rnn_hidden: torch.Tensor, eval: bool = False):
        """
        確率的状態を入力として，criticで推定される価値が最大となる行動を出力する．

        Parameters
        ----------
        state : torch.Tensor (B, state_dim * num_classes)
            確率的状態．
        rnn_hidden : torch.Tensor (B, rnn_hidden_dim)
            決定論的状態．

        Returns
        -------
        action : torch.Tensor (B, 1)
            行動．
        action_log_prob : torch.Tensor(B, 1)
            予測した行動をとる確率の対数．
        action_entropy : torch.Tensor(B, 1)
            予測した確率分布のエントロピー．エントロピー正則化に使用．
        """
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        mean = self.mean(hidden)
        stddev = self.std(hidden)

        mean = torch.tanh(mean)
        stddev = 2 * torch.sigmoid((stddev + self.init_stddev) / 2) + self.min_stddev
        if eval:
            action = mean
            return action, None, None

        action_dist = td.Independent(TruncNormalDist(mean, stddev, -1, 1), 1)  # 行動をサンプリングする分布: p_{\psi} (\hat{a}_t | \hat{z}_t)
        action = action_dist.sample()  # 行動: \hat{a}_t

        action_log_prob = action_dist.log_prob(action)
        action_entropy = action_dist.entropy()

        return action, action_log_prob, action_entropy


class DiscreteActor(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim * num_classes + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.tensor, rnn_hidden: torch.Tensor, eval: bool = False):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        logits = self.out(hidden)

        if eval:
            # 評価時は決定論的(argmax)な行動をとる
            action = torch.argmax(logits, dim=-1)
            # action is (B,). DiscreteActor usually returns index for action?
            # But RSSM expects OneHot or Something?
            # RSSM: transition_hidden(state_dim*num + action_dim)
            # It expects vector.
            # If action is index, we need OneHot.
            # But here we return index?
            # Wait, let's check DiscreteActor convention.
            # If RSSM expects (B, A), then we should return OneHot.
            # But student_code implementation of DiscreteActor returns index in eval?
            # Let's check non-eval path.
            return action, None, None

        # 探索時は確率的(Categorical分布)に行動選ぶ
        action_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        action = action_dist.sample()
        
        # log_prob of OneHotCategorical is (B,)
        action_log_prob = action_dist.log_prob(action)
        action_entropy = action_dist.entropy()
        
        return action, action_log_prob, action_entropy


class Critic(nn.Module):
    def __init__(self, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()

        self.fc1 = nn.Linear(state_dim * num_classes + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.tensor, rnn_hidden: torch.Tensor):
        """
        確率的状態を入力として，価値関数(lambda target)の値を予測する．．

        Parameters
        ----------
        state : torch.Tensor (B, state_dim * num_classes)
            確率的状態．
        rnn_hidden : torch.Tensor (B, rnn_hidden_dim)
            決定論的状態．

        Returns
        -------
        value : torch.Tensor (B, 1)
            入力された状態に対する状態価値関数の予測値．
        """
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        mean = self.out(hidden)

        return mean


class ErrorPredictor(nn.Module):
    """
    LPM (Learning Progress Motivation) のための予測誤差予測モデル
    
    【役割】
    エージェントが「自分の予測能力のなさ（＝難しさ）」をメタ認知するためのモデルです。
    「今の状態」と「次の行動」から、「次の瞬間の世界モデルの予測誤差」を予測します。
    
    【入力】
    - state (z_t)      : 確率的状態 (Latent State)。画像の抽象表現。
    - rnn_hidden (h_t) : 決定論的状態 (Recurrent State)。時間的な文脈情報。
    - action (a_t)     : これから取る行動。
    
    【出力】
    - pred_error       : スカラー値。予測されるReconstruction Lossの大きさ。
                         この値が大きい＝「この行動をとると未来が予測しにくい（未知/学習不足）」と判断されます。
    """
    def __init__(self, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int, action_dim: int):
        super().__init__()
        # 入力: [確率的状態 z] + [RNN隠れ状態 h] + [行動 a]
        # これらを結合して、未来の「難しさ」を推論します。
        self.fc1 = nn.Linear(state_dim * num_classes + rnn_hidden_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1) # 出力は単一の数値(予測誤差)

    def forward(self, state: torch.tensor, rnn_hidden: torch.Tensor, action: torch.Tensor):
        # 1. 入力データを結合(concatenation)
        x = torch.cat([state, rnn_hidden, action], dim=1)
        
        # 2. MLP(多層パーセプトロン)で計算
        hidden = F.elu(self.fc1(x))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        
        # 3. 予測誤差(スカラー)を出力
        pred_error = self.fc4(hidden)
        return pred_error



class Agent(nn.Module):
    """
    ActionModelに基づき行動を決定する. そのためにRSSMを用いて状態表現をリアルタイムで推論して維持するクラス
    """
    def __init__(self, encoder, decoder, rssm, action_model, error_predictor=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rssm = rssm
        self.action_model = action_model
        self.error_predictor = error_predictor

        self.device = next(self.action_model.parameters()).device
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs, eval=True):
        # 1. 前処理 & テンソル変換
        # 画像を [0, 255] -> [-0.5, 0.5] に正規化し、PyTorch形式 (B, C, H, W) に合わせます。
        obs = preprocess_obs(obs)
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0) # (1, C, H, W)

        with torch.no_grad():
            # 2. Imagination (世界モデルによる未来予測 - 可視化用)
            # 現在の記憶(RNN state)だけから「次の画像」を想像してみます。
            # これは行動決定には使いませんが、デバッグ画面で「AIが見ている夢」を表示するために返します。
            state_prior = self.rssm.get_prior(self.rnn_hidden)
            state_pred = state_prior.sample().flatten(1)
            obs_dist = self.decoder(state_pred, self.rnn_hidden)
            obs_pred_img = obs_dist.sample()

            # 3. 状態認識 (State Estimation) - 「現実を見る」
            # 実際の観測画像(obs)を使って、世界モデルの認識(Posterior)を更新します。
            # 「目を閉じて予想していた状態」を「目で見た現実」で修正するプロセスです。
            embedded_obs = self.encoder(obs)
            state_posterior = self.rssm.get_posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample().flatten(1)
            
            # 4. 行動決定 (Action Selection)
            # 「現在の正確な状態(z_t)」と「記憶(h_t)」を元に、Actorが次の行動を決めます。
            action, _, _  = self.action_model(state, self.rnn_hidden, eval=eval)

            # 5. 記憶の更新 (Recurrent Step)
            # 次のステップのために、現在の状態を保存します。
            self.last_state = state
            self.last_rnn_hidden = self.rnn_hidden
            if state.ndim == 1:
                state = state.unsqueeze(0)
        
            # Ensure action has batch dim
            if action.ndim == 1:
                action = action.unsqueeze(0)
            
            # If using DiscreteActor, RSSM expects One-Hot, but action_model might return scalar/logits?
            # DiscreteActor returns One-Hot if we look at student_code.DiscreteActor...
            # Wait, DiscreteActor in student_code returns Independent(OneHotCategorical(...)).sample().
            # So it returns One-Hot vector (B, num_classes).
            # BUT if action_dim mismatch occurs, it means action is scalar?
            # Ah, maybe I was fixing phantom bug. The error "1025" (1024+1) suggests action IS size 1.
            # So DiscreteActor.forward MUST return size 1?
            # OneHotCategorical sample returns one-hot vector...
            
            # Let's check if action needs to be one-hot encoded manually if it's not.
            # RSSM expects input size 1042 (1024 + 18).
            # If action is size 1, we must one-hot encode it.
            if action.shape[-1] == 1 and self.rssm.transition_hidden.in_features > (state.shape[-1] + 1):
                 # Assume 18 classes
                 num_classes = self.rssm.transition_hidden.in_features - state.shape[-1]
                 # Action is likely index (float or int)
                 action_idx = action.long()
                 action = F.one_hot(action_idx, num_classes=num_classes).float().squeeze(1)
            
            self.rnn_hidden = self.rssm.recurrent(state, action, self.rnn_hidden)

        return action.squeeze().cpu().numpy(), (obs_pred_img.squeeze().cpu().numpy().transpose(1, 2, 0) + 0.5).clip(0.0, 1.0)

    #RNNの隠れ状態をリセット
    def reset(self):
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)
        self.last_state = None
        self.last_rnn_hidden = None

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        self.rssm.to(device)
        if self.error_predictor:
            self.error_predictor.to(device)
        self.rnn_hidden = self.rnn_hidden.to(device)
        return self


def preprocess_obs(obs):
    """
    画像の変換. [0, 255] -> [-0.5, 0.5]
    """
    obs = obs.astype(np.float32)
    normalized_obs = obs / 255.0 - 0.5
    return normalized_obs
