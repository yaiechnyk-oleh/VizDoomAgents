from __future__ import annotations

import torch as th
import torch.nn as nn

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    CNN extractor for images in CHW format.
    Supports any channel count (e.g., stack=2 grayscale => C=2).
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = int(observation_space.shape[0])

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float() / 255.0
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.float() / 255.0
        return self.linear(self.cnn(x))


class LSTMCompatibleGRU(nn.Module):
    """
    GRU with LSTM-compatible API (h,c).
    sb3_contrib expects policy.lstm-like attributes.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: th.Tensor, states: tuple[th.Tensor, th.Tensor]):
        h, c = states
        out, h_new = self.gru(x, h)
        return out, (h_new, c)

    def flatten_parameters(self):
        return self.gru.flatten_parameters()


class CnnGruPolicy(RecurrentActorCriticPolicy):
    """
    RecurrentActorCriticPolicy where internal LSTM is replaced with GRU,
    but we keep SB3's expected (h,c) interface.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if hasattr(self, "lstm") and self.lstm is not None:
            input_size = self.lstm.input_size
            hidden_size = self.lstm.hidden_size
            num_layers = self.lstm.num_layers
            self.lstm = LSTMCompatibleGRU(input_size, hidden_size, num_layers).to(self.device)

        if hasattr(self, "lstm_actor") and self.lstm_actor is not None:
            input_size = self.lstm_actor.input_size
            hidden_size = self.lstm_actor.hidden_size
            num_layers = self.lstm_actor.num_layers
            self.lstm_actor = LSTMCompatibleGRU(input_size, hidden_size, num_layers).to(self.device)

        if hasattr(self, "lstm_critic") and self.lstm_critic is not None:
            input_size = self.lstm_critic.input_size
            hidden_size = self.lstm_critic.hidden_size
            num_layers = self.lstm_critic.num_layers
            self.lstm_critic = LSTMCompatibleGRU(input_size, hidden_size, num_layers).to(self.device)
