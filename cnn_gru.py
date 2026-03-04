from __future__ import annotations

import torch as th
import torch.nn as nn

from gymnasium import spaces
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CnnStateExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observations:
      - "image": processed by a CNN (Nature-DQN architecture)
      - "state": game variables vector, concatenated after CNN features

    Final output = Linear(cnn_flat + state_dim) -> ReLU -> features_dim
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Must call super with the actual features_dim we'll output
        super().__init__(observation_space, features_dim)

        image_space = observation_space.spaces["image"]
        state_space = observation_space.spaces["state"]

        n_input_channels = int(image_space.shape[0])
        state_dim = int(state_space.shape[0])

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size
        with th.no_grad():
            sample = th.as_tensor(image_space.sample()[None]).float() / 255.0
            n_cnn_flat = self.cnn(sample).shape[1]

        # State branch: small MLP to give state features some nonlinearity
        state_hidden = max(state_dim * 2, 16)
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, state_hidden),
            nn.ReLU(),
        )

        # Combined projection
        self.linear = nn.Sequential(
            nn.Linear(n_cnn_flat + state_hidden, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> th.Tensor:
        img = observations["image"].float() / 255.0
        state = observations["state"].float()

        # --- V4 Debug: Verify state vector is alive ---
        if not hasattr(self, "_debug_printed"):
            self._debug_printed = 0
        if self._debug_printed < 5:
            # Print a few times at the start of training/eval to confirm
            print(f"[DEBUG V4] state vector shape: {state.shape}, min: {state.min().item():.3f}, max: {state.max().item():.3f}, mean: {state.mean().item():.3f}")
            self._debug_printed += 1
        # ----------------------------------------------

        cnn_features = self.cnn(img)
        state_features = self.state_net(state)

        combined = th.cat([cnn_features, state_features], dim=-1)
        return self.linear(combined)


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
