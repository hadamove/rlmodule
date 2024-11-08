from typing import Sequence, Tuple, Union

from collections.abc import Callable
from dataclasses import MISSING
import gym
import gymnasium

import torch.nn as nn

from rlmodule import logger
from rlmodule.source.network import CNN, GRU, LSTM, MLP, RNN, RnnBase, RnnMlp


# use isaac-lab native configclass if available to avoid it being declared twice
try:
    from omni.isaac.lab.utils import configclass
except ImportError:
    logger.info("Importing local configclass.")
    from rlmodule.source.nvidia_utils import configclass


@configclass
class NetworkCfg:
    module: Union[nn.Module, Callable[..., nn.Module]] = MISSING
    input_states: Union[int, Sequence[int], gym.Space, gymnasium.Space] = None
    """Observations passed to the network as an input."""
    input_actions: Union[int, Sequence[int], gym.Space, gymnasium.Space] = None
    """Action passed to the network as an input. If it is None, actions will not be passed."""


@configclass
class MlpCfg(NetworkCfg):
    module: type[MLP] = MLP

    hidden_units: Sequence[int] = MISSING
    activation: type[nn.Module] = MISSING


@configclass
class RnnBaseCfg(NetworkCfg):
    num_envs: int = MISSING
    num_layers: int = MISSING
    hidden_size: int = MISSING
    sequence_length: int = MISSING


@configclass
class RnnCfg(RnnBaseCfg):
    module: type[RNN] = RNN


@configclass
class GruCfg(RnnBaseCfg):
    module: type[GRU] = GRU


@configclass
class LstmCfg(RnnBaseCfg):
    module: type[LSTM] = LSTM


@configclass
class RnnMlpCfg(NetworkCfg):
    module: type[RnnBase] = RnnMlp

    rnn: RnnBaseCfg = MISSING
    mlp: MlpCfg = MISSING


@configclass
class CnnConvLayerCfg:
    in_channels: int = MISSING
    """Number of input channels."""

    out_channels: int = MISSING
    """Number of output channels."""

    kernel_size: Union[int, Tuple[int, int]] = MISSING
    """Size of the kernel."""

    stride: int = MISSING
    """Stride of the convolution or pooling operation."""

    activation: type[nn.Module] = MISSING
    """Activation function to use after the layer."""


@configclass
class CnnPoolLayerCfg:
    kernel_size: Union[int, Tuple[int, int]] = MISSING
    """Size of the kernel."""

    stride: int = MISSING
    """Stride of the convolution or pooling operation."""


@configclass
class CnnDenseLayerCfg:
    in_features: int = MISSING
    """Number of input features."""

    out_features: int = MISSING
    """Number of output features."""

    activation: type[nn.Module] = MISSING
    """Activation function to use after the layer."""


@configclass
class CnnCfg(NetworkCfg):
    module: type[CNN] = CNN

    layers: Sequence[nn.Module] = MISSING
    """Layers of convolutional neural network."""

    # activations: Sequence[type[nn.Module]]
    # """Activations to be applied after each layer."""
