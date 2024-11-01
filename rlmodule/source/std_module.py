from abc import ABC, abstractmethod
from typing import Union
import torch
import torch.nn as nn
from dataclasses import MISSING
from itertools import repeat

class StdModule(ABC):
    """Base class that defines interface for standard deviation computation in GaussianLayer."""

    @abstractmethod
    def __init__(self, device: Union[str, torch.device], input_size: int, output_size: int):
        """Initialize Standard deviation computation module."""
        self.device = device

        self._input_size = input_size
        self._output_size = output_size

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation from layer input."""
        raise NotImplementedError(f'Forward method is not implemented for {type(self)}')

class ParameterStdModule(StdModule):
    """Class that computes standard deviation in GaussianLayer from nn.Parameter."""

    def __init__(self, device: Union[str, torch.device], input_size: int, output_size: int, cfg):
        """Initialize Standard deviation computation module."""
        super().__init__(device, input_size, output_size)

        self._log_parameter = nn.Parameter(
            cfg.initial_log_std * torch.ones(self._output_size, device=device)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation from layer input."""
        return self._log_parameter


class NNStdModule(StdModule):
    """Class that computes standard deviation in GaussianLayer from user-provided network.
    
    Note:
        Currently support only MLP.
    """

    def __init__(self, device: Union[str, torch.device], input_size: int, output_size: int, cfg):
        """Initialize Standard deviation computation module."""
        super().__init__(device, input_size, output_size)

        network_cfg = cfg.network_cfg
        print(network_cfg)
        # If user did not provide any hidden layers
        if network_cfg is None:
            self._net = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Identity(),
            )
        else:
            network_cfg.input_size = input_size
            hidden_net =  network_cfg.module(network_cfg).to(device)

            layers = [
                hidden_net,
                nn.Linear(network_cfg.hidden_units[-1], output_size),
                nn.Identity(),
            ]
            self._net = nn.Sequential(*layers)

        self._net.to(device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation from layer input."""
        return self._net(input)
    

class NNStdModule(StdModule):
    """Class that computes standard deviation in GaussianLayer from user-provided neural network.
    
    Note:
        Currently support only MLP.
    """

    def __init__(self, device: Union[str, torch.device], input_size: int, output_size: int, cfg):
        """Initialize Standard deviation computation module."""
        super().__init__(device, input_size, output_size)

        network_cfg = cfg.network_cfg
        print(network_cfg)
        # If user did not provide any hidden layers
        if network_cfg is None:
            self._net = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Identity(),
            )
        else:
            network_cfg.input_size = input_size
            hidden_net =  network_cfg.module(network_cfg).to(device)

            layers = [
                hidden_net,
                nn.Linear(network_cfg.hidden_units[-1], output_size),
                nn.Identity(),
            ]
            self._net = nn.Sequential(*layers)

        self._net.to(device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation from layer input."""
        return self._net(input)
    

class CombinedStdModule(StdModule):
    """Std module that combines functionality of multiple std modules with predefined function."""

    def __init__(self, device: Union[str, torch.device], input_size: int, output_size: int, cfg):
        super().__init__(device, input_size, output_size)

        self._modules = [
            module_cfg.class_type(device, input_size, output_size, module_cfg)
            for module_cfg in cfg.combined_modules
        ]

        if cfg.combination_constants is None:
            self._combination_constants = list(repeat(1.0, len(self._modules)))
        else:
            self._combination_constants = cfg.combination_constants
            if len(self._modules) != len(self._combination_constants):
                raise ValueError(f'Number of combination constants {len(self._combination_constants)} does not match number of modules {len(self._modules)}')
        
        
        if cfg.combination_method == 'max':
            self._combination_method = torch.max
        elif cfg.combination_method == 'min':
            self._combination_method = torch.min
        elif cfg.combination_method == 'mean':
            self._combination_method = torch.mean
        else:
            raise ValueError(f'Combination method `{cfg.combination_method}` for standard deviation combination module unknown.')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        collected_stds = [
            self._combination_constants[module_id] * module.forward(input) 
            for module_id, module in enumerate(self._modules)
        ]
        return self._combination_method(*collected_stds)
