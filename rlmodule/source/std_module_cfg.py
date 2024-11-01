from dataclasses import MISSING
from typing import Sequence, Union, Optional, List
import torch.nn as nn

from rlmodule.source.std_module import StdModule, ParameterStdModule, NNStdModule, CombinedStdModule
from rlmodule.source.network_cfg import NetworkCfg, MlpCfg

# use isaac-lab native configclass if available to avoid double declaration
try:
    from omni.isaac.lab.utils import configclass
except ImportError:
    from rlmodule.source.nvidia_utils import configclass

@configclass
class StdModuleCfg:

    class_type: type[StdModule] = StdModule

@configclass
class ParameterStdModuleCfg:

    class_type: type[StdModule] = ParameterStdModule
    
    initial_log_std: float = 0.0
    """Initial value for the log standard deviation"""


@configclass
class NNStdModuleCfg:

    class_type: type[StdModule] = NNStdModule
    
    network_cfg: Optional[MlpCfg] = None
    # MlpCfg(
        # hidden_units=[512],
        # activation=nn.ReLU,
    #)
    """
    Config for the hidden part of the network.

    Input and output sizes are computed on runtime.
    Output layer is automatically connected with linear layer with
    identity activation function. 
    
    If network_cfg is None, then input is connected to output with linear
    layer.
    """

@configclass
class CombinedStdModuleCfg:

    class_type: type[StdModule] = CombinedStdModule
    
    combination_method: str = MISSING
    """
    Method used to combine the standard deviations from multiple modules.
    
    Currently supported modes:
    - max
    - mean
    - min
    """

    combined_modules: List[StdModuleCfg] = MISSING
    """List of modules to be combined."""

    combination_constants: Optional[List[float]] = None
    """List of constants used to multiple Std from each module before combination."""