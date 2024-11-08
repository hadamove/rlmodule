import os
from datetime import datetime
import gymnasium as gym

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG, SAC_RNN
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import torch.nn as nn

from rlmodule.skrl.torch import RLModelCfg, build_model
from rlmodule.skrl.torch.network import MlpCfg
from rlmodule.skrl.torch.output_layer import DeterministicLayerCfg, GaussianLayerCfg


def get_model(env):
    """Instantiate the agent's models (function approximators)."""

    net_cfg = MlpCfg(
        input_states=env.observation_space,
        hidden_units=[400, 300],
        activation=nn.ReLU,
    )

    net_cfg_with_actions = net_cfg.copy()
    net_cfg_with_actions.input_actions = env.action_space

    policy_model = build_model(
        RLModelCfg(
            network=net_cfg,
            device=device,
            output_layer=GaussianLayerCfg(
                output_size=env.action_space,
                output_scale=2.0,
                min_log_std=-1.2,
                max_log_std=2,
                initial_log_std=0.0,
            ),
        )
    )

    critic_model_cfg = RLModelCfg(
        network=net_cfg_with_actions,
        device=device,
        output_layer=DeterministicLayerCfg(),
    )

    models = {
        "policy": policy_model,
        "critic_1": build_model(critic_model_cfg),
        "critic_2": build_model(critic_model_cfg),
        "target_critic_1": build_model(critic_model_cfg),
        "target_critic_2": build_model(critic_model_cfg),
    }
    print(models)
    return models


# set seed for reproducibility
seed = 123
set_seed(seed)

# load and wrap the gymnasium environment.
env = gym.make_vec("Pendulum-v1", num_envs=4, vectorization_mode="sync")
env.reset(seed=seed)
env = GymnasiumWrapper(env)
device = env.device

# instantiate a memory as experience replay
memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)

# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = get_model(env)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.98
cfg["batch_size"] = 100
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1000
cfg["learn_entropy"] = True
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 75
cfg["experiment"]["checkpoint_interval"] = 750
cfg["experiment"]["directory"] = "runs/skrl/torch/"
cfg["experiment"]["experiment_name"] = (
    os.path.splitext(os.path.basename(__file__))[0] + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
)

params = {
    "models": models,
    "memory": memory,
    "cfg": cfg,
    "observation_space": env.observation_space,
    "action_space": env.action_space,
    "device": device,
}

if models["policy"].is_rnn:
    agent = SAC_RNN(**params)
else:
    agent = SAC(**params)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 15000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
