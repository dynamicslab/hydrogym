import argparse

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO", help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument("--stop-iters", type=int, default=10000, help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=1000000,
    help="Max number of timesteps to train.",
)
parser.add_argument(
    "--stop-reward", type=float, default=1e8, help="Reward at which we stop training."
)
parser.add_argument(
    "--tune",
    action="store_true",
    help="Run using Tune with grid search and TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        if isinstance(input_dict["obs"], tuple):
            input_dict["obs"] = torch.stack(input_dict["obs"], dim=1)
            input_dict["obs_flat"] = input_dict["obs"]
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])
