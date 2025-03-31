import os
from ray.rllib.algorithms.ppo import PPOConfig
from pprint import pprint
from firedrake import logging
from tap import Tap

import hydrogym


class ArgumentParser(Tap):
    reynolds_number: int = 100  # Reynolds number
    mesh_resolution: str = "medium"  # Mesh resolution
    time_step: float = 1e-2  # Time step
    learning_rate: float = 0.0002  # Learning rate
    batch_size_per_learner: int = 2000  # Batch size per learner
    number_of_epochs: int = 10  # Number of epochs
    num_runners: int = 16  # Number of environment runners
    checkpoint_path: str = (
        "/home/lpaehler/Work/ReinforcementLearning/hydrogym-dev/ppo_checkpoint"  # Path to store checkpoints into
    )


def main():
    # Read in the command-line arguments
    args = ArgumentParser().parse_args()

    # Set the number of OpenMP threads to 1 to improve the performance of Firedrake
    os.environ["OMP_NUM_THREADS"] = "1"

    # Return Debug logging info
    logging.set_log_level(logging.DEBUG)

    # Define the logging callback
    log = hydrogym.firedrake.utils.io.LogCallback(
        postprocess=lambda flow: flow.get_observations(),
        nvals=2,
        interval=1,
        print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}",
        filename=None,
    )

    # Dictionary of the flow configuration
    flow_dict = {
        "flow": hydrogym.firedrake.Cylinder,
        "flow_config": {
            "Re": args.reynolds_number,
            "mesh": args.mesh_resolution,
        },
        "solver": hydrogym.firedrake.SemiImplicitBDF,
        "solver_config": {
            "dt": args.time_step,
        },
        "callbacks": [log],
        "max_steps": 10000,
    }

    # Create a config instance for the PPO algorithm.
    config = (
        PPOConfig()
        .environment(
            hydrogym.FlowEnv,
            env_config=flow_dict,
        )
        .env_runners(num_env_runners=args.num_runners, sample_timeout_s=300.0)
        .training(
            lr=args.learning_rate,
            train_batch_size_per_learner=args.batch_size_per_learner,
            num_epochs=args.number_of_epochs,
        )
    )

    # Construct the actual algorithm
    ppo = config.build_algo()

    for _ in range(4):
        pprint(ppo.train())

    # Store the trained algorithm
    ppo.save_to_path(args.checkpoint_path)


if __name__ == "__main__":
    main()
