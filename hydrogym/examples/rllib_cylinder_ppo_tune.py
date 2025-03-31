import os
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from firedrake import logging
from tap import Tap

import hydrogym


class ArgumentParser(Tap):
    reynolds_number: int = 100  # Reynolds number
    mesh_resolution: str = "medium"  # Mesh resolution
    time_step: float = 1e-2  # Time step
    batch_size_per_learner: int = 2000  # Batch size per learner
    number_of_epochs: int = 10  # Number of epochs
    num_runners: int = 8  # Number of environment runners


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

    config = (
        PPOConfig()
        .environment(
            hydrogym.FlowEnv,
            env_config=flow_dict,
        )
        .env_runners(num_env_runners=args.num_runners, sample_timeout_s=300.0)
        .training(
            lr=tune.grid_search([0.0005, 0.0002, 0.0001]),
            train_batch_size_per_learner=args.batch_size_per_learner,
            num_epochs=args.number_of_epochs,
        )
    )

    # Creating a tuner instance to manage the trials
    tuner = tune.Tuner(
        config.algo_class,
        param_space=config,
        # Specify a stopping criterion. Note that the criterion has to match one of the
        # pretty printed result metrics from the results returned previously by
        # ``.train()``
        # TODO: Doesn't work properly so far. The stop condition is not read in.
        run_config=train.RunConfig(
            stop={"CD": 0.2},
        ),
    )

    # Run the Tuner and capture the results.
    tuner.fit()


if __name__ == "__main__":
    main()
