# Modified from https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
import os

import ray
from common import CustomModel, TorchCustomModel, parser
from firedrake import logging
from ray import tune
from ray.rllib.agents import ppo  # ray.rllib.algorithms in latest version
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

import hydrogym

logging.set_log_level(logging.DEBUG)

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "cyl_actor", TorchCustomModel if args.framework == "torch" else CustomModel
    )

    # Set up the printing callback
    log = hydrogym.firedrake.utils.io.LogCallback(
        postprocess=lambda flow: flow.get_observations(),
        nvals=2,
        interval=1,
        print_fmt="t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}",
        filename=None,
    )

    config = {
        "log_level": "DEBUG",
        "horizon": 10000,
        "env": hydrogym.FlowEnv,
        "env_config": {
            "flow": hydrogym.firedrake.Cylinder,
            "flow_config": {
                "Re": 100,
                "restart": "../demo/checkpoint-coarse.h5",
                "mesh": "coarse",
            },
            "solver": hydrogym.firedrake.IPCS,
            "solver_config": {
                "dt": 1e-2,
            },
            "callbacks": [log],
            "max_steps": 10000,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "cyl_actor",
            "vf_share_layers": True,
        },
        # "model": {
        #     "fcnet_hiddens": [64, 64],
        #     "fcnet_activation": "relu",
        # },
        "num_workers": 1,  # parallelism
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if not args.tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config)
        # use fixed learning rate instead of grid search (needs tune)
        ppo_config["lr"] = 1e-3
        trainer = ppo.PPOTrainer(config=ppo_config, env=config["env"])
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = trainer.train()
            print(pretty_print(result))
            trainer.save("./rllib_checkpoint")
            # stop training of the target train steps or reward are reached
            if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["episode_reward_mean"] >= args.stop_reward
            ):
                break
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        results = tune.run(args.run, config=config, stop=stop)

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
