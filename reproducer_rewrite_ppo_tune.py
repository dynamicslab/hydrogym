from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("Pendulum-v1")
    .training(
        lr=tune.grid_search([0.001, 0.0005, 0.0001]),
    )
)

# Creating a tuner instance to manage the trials
tuner = tune.Tuner(
    config.algo_class,
    param_space=config,
    # Specify a stopping criterion. Note that the criterion has to match one of the
    # pretty printed result metrics from the results returned previously by
    # ``.train()``. Also note that -1100 is not a good episode return for
    # Pendulum-v1, we are using it here to shorten the experiment time.
    run_config=train.RunConfig(
        stop={"env_runners/episode_return_mean": -1100.0},
    ),
)
# Run the Tuner and capture the results.
results = tuner.fit()