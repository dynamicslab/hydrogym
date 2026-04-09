import os

from hydrogym.jaxfluids import Nozzle2D


def main():
    env_config = {
        "environment_name": "Nozzle2D_coarse",
        "configuration_file": os.path.abspath("environment_config.yaml")
    }

    env = Nozzle2D(env_config=env_config)

    observation, info = env.reset(seed=0)
    env.render()

    for i in range(1000):

        # Random action
        # action = env.action_space.sample()       

        # Fixed action
        action = [0.0, 0.5]                  

        observation, reward, terminated, truncated, info = env.step(action)

        if env.env_step % 10 == 0:
            env.render()

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()