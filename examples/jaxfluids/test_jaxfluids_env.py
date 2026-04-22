"""
JAX-Fluids Environment Test Script
=============================

Test script for JAX-Fluids-based CFD environments

All JAX-Fluids-based hydrogym environments inherit from JAXFluidsFlowEnv,
which serves as the common base class.

JAXFluidsFlowEnv has the following arguments:
    - environment_name: Required. Name of the enviroment.
    - hf_repo_id: Hugging Face repository (default: 'dynamicslab/HydroGym-environments')

    - use_clean_cache: Use clean cache directory (default: True)
        * True - Creates fresh workspace copy (recommended for production)
        * False - Uses cached workspace (faster for development/testing)
    - local_fallback_dir: Local directory for offline usage
    - configuration_file: Custom path to MAIA config.yaml (optional)

    - output_dir: Optional. String indicating where the environment outputs are saved.
        Defaults to 'outputs'.
    - run_name: Optional. String for the name of the run. Defaults to datetime.now().

    - steps_per_action: Required. Integer, number of JAX-Fluids integration steps
        per action.

    - render_mode: Optional. String or None, indicating whether and how the environment
        is rendered. Defaults to None.
    - render_dpi: Optional. Integer for the dpi of the rendered images. Defaults to 300.

    - log_level: Optional. String indicating the log level.
    - log_to_file: Optional. Boolean indicating whether logs are written to a log file.
    - log_every_steps: Optional. Integer indicating every how many steps log is written.
        Defaults to 10.


Available environments:
    2D & 3D Flows:
    - Nozzle2D: Flow through a 2D nozzle with shock vector control by secondary injection
        The Nozzle2D environment has the following additional arguments:
            - secondary_pressure_ratio: Optional. Float, must be between 0.7 and 0.9.
            Defaults to 0.7.
            - resolution: Optional. Spatial resolution of the environment. Choose either 'coarse' or 'fine'.
            - ngpus: Optional. Number of GPUs for running the environment.
            - is_pressure_probes: Optional. Boolean indicating whether pressure probes
                are part of the observation
            - is_scale_observations: Optional. Boolean indicating whether observations are scaled to [0, 1].
            - target_fn: Optional. Target thrust vector function. Choose either 'sine' or 'step'.

        The Nozzle3D environment has the following additional arguments:
            - num_actuators: Required. Integer number of actuators. Must be between 4 and 12.
            - secondary_pressure_ratio: Optional. Float, must be between 0.7 and 0.9.
                Defaults to 0.7.
            - resolution: Optional. Spatial resolution of the environment. Choose either 'coarse' or 'fine'.
            - ngpus: Optional. Number of GPUs for running the environment.
            - is_pressure_probes: Optional. Boolean indicating whether pressure probes
                are part of the observation
            - is_scale_observations: Optional. Boolean indicating whether observations are scaled to [0, 1].
            - target_fn: Optional. Target thrust vector function. Choose either 'sine' or 'step'.

"""

import os

from hydrogym.jaxfluids import Nozzle2D


def main():
    env_config = {
        "environment_name": "Nozzle2D_coarse",
        "configuration_file": os.path.abspath("environment_config.yaml"),
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
