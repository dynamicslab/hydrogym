import os

import hydrogym.firedrake as hgym


def test_checkpointing(tmp_dir="tmp"):
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    checkpoint_path = f"{tmp_dir}/cyl.h5"

    env_config = {
        "flow": hgym.Cylinder,
        "flow_config": {
            "mesh": "coarse",  # Default mesh
        },
        "solver": hgym.IPCS,
    }
    env = hgym.FlowEnv(env_config)

    env.reset()
    for i in range(10):
        env.step(1)

    env.flow.save_checkpoint(checkpoint_path)
    omega = env.flow.actuators[0].value

    # new environment loading checkpoint
    env_config2 = {
        "flow": hgym.Cylinder,
        "flow_config": {
            "mesh": "coarse",  # Default mesh
            "restart": checkpoint_path,
        },
        "solver": hgym.IPCS,
    }
    env2 = hgym.FlowEnv(env_config2)

    # env2.reset()
    omega2 = env2.flow.actuators[0].value
    assert omega == omega2

    # Check that resetting still clears the actuator state
    env2.reset()
    omega3 = env2.flow.actuators[0].value
    assert omega3 == 0.0


if __name__ == "__main__":
    test_checkpointing()
