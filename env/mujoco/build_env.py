import os

import gymnasium as gym
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv
from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv
from gymnasium.wrappers import TimeLimit


def build_env(env_name, shift_level=None):
    if "kinematic" in env_name or "morph" in env_name:
        assert shift_level is not None
        assert shift_level.lower() in ["easy", "medium", "hard"]
    elif "friction" in env_name or "gravity" in env_name:
        assert shift_level in [0.1, 0.5, 2.0, 5.0]

    env_class = {
        "halfcheetah": HalfCheetahEnv,
        "hopper": HopperEnv,
        "walker2d": Walker2dEnv,
        "ant": AntEnv,
    }[env_name.split("_")[0]]

    gym_env_name = {
        "halfcheetah": "HalfCheetah-v5",
        "hopper": "Hopper-v5",
        "walker2d": "Walker2d-v5",
        "ant": "Ant-v5",
    }

    if env_name.lower() in ["halfcheetah", "hopper", "walker2d", "ant"]:
        return gym.make(gym_env_name[env_name.split("_")[0]])

    assert shift_level is not None
    xml_file_path = f"./env/mujoco/assets/{env_name}_{shift_level}.xml"
    assert os.path.exists(xml_file_path)

    return TimeLimit(
        env_class(xml_file=xml_file_path),
        max_episode_steps=1000,
    )
    return 3
