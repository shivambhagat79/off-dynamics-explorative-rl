import os

import yaml


def build_config(args, domain, env, device):
    config_file_path = (
        f"./config/{domain}/{args.policy.lower()}/{args.env.lower().split('_')[0]}.yaml"
    )
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    max_action = env.action_space.high[0]
    min_action = -max_action

    config.update(
        {
            "env_name": args.env.lower(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "tar_env_interact_interval": args.tar_env_interact_interval,
            "max_steps": args.max_steps,
            "max_action": max_action,
            "min_action": min_action,
            "shift_level": args.shift_level,
            "domain": domain,
            "device": device,
        }
    )
    return config
