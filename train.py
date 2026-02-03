import argparse

import setproctitle

from env.mujoco.call_env import call_env as call_mujoco_env
from policy.call_policy import call_policy

if __name__ == "__main__":
    # Parse required arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs", help="directory to save logs")
    parser.add_argument("--policy", default="SAC", help="policy to use")
    parser.add_argument("--env", default="halfcheetah-friction")
    parser.add_argument(
        "--seed", type=int, default=0, help="Manual seed for replication"
    )
    parser.add_argument(
        "--shift_level",
        default=0.1,
        help="The scale of the dynamics shift. Note that this value varies on different settins",
    )
    parser.add_argument(
        "--tar_env_interaction_interval",
        default=10,
        type=int,
        help="The interval of interactions with the target environment",
    )
    parser.add_argument("--max_steps", default=int(1e6), type=int)

    args = parser.parse_args()

    # Replace '-' with '_' foconsistency in argument names
    if "-" in args.env:
        args.env = args.env.replace("-", "_")

    setproctitle.setproctitle(f"{args.seed}-{args.policy}-{args.env}")

    env = call_mujoco_env()
    policy = call_policy()
    
    for step in range(args.max_steps):
        
