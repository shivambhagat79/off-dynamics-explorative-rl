import argparse

import setproctitle
import torch

from algo.build_policy import build_policy
from config.config import build_config
from env.build_envs import build_envs

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
        "--tar_env_interact_interval",
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

    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA (GPU) is available. Using GPU.")
    else:
        device = "cpu"
        print("CUDA not available. Using CPU.")

    if any(s in args.env.lower() for s in ["halfcheetah", "walker2d", "ant", "hopper"]):
        domain = "mujoco"
    else:
        raise NotImplementedError("Environment not supported")

    # Build Environments
    src_env, tar_env, src_eval_env, tar_eval_env = build_envs(args, domain)

    # Generate Config
    config = build_config(args, domain, src_env, device)

    # Get src and tar policies
    # for baseline algorithms, src_policy = tar_policy
    src_policy, tar_policy = build_policy()

    for step in range(args.max_steps):
        pass
