import argparse
import random

import numpy as np
import setproctitle
import torch
from tensorboardX import SummaryWriter
from tqdm import trange

from algo.build_policy import build_policy
from config.config import build_config
from env.build_envs import build_envs
from utils.eval_policy import eval_policy
from utils.replay_buffer import ReplayBuffer
from utils.tsne_plot import plot_tsne

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
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Generate t-SNE plot of state-space coverage at the end of training",
    )

    args = parser.parse_args()

    # Replace '-' with '_' foconsistency in argument names
    if "-" in args.env:
        args.env = args.env.replace("-", "_")

    setproctitle.setproctitle(f"{args.seed}-{args.policy}-{args.env}")

    # Seeding for consistency
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialise Tensorboard directory and writer
    output_dir = (
        f"{args.dir}/{args.policy.upper()}/{args.env.lower()}/r{str(args.seed)}"
    )
    writer = SummaryWriter(f"{output_dir}/tb")

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
    device = torch.device(config["device"])

    # Get src and tar policies
    # for baseline algorithms, src_policy = tar_policy
    src_policy, tar_policy = build_policy(args.policy, config, device)

    # Initialize Replay Buffers
    src_replay_buffer = ReplayBuffer(config["state_dim"], config["action_dim"], device)
    tar_replay_buffer = ReplayBuffer(config["state_dim"], config["action_dim"], device)

    # Online state collection for t-SNE visualization
    src_states_collected = []
    tar_states_collected = []

    # Initialise state variables
    src_state, _ = src_env.reset()
    tar_state, _ = tar_env.reset()
    src_done, tar_done = False, False

    src_episode_reward, src_episode_num = 0, 1
    tar_episode_reward, tar_episode_num = 0, 1
    eval_cnt = 0

    pbar = trange(config["max_steps"], desc="Timesteps")
    pbar.set_postfix({"src_ep": src_episode_num, "tar_ep": tar_episode_num})

    for step in pbar:
        # Pick source Action
        src_action = (
            src_policy.select_action(np.array(src_state), test=False)
            + np.random.normal(0, config["max_action"] * 0.2, size=config["action_dim"])
        ).clip(config["min_action"], config["max_action"])

        # Step in source environment
        src_next_state, src_reward, src_terminated, src_truncated, _ = src_env.step(
            src_action
        )
        src_done = float(src_terminated or src_truncated)

        src_replay_buffer.add(
            src_state, src_action, src_next_state, src_reward, src_done
        )

        if args.tsne:
            src_states_collected.append(src_state)
        src_state = src_next_state
        src_episode_reward += float(src_reward)

        if step % config["tar_env_interact_interval"] == 0:
            # Pick target Action
            tar_action = tar_policy.select_action(np.array(tar_state), test=False)

            # Step in target environment
            tar_next_state, tar_reward, tar_terminated, tar_truncated, _ = tar_env.step(
                tar_action
            )
            tar_done = float(tar_terminated or tar_truncated)

            tar_replay_buffer.add(
                tar_state, tar_action, tar_next_state, tar_reward, tar_done
            )

            if args.tsne:
                tar_states_collected.append(tar_state)
            tar_state = tar_next_state
            tar_episode_reward += float(tar_reward)

        # Train the policy
        src_policy.train(
            src_replay_buffer, tar_replay_buffer, config["batch_size"], writer
        )
        # Separately train target policy when required by algorithm
        if False:
            tar_policy.train(
                src_replay_buffer, tar_replay_buffer, config["batch_size"], writer
            )

        # Src episode complete
        if src_done:
            writer.add_scalar(
                "train/source return", src_episode_reward, global_step=step + 1
            )
            src_state, _ = src_env.reset()
            src_done = False
            src_episode_reward = 0
            src_episode_num += 1
            pbar.set_postfix({"src_ep": src_episode_num, "tar_ep": tar_episode_num})

        # Tar episode complete
        if tar_done:
            writer.add_scalar(
                "train/target return", tar_episode_reward, global_step=step + 1
            )
            tar_state, _ = tar_env.reset()
            tar_done = False
            tar_episode_reward = 0
            tar_episode_num += 1
            pbar.set_postfix({"src_ep": src_episode_num, "tar_ep": tar_episode_num})

        # Evaluate the policy
        if (step + 1) % config["eval_freq"] == 0:
            src_eval_return = eval_policy(src_policy, src_eval_env, eval_cnt=eval_cnt)
            tar_eval_return = eval_policy(src_policy, tar_eval_env, eval_cnt=eval_cnt)
            writer.add_scalar(
                "test/source return", src_eval_return, global_step=step + 1
            )
            writer.add_scalar(
                "test/target return", tar_eval_return, global_step=step + 1
            )
            eval_cnt += 1

    # Generate t-SNE plot at the end of training
    if args.tsne:
        tsne_output_dir = f"./tsne_plots/{args.policy.upper()}/{args.env.lower()}/r{args.seed}"
        plot_tsne(
            np.array(src_states_collected),
            np.array(tar_states_collected),
            tsne_output_dir,
        )
