import argparse
import json
import os
import random

import numpy as np
import setproctitle
import torch
from tensorboardX import SummaryWriter
from tqdm import trange

from algo.build_policy import build_policy
from config.config import build_config
from env.build_envs import build_envs
from env.mujoco.build_env import build_env as build_mujoco_env
from utils.eval_policy import eval_policy
from utils.replay_buffer import ReplayBuffer

if __name__ == "__main__":
    # Parse required arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs", help="directory to save logs")
    parser.add_argument("--policy", default="TUNE_TB", help="policy to use")
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
        help="Save target-state coverage data to disk for later t-SNE visualisation",
    )

    args = parser.parse_args()

    assert args.policy.lower() in ["tune_tb", "tune_se"]

    # Replace '-' with '_' for consistency in argument names
    if "-" in args.env:
        args.env = args.env.replace("-", "_")

    setproctitle.setproctitle(f"{args.seed}-{args.policy}-{args.env}")

    # Seeding for consistency
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialise Tensorboard directory and writer
    output_dir = f"{args.dir}/{args.policy.upper()}/{args.env.lower()}_{args.shift_level}/r{str(args.seed)}"
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
    # for baseline algorithms, policy = policy
    policy = build_policy(args.policy, config, device)

    # Initialize Replay Buffers
    src_replay_buffer = ReplayBuffer(config["state_dim"], config["action_dim"], device)
    tar_replay_buffer = ReplayBuffer(config["state_dim"], config["action_dim"], device)

    # ---- t-SNE data collection setup ----
    # When --tsne is passed we write target next-states incrementally to binary
    # files so that we never carry large in-memory arrays.
    #
    # Every policy stores exactly one file per seed:
    #   tsne_data/{env}_{shift_level}/{POLICY}/r{seed}/target_states.bin
    #
    # For TUNE_TB specifically we also maintain a *shadow* target env that
    # always follows the un-tuned policy.  Its states are stored under a
    # virtual baseline policy called "UNTUNED":
    #   tsne_data/{env}_{shift_level}/UNTUNED/r{seed}/target_states.bin
    is_tune_tb = args.policy.upper() == "TUNE_TB"
    tsne_env_dir = None
    policy_states_file = None
    untuned_states_file = None
    shadow_tar_env = None

    if args.tsne:
        tsne_env_dir = f"./tsne_data/{args.env.lower()}_{args.shift_level}"

        # --- Directory & metadata for the actual policy ---
        policy_tsne_dir = os.path.join(
            tsne_env_dir, args.policy.upper(), f"r{args.seed}"
        )
        os.makedirs(policy_tsne_dir, exist_ok=True)

        with open(os.path.join(policy_tsne_dir, "metadata.json"), "w") as f:
            json.dump({"state_dim": config["state_dim"]}, f)

        policy_states_file = open(
            os.path.join(policy_tsne_dir, "target_states.bin"), "wb"
        )

        # --- TUNE_TB only: shadow env + UNTUNED baseline folder ---
        if is_tune_tb:
            untuned_tsne_dir = os.path.join(tsne_env_dir, "UNTUNED", f"r{args.seed}")
            os.makedirs(untuned_tsne_dir, exist_ok=True)

            with open(os.path.join(untuned_tsne_dir, "metadata.json"), "w") as f:
                json.dump({"state_dim": config["state_dim"]}, f)

            untuned_states_file = open(
                os.path.join(untuned_tsne_dir, "target_states.bin"), "wb"
            )

            # Build a shadow target env with the same dynamics & seed so that
            # the only variable between the two rollouts is the policy.
            # Mirrors how build_envs creates tar_env:
            #   build_env(args.env.lower(), args.shift_level)
            shadow_tar_env = build_mujoco_env(args.env.lower(), args.shift_level)
            shadow_tar_env.reset(seed=args.seed)
            shadow_tar_env.action_space.seed(args.seed)

    # Initialise state variables
    src_state, _ = src_env.reset()
    tar_state, _ = tar_env.reset()
    src_done, tar_done = False, False

    # Shadow target env state (only used when --tsne and TUNE_TB)
    if shadow_tar_env is not None:
        shadow_tar_state, _ = shadow_tar_env.reset()
        shadow_tar_done = False
    else:
        shadow_tar_state = None
        shadow_tar_done = False

    src_episode_reward, src_episode_num = 0, 1
    tar_episode_reward, tar_episode_num = 0, 1
    eval_cnt = 0
    src_eval_return = None
    tar_eval_return = None

    pbar = trange(config["max_steps"], desc="Timesteps")
    pbar.set_postfix({"src_eval": "N/A", "tar_eval": "N/A"})

    warmup_steps = config["max_steps"] // 10  # 10% of total steps for warmup

    for step in pbar:
        # Pick source Action
        src_action = (
            policy.select_action(np.array(src_state), test=False)
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

        src_state = src_next_state
        src_episode_reward += float(src_reward)

        if step % config["tar_env_interact_interval"] == 0:
            if step < warmup_steps:
                # Pick target Action normally (untuned)
                tar_action = policy.select_action(np.array(tar_state), test=False)
            else:
                tar_action = policy.select_tuned_action(
                    np.array(tar_state), tar_replay_buffer, config["batch_size"]
                )

            # Step in target environment
            tar_next_state, tar_reward, tar_terminated, tar_truncated, _ = tar_env.step(
                tar_action
            )
            tar_done = float(tar_terminated or tar_truncated)

            tar_replay_buffer.add(
                tar_state, tar_action, tar_next_state, tar_reward, tar_done
            )

            # ---- t-SNE: write policy's target next-state to disk ----
            if policy_states_file is not None:
                np.array(tar_next_state, dtype=np.float32).tofile(policy_states_file)

            tar_state = tar_next_state
            tar_episode_reward += float(tar_reward)

            # ---- Shadow (untuned) target env rollout for t-SNE (TUNE_TB only) ----
            if shadow_tar_env is not None:
                # The shadow env *always* uses the un-tuned policy
                shadow_action = policy.select_action(
                    np.array(shadow_tar_state), test=False
                )

                (
                    shadow_next_state,
                    _shadow_reward,
                    shadow_terminated,
                    shadow_truncated,
                    _,
                ) = shadow_tar_env.step(shadow_action)
                shadow_tar_done = float(shadow_terminated or shadow_truncated)

                # Write untuned next-state to the UNTUNED baseline file
                if untuned_states_file is not None:
                    np.array(shadow_next_state, dtype=np.float32).tofile(
                        untuned_states_file
                    )

                shadow_tar_state = shadow_next_state

                if shadow_tar_done:
                    shadow_tar_state, _ = shadow_tar_env.reset()
                    shadow_tar_done = False

        # Train the policy
        policy.train(src_replay_buffer, tar_replay_buffer, config["batch_size"], writer)

        # Src episode complete
        if src_done:
            writer.add_scalar(
                "train/source return", src_episode_reward, global_step=step + 1
            )
            src_state, _ = src_env.reset()
            src_done = False
            src_episode_reward = 0
            src_episode_num += 1

        # Tar episode complete
        if tar_done:
            writer.add_scalar(
                "train/target return", tar_episode_reward, global_step=step + 1
            )
            tar_state, _ = tar_env.reset()
            tar_done = False
            tar_episode_reward = 0
            tar_episode_num += 1

        # Evaluate the policy
        if (step + 1) % config["eval_freq"] == 0:
            src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
            tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
            writer.add_scalar(
                "test/source return", src_eval_return, global_step=step + 1
            )
            writer.add_scalar(
                "test/target return", tar_eval_return, global_step=step + 1
            )
            pbar.set_postfix(
                {
                    "src_eval": f"{src_eval_return:.1f}",
                    "tar_eval": f"{tar_eval_return:.1f}",
                }
            )
            eval_cnt += 1

    # ---- Cleanup t-SNE files ----
    if policy_states_file is not None:
        policy_states_file.close()
    if untuned_states_file is not None:
        untuned_states_file.close()
    if shadow_tar_env is not None:
        shadow_tar_env.close()

    if args.tsne and tsne_env_dir is not None:
        print(f"t-SNE state data saved to {tsne_env_dir}")
