from env.mujoco.build_env import build_env as build_mujoco_env


def build_envs(args, domain):

    # Select env builder
    build_env = build_mujoco_env

    # Build Environments
    src_env = build_env(args.env.lower().split("_")[0])
    tar_env = build_env(args.env.lower(), args.shift_level)
    src_eval_env = build_env(args.env.lower().split("_")[0])
    tar_eval_env = build_env(args.env.lower(), args.shift_level)

    # Set Seeds for the Environments
    src_env.reset(seed=args.seed)
    tar_env.reset(seed=args.seed)
    src_eval_env.reset(seed=args.seed + 42)
    tar_eval_env.reset(seed=args.seed + 42)

    return src_env, tar_env, src_eval_env, tar_eval_env
