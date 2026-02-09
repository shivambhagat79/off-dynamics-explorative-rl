from algo.algos.darc import DARC
from algo.algos.par import PAR
from algo.algos.sac import SAC
from algo.algos.vgdf import VGDF


def build_policy(algo_name, config, device):
    algo_name = algo_name.lower()

    algo_map = {"sac": SAC, "darc": DARC, "vgdf": VGDF, "par": PAR}

    algo = algo_map[algo_name]
    policy = algo(config, device)
    tar_policy = policy

    # For policies with separate src and tar policies
    if False:
        tar_policy = algo(config, device)

    return policy, tar_policy
