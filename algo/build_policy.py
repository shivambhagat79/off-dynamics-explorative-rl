from algo.algos.darc import DARC
from algo.algos.par import PAR
from algo.algos.sac import SAC
from algo.algos.tune_se import TUNE_SE
from algo.algos.tune_tb import TUNE_TB
from algo.algos.vgdf import VGDF


def build_policy(algo_name, config, device):
    algo_name = algo_name.lower()

    algo_map = {
        "sac": SAC,
        "darc": DARC,
        "vgdf": VGDF,
        "par": PAR,
        "tune_tb": TUNE_TB,
        "tune_se": TUNE_SE,
    }

    algo = algo_map[algo_name]
    policy = algo(config, device)

    return policy
