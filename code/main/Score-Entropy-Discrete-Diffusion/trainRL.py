import hydra
import numpy as np
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import open_dict
import utils
import os
import run_rl
import torch

@hydra.main(version_base=None, config_path="configs", config_name="rl_latex")
def main(cfg):
    ngpus = min(cfg.ngpus, torch.cuda.device_count())  # 自动适配，单卡时为1

    base_dir = "/root/autodl-tmp/exp_rl"
    utils.makedirs(base_dir)

    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode == RunMode.RUN:
        model_str = "model-rl-latex1"
        work_dir = os.path.join(base_dir, model_str)
    else:
        work_dir = os.path.join(base_dir, hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    utils.makedirs(work_dir)

    with open_dict(cfg):
        cfg.ngpus = ngpus
        cfg.work_dir = work_dir

    port = int(np.random.randint(10000, 20000))
    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    try:
        mp.set_start_method("forkserver")
        mp.spawn(
            run_rl.run_multiprocess,
            args=(ngpus, cfg, port),
            nprocs=ngpus,
            join=True
        )
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()