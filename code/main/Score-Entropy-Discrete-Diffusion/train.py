"""Training and evaluation"""

import hydra
import os
import numpy as np
import run_train
import utils
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict
import torch

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    # ngpus = cfg.ngpus
    ngpus = min(cfg.ngpus, torch.cuda.device_count())#取gpu配置和实际使用的最小值
    if "load_dir" in cfg:
        hydra_cfg_path = os.path.join(cfg.load_dir, ".hydra/hydra.yaml")
        hydra_cfg = OmegaConf.load(hydra_cfg_path).hydra

        cfg = utils.load_hydra_config_from_run(cfg.load_dir)
        # 强制将断点续训的work_dir也指向数据盘
        work_dir = cfg.work_dir.replace("/root/", "/root/autodl-tmp/") if "/root/" in cfg.work_dir else os.path.join(
            "/root/autodl-tmp/exp_local", os.path.basename(cfg.work_dir))
        utils.makedirs(work_dir)
        # work_dir = cfg.work_dir
        # utils.makedirs(work_dir)

    # else:
    #     hydra_cfg = HydraConfig.get()
    #     work_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    #     utils.makedirs(work_dir)
    else:
        hydra_cfg = HydraConfig.get()
        # 固定数据盘根目录
        base_dir = "/root/autodl-tmp/exp_local"
        utils.makedirs(base_dir)
        # 生成子目录（保持原有目录结构，仅换根路径）
        if hydra_cfg.mode == RunMode.RUN:
            model_str = "model-small-latex0.0"
            work_dir = os.path.join(base_dir, model_str)
        else:
            work_dir = os.path.join(base_dir, hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
        utils.makedirs(work_dir)

    with open_dict(cfg):
        cfg.ngpus = ngpus
        cfg.work_dir = work_dir
        cfg.wandb_name = os.path.basename(os.path.normpath(work_dir))

	# Run the training pipeline
    port = int(np.random.randint(10000, 20000))
    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_cfg.job.id}")

    try:
        mp.set_start_method("forkserver")
        mp.spawn(run_train.run_multiprocess, args=(ngpus, cfg, port), nprocs=ngpus, join=True)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()