import argparse
import logging
import os

from domain.config import ExperimentConfig
from train import run


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], default=4)
    ap.add_argument("--valid_period", type=int, choices=[1, 2, 3, 4], default=1)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    exp_cfg = ExperimentConfig()
    exp_cfg.run.valid_period = args.valid_period
    exp_cfg.graph.hist_len = exp_cfg.feature.hist_len
    assert exp_cfg.feature.hist_len <= exp_cfg.run.gap_days

    run(exp_cfg)


if __name__ == "__main__":
    main()
