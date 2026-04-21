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
    ap.add_argument("--gpus", type=int, choices=[0, 1, 2, 3], default=0)
    ap.add_argument("--valid_period", type=int, choices=[1, 2, 3, 4], default=1)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    exp_cfg = ExperimentConfig()
    exp_cfg.run.valid_period = args.valid_period

    run(exp_cfg)


if __name__ == "__main__":
    main()
