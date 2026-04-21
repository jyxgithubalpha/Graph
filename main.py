import argparse
import os

from domain.config import ExperimentConfig
from data.io import load_raw_dfs,clean_raw_dfs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, choices=[0, 1, 2, 3], default=0)
    ap.add_argument("--valid_period", type=int, choices=[1, 2, 3, 4], default=1)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    exp_cfg = ExperimentConfig()
    exp_cfg.run.valid_period = args.valid_period
    
        


if __name__ == "__main__":
    main()
