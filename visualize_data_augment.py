"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

from engine.misc import dist_utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed()
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    solver.visualize_data_augmentation(num_batches=args.num_batches)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, default='')
    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')
    # env
    parser.add_argument('--num_batches', type=int, default=0, help='number of batches to visualize')
    args = parser.parse_args()

    main(args)
