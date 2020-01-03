import os
from argparse import ArgumentParser
from shutil import copy
from time import gmtime, strftime

import yaml

from logger import Logger
from train import Trainer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    if args.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(args.checkpoint)[:-1])
    else:
        log_dir = os.path.join(args.log_dir, os.path.basename(args.config).split('.')[0])
        log_dir += ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime())

    args.log_dir = log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
        copy(args.config, log_dir)

    logger = Logger(log_dir)

    trainer = Trainer(logger, args.checkpoint, args.device_ids, config)
    trainer.train()
