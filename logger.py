import os
from collections import OrderedDict

import numpy as np
import torch
from torchvision.utils import save_image

from util import denorm


class Logger:
    def __init__(self, log_dir, log_file='log.txt', vis_dir='train-vis', eval_dir='eval'):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.vis_dir = os.path.join(log_dir, vis_dir)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        self.eval_dir = os.path.join(log_dir, eval_dir)
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

        self.cpk_dir = log_dir

    def save_images(self, epoch, *args):
        x_concat = torch.cat(args, dim=3)
        sample_path = os.path.join(self.vis_dir, str(epoch).zfill(5) + '.png')
        save_image(denorm(x_concat.data.cpu()), sample_path, nrow=8, padding=0)

    def save_evaluation_images(self, epoch, images):
        sample_path = os.path.join(self.eval_dir, str(epoch).zfill(5) + '.npy')
        np.save(sample_path, images)

    def log(self, epoch, scores):
        result = ''
        for key, value in OrderedDict(scores).items():
            result += '{}:{} '.format(key, value)
        with open(self.log_file, 'a') as f:
            print('{})'.format(str(epoch).zfill(3)) + result, file=f)
