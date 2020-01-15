import os
from collections import defaultdict

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from chunk_rectangle_dataset import ChunkRectangleDataset
from networks import DCGenerator, DCDiscriminator
from sync_batchnorm import DataParallelWithCallback


def get_dataset(dataset, dataset_params):
    assert dataset in ['cifar10', 'cifar100', 'chunk_rectangle']
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    if dataset == 'cifar10':
        return torchvision.datasets.CIFAR10(root='cifar_data', train=True, download=True, transform=train_transform), 10
    elif dataset == 'cifar100':
        return torchvision.datasets.CIFAR100(root='cifar_data', train=True, download=True,
                                             transform=train_transform), 100
    elif dataset == 'chunk_rectangle':
        return ChunkRectangleDataset(transform=train_transform, **dataset_params), 1


class Trainer:
    def __init__(self, logger, checkpoint, device_ids, config):
        self.config = config
        self.logger = logger
        self.device_ids = device_ids

        self.dataset, n_classes = get_dataset(config['dataset'], config['dataset_params'])

        if self.config['with_labels']:
            self.config['generator_params']['n_classes'] = n_classes
            self.config['discriminator_params']['n_classes'] = n_classes
            self.config['n_classes'] = n_classes
        else:
            self.config['generator_params']['n_classes'] = None
            self.config['discriminator_params']['n_classes'] = None

        self.restore(checkpoint)

        print("Generator...")
        print(self.generator)

        print("Discriminator...")
        print(self.discriminator)

    def restore(self, checkpoint):
        self.epoch = 0

        self.generator = DCGenerator(**self.config['generator_params'])
        self.generator = DataParallelWithCallback(self.generator, device_ids=self.device_ids)
        self.optimizer_generator = torch.optim.Adam(params=self.generator.parameters(), lr=self.config['lr_generator'],
                                                    betas=(self.config['b1_generator'], self.config['b2_generator']),
                                                    weight_decay=0, eps=1e-8)

        self.discriminator = DCDiscriminator(**self.config['discriminator_params'])
        self.discriminator = DataParallelWithCallback(self.discriminator, device_ids=self.device_ids)
        self.optimizer_discriminator = torch.optim.Adam(params=self.discriminator.parameters(),
                                                        lr=self.config['lr_discriminator'],
                                                        betas=(self.config['b1_discriminator'],
                                                               self.config['b2_discriminator']),
                                                        weight_decay=0, eps=1e-8)

        if checkpoint is not None:
            data = torch.load(checkpoint)
            for key, value in data:
                if key == 'epoch':
                    self.epoch = value
                else:
                    self.__dict__[key].load_state_dict(value)

        lr_lambda = lambda epoch: 1 - epoch / self.config['num_epochs']
        self.scheduler_generator = torch.optim.lr_scheduler.LambdaLR(self.optimizer_generator, lr_lambda,
                                                                     last_epoch=self.epoch - 1)
        self.scheduler_discriminator = torch.optim.lr_scheduler.LambdaLR(self.optimizer_discriminator, lr_lambda,
                                                                         last_epoch=self.epoch - 1)

    def save(self):
        state_dict = {'epoch': self.epoch,
                      'generator': self.generator.state_dict(),
                      'optimizer_generator': self.optimizer_generator.state_dict(),
                      'discriminator': self.discriminator.state_dict(),
                      'optimizer_discriminator': self.optimizer_discriminator.state_dict()}

        torch.save(state_dict, os.path.join(self.logger.log_dir, 'cpk.pth'))

    def train(self):
        loader = DataLoader(self.dataset, batch_size=self.config['discriminator_bs'], shuffle=False,
                            drop_last=True, num_workers=self.config['num_workers'])
        noise = torch.zeros((max(self.config['generator_bs'], self.config['discriminator_bs']),
                             self.config['generator_params']['dim_z'])).cuda()
        if self.config['with_labels']:
            labels_fake = torch.zeros(max(self.config['generator_bs'], self.config['discriminator_bs'])).type(
                torch.LongTensor).cuda()
        else:
            labels_fake = None

        y_fake = None
        # Keep track of current iteration for update generator
        current_iter = 0
        loss_dict = defaultdict(lambda: 0.0)

        for self.epoch in tqdm(range(self.epoch, self.config['num_epochs'])):
            for data in tqdm(loader):
                self.generator.train()
                current_iter += 1

                images, labels_real = data
                y_real = None if not self.config['with_labels'] else labels_real

                self.optimizer_generator.zero_grad()
                self.optimizer_discriminator.zero_grad()

                z = noise.normal_()[:self.config['discriminator_bs']]
                if self.config['with_labels']:
                    y_fake = labels_fake.random_(self.config['n_classes'])[:self.config['discriminator_bs']]

                with torch.no_grad():
                    images_fake = self.generator(z, y_fake)

                logits_real = self.discriminator(images, y_real)
                logits_fake = self.discriminator(images_fake, y_fake)

                loss_fake = torch.relu(1 + logits_fake).mean()
                loss_real = torch.relu(1 - logits_real).mean()

                loss_dict['loss_fake'] += loss_fake.detach().cpu().numpy()
                loss_dict['loss_real'] += loss_real.detach().cpu().numpy()

                (loss_fake + loss_real).backward()
                self.optimizer_discriminator.step()

                if current_iter % self.config['num_discriminator_updates'] == 0:
                    self.optimizer_discriminator.zero_grad()
                    self.optimizer_generator.zero_grad()

                    z = noise.normal_()[:self.config['generator_bs']]
                    if self.config['with_labels']:
                        y_fake = labels_fake.random_(self.config['n_classes'])[:self.config['generator_bs']]

                    images_fake = self.generator(z, y_fake)
                    logits_fake = self.discriminator(images_fake, y_fake)

                    adversarial_loss = -logits_fake.mean()
                    loss_dict['adversarial_loss'] += adversarial_loss.detach().cpu().numpy()

                    adversarial_loss.backward()
                    self.optimizer_generator.step()

            save_dict = {key: value / current_iter for key, value in loss_dict.items()}
            save_dict['lr'] = self.optimizer_generator.param_groups[0]['lr']

            loss_dict = defaultdict(lambda: 0.0)
            current_iter = 0

            with torch.no_grad():
                noise = noise.normal_()
                if self.config['with_labels']:
                    labels_fake = labels_fake.random_(self.config['n_classes'])
                images = self.generator(noise, labels_fake)
                self.logger.save_images(self.epoch, images)

            # if self.epoch % self.config['eval_frequency'] == 0 or self.epoch == self.config['num_epochs'] - 1:
            #     self.generator.eval()
            #
            #     if self.config['samples_evaluation'] != 0:
            #         generated = []
            #         with torch.no_grad():
            #             for i in range(self.config['samples_evaluation'] // noise.shape[0] + 1):
            #                 noise = noise.normal_()
            #                 if self.config['with_labels']:
            #                     labels_fake = labels_fake.random_(self.config['n_classes'])
            #
            #                 generated.append((127.5 * self.generator(noise, labels_fake) + 127.5).cpu().numpy())
            #
            #             generated = np.concatenate(generated)[:self.config['samples_evaluation']]
            #             self.logger.save_evaluation_images(self.epoch, generated)

            self.logger.log(self.epoch, save_dict)

            self.scheduler_generator.step()
            self.scheduler_discriminator.step()
            self.save()
