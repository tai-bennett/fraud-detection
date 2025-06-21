import numpy as np

from pprint import pprint
from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from src.data.dataset import FraudDataloader
from graphs.losses.cross_entropy import CrossEntropyLoss
from graphs.models.fraudnet import FraudNet

from agents.base import BaseAgent

# import your classes here

#from tensorboardX import SummaryWriter
#from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class FraudNetAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # define data_loader
        self.data_loader = FraudDataloader(config)
        self.config.input_dim = self.data_loader.dataset.dim
        class_weights = self.data_loader.class_weights
        self.config.class_weights = []
        for key, value in class_weights.items():
            self.config.class_weights.append(value)


        self.logger.info("Building model...")
        # define models
        self.model = FraudNet(config)

        self.logger.info("Define loss...")
        # define loss
        self.loss = CrossEntropyLoss(config)

        # define optimizers for both generator and discriminator
        self.optimizer = None

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        pass

    def run(self):
        """
        The main operator
        :return:
        """
        self.logger.info("Running agent...")
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        pass

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        pass

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info("Finalizing...")
