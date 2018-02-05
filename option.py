##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os

class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='Tank Shot')
        parser.add_argument('--log-dir', default='./logs',type=str,
                            help='folder to output model checkpoints')
        parser.add_argument('--batchSize', default=128,type=int,
                            help='Batch Size')
        parser.add_argument('--nthreads', default=4,type=int,
                            help='threads num to load data')
        parser.add_argument('--tensorname',default='EmbeddingNetwork',type=str,
                            help='tensorboard curve name')
        parser.add_argument('--decayepoch', default=60,type=int,
                            help='number of epoches to rescale learning rate')
        parser.add_argument('--network' ,default='resnet18',type=str,
                            help='network to provide image features')
        parser.add_argument('--trainepoch', default=300,type=int,
                            help='number of epoches to end training')
        parser.add_argument('--LR', default=0.1,type=float,
                            help='Learning rate of the Encoder Network')
        parser.add_argument('--ways', default=5,type=int,
                            help='number of class for one test')
        parser.add_argument('--shots', default=5,type=int,
                            help='number of pictures of each class to support')
        parser.add_argument('--test_num', default=15,type=int,
                            help='number of pictures of each class for test')
        parser.add_argument('--dimension', default=100,type=int,
                            help='dimension of the new embedding space')
        parser.add_argument('--trainways', default=5,type=int,
                            help='number of class for one episode in training')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
