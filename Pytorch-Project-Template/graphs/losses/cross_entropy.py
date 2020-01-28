"""
Cross Entropy 2D for CondenseNet
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from utils.generate_class_weights import *

####################

from sklearn.utils.class_weight import compute_class_weight

import scipy.io as sio
import PIL
from PIL import Image

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as standard_transforms

import utils.voc_utils as extended_transforms
from utils.voc_utils import make_dataset
'''
class Config:
            mode = "train"
            num_classes = 21
            batch_size = 32
            max_epoch = 150
            validate_every = 2
            checkpoint_file = "checkpoint.pth.tar"
            data_loader = "VOCDataLoader"
            data_root = "../data/"
            data_loader_workers = 4
            pin_memory = True
            async_loading = True
'''
class CrossEntropyLoss(nn.Module):
    
    def __init__(self, config=None):
        
        super(CrossEntropyLoss, self).__init__()
        ###########
        ##default##
        ###########
        #self.loss = nn.CrossEntropyLoss()
        
        if config == None:
            self.loss = nn.CrossEntropyLoss()
        else:
            '''
            # Create an instance from the data loader
            from tqdm import tqdm
            data_loader = VOCDataLoader(Config)
            z = np.zeros((Config.num_classes,))
            # Initialize tqdm
            tqdm_batch = tqdm(data_loader.train_loader, total=data_loader.train_iterations)
            
            for _, y in tqdm_batch:
                labels = y.numpy().astype(np.uint8).ravel().tolist()
                z += np.bincount(labels, minlength=Config.num_classes)
            tqdm_batch.close()
            #ret = compute_class_weight(class_weight='balanced', classes=np.arange(21), y=np.asarray(labels, dtype=np.uint8))
            total_frequency = np.sum(z)
            print(z)
            print(total_frequency)
            class_weights = []
            for frequency in z:
                class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
                class_weights.append(class_weight)
            ret = np.array(class_weights)
            np.save('../pretrained_weights/voc2012_256_class_weights', ret)
            print(ret)
            '''
            calculate_weigths_labels()
            class_weights = np.load(config.class_weights)
            print(class_weights)
            self.loss = nn.CrossEntropyLoss(ignore_index=config.ignore_index,
                                      weight=torch.from_numpy(class_weights.astype(np.float32)),
                                      size_average=True, reduce=True)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)
    
    