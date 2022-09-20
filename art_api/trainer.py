from art_api import config
import pandas as pd
import os
import numpy as np

# -*- coding: utf-8 -*-
'''Abstract base model'''

class BaseModel(ABC):
    '''Abstract Model class that is inherited to all models
    Behaviours:
    Get X and y
    Resize, load as array
    Preprocess input
    Get_config: hyperparameter tuning
    Model: baseline, VGG16, VGG19, ResNet, Inception, Xception
    Unfreeze layer
    Compile
    Fit
    Evaluate: accuracy, classification report, confusion matrix
    Predict with different thresholds (0.5, mean, median)
    Save model
    Df to csv
    '''
    def __init__(self, cfg):
        self.config = config.wandb.config

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

