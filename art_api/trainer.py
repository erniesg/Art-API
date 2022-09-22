from art_api import config
import pandas as pd
import os
import numpy as np
from abc import ABC, abstractmethod

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
    Evaluate: accuracy, losses, classification report, confusion matrix
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
        """model.compile
        """
        pass

    @abstractmethod
    def train(self):
        """model.fit
        """
        pass

    @abstractmethod
    def evaluate(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass

class BaselineModel(BaseModel):
    def __init__(self, config):
       super().__init__(config)
       self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.config.model.input, include_top=False)

    def load_data(self):
        # self.X = 
        # self. y =
        # self.dataset, self.info = DataLoader().load_data(self.config.data )
        # self._preprocess_data()

    def build(self):
        """Builds the Keras model"""

        self.model = tf.keras.Model(inputs=inputs, outputs=x)
        
        layer_names = [
            "base_layer",
            "flatten_layer",
            "dense_layer",
            "prediction_layer",
        ]
        layers = [self.base_model.get_layer(name).output for name in layer_names]
        
        

    def train(self):
        self.model.compile(optimizer=self.config.train.optimizer.type,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=self.config.train.metrics)

        model_history = self.model.fit(self.train_dataset, epochs=self.epoches,
                                       steps_per_epoch=self.steps_per_epoch,
                                       validation_steps=self.validation_steps,
                                       validation_data=self.test_dataset)

        return model_history.history['loss'], model_history.history['val_loss']

    def evaluate(self):
        predictions = []
        for image, mask in self.dataset.take(1):
            predictions.append(self.model.predict(image))

        return predictions

class VGG16(BaseModel):
  def __init__(self):
    super(LenetModel, self).__init__()

    self.feature_extractor = FeatureExtractor(8, 3, 1, "valid", "relu", 2)

    self.flatten = Flatten()

    self.dense_1 = Dense(100, activation = "relu")
    self.batch_1 = BatchNormalization()

    self.dense_2 = Dense(10, activation = "relu")
    self.batch_2 = BatchNormalization()

    self.dense_3 = Dense(1, activation = "sigmoid")
    
  def call(self, x, training):

    x = self.feature_extractor(x)
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.batch_1(x)
    x = self.dense_2(x)
    x = self.batch_2(x)
    x = self.dense_3(x)

    return x
    
lenet_sub_classed = LenetModel()
lenet_sub_classed(tf.zeros([1,224,224,3]))
lenet_sub_classed.summary()