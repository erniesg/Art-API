from art_api import config
import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Input, Dropout, RandomFlip, RandomRotation, Resizing, Rescaling
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC, binary_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers  import L2, L1
from tensorboard.plugins.hparams import api as hp
from abc import ABC, abstractmethod

# -*- coding: utf-8 -*-

"""Use functional API to write all models
    Get X and y
    Resize, load as array
    Preprocess input
    Get_config: hyperparameter tuning
    Model: baseline, VGG16, VGG19, ResNet, Inception, Xception
        Shared layers: rescaling, augmentation, flatten, dense, prediction
    Unfreeze layer
    Compile
    Fit
    Evaluate: accuracy, losses, classification report, confusion matrix
    Predict with different thresholds (0.5, mean, median)
    Save model
    Df to csv
"""

func_input = Input(shape = (config.IM_SIZE, config.IM_SIZE, 3), name = "Input image")

x = Rescaling(1./255)
x = Conv2D(filters = 16, kernel_size = 10, activation = 'relu')(func_input)
x = MaxPool2D(pool_size = 3)(x)

x = Conv2D(filters = 32, kernel_size = 32, activation = 'relu')(func_input)
x = MaxPool2D(pool_size = 3)(x)

x = Conv2D(filters = 32, kernel_size = 32, activation = 'relu')(func_input)
x = MaxPool2D(pool_size = 3)(x)

x = Flatten()(x)
x = Dense(100, activation="relu")
x = Dense(10, activation="sigmoid")

x = Conv2D(filters = 6, kernel_size = 3, strides=1, padding='valid', activation = 'relu')(func_input)
x = BatchNormalization()(x)
x = MaxPool2D (pool_size = 2, strides= 2)(x)

x = Conv2D(filters = 16, kernel_size = 3, strides=1, padding='valid', activation = 'relu')(x)
x = BatchNormalization()(x)
output = MaxPool2D (pool_size = 2, strides= 2)(x)

feature_extractor_model = Model(func_input, output, name = "Feature_Extractor")
feature_extractor_model.summary()
feature_extractor_seq_model = tf.keras.Sequential([
                             InputLayer(input_shape = (IM_SIZE, IM_SIZE, 3)),

                             Conv2D(filters = 6, kernel_size = 3, strides=1, padding='valid', activation = 'relu'),
                             BatchNormalization(),
                             MaxPool2D (pool_size = 2, strides= 2),

                             Conv2D(filters = 16, kernel_size = 3, strides=1, padding='valid', activation = 'relu'),
                             BatchNormalization(),
                             MaxPool2D (pool_size = 2, strides= 2),

                             

])
feature_extractor_seq_model.summary()

func_input = Input(shape = (IM_SIZE, IM_SIZE, 3), name = "Input Image")

x = feature_extractor_seq_model(func_input)

x = Flatten()(x)

x = Dense(100, activation = "relu")(x)
x = BatchNormalization()(x)

x = Dense(10, activation = "relu")(x)
x = BatchNormalization()(x)

func_output = Dense(1, activation = "sigmoid")(x)

lenet_model_func = Model(func_input, func_output, name = "Lenet_Model")
lenet_model_func.summary()

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