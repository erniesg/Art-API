from art_api import config, data, utils, preproc, trainer
import pandas as pd
import os
from google.cloud import storage
from IPython.display import Image
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential, layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input as preproc_resnet
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
from keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow import keras

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        
loss_history = LossHistory()
lrate = LearningRateScheduler(step_decay)

def tuner():
    """Add to dataset with images scraped off Google +
    Augment images for training (xxx)
    + Unfree last block
    + Use batch size of 16
    """
    imgs = []
    df = data.DataLoader.add_img("../raw_data/google_sm")
    df_sample = preproc.sample(df)
    X, y = utils.load_add(df_sample)
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
    print(f"the shape of X_train is {X_train.shape}")
    print(f"the shape of X_val is {X_val.shape}")
    print(f"the shape of X_test is {X_test.shape}")
    print(f"the shape of y_train is {y_train.shape}")
    print(f"the shape of y_val is {y_val.shape}")
    print(f"the shape of y_test is {y_test.shape}")

    print("\n\nMake sure we're using GPU\n\n")
    
    try:
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        print(os.getenv("TF_GPU_ALLOCATOR"))
        print(f"memory usage {tf.config.experimental.get_memory_info('GPU:0')['current'] / 10 ** 9} GB")
    except:
        pass
    
    print("\n\nFine-tune pre-trained ResNet152 model\n\n")
    
    model = load_model("../raw_data/models/resnet152")
    model_name = "tuned_resnet152_existing_v4_aug"
    model.summary()
    
    print("\n\nIndex the number of layers in the ResNet152 base\n\n")
    
    model.layers[0].summary()
    print(f"There are {len(model.layers[0].layers)} layers in the base model.")

    # freeze base, with exception of last layer
    model.trainable = True
    set_trainable = False

    for layer in model.layers[0].layers:
        if 'conv5' in layer.name:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    
    # sanity check on trainable/untrainable params in base
    model.layers[0].summary()
    
    # sanity check on trainable/untrainable params in model
    print(f"Model summary after unfreezing conv5 for training")
    model.summary()
    
    data_augmentation = Sequential([layers.RandomFlip(),
                                layers.RandomRotation(0.1),
                                layers.RandomZoom(0.1),
                                layers.RandomTranslation(height_factor=0.1, width_factor=0.1)])

    x_train = data_augmentation(X_train) 

    resnet50_X_train = preproc_resnet(x_train) 
    resnet50_X_val = preproc_resnet(X_val)
    resnet50_X_test = preproc_resnet(X_test)
    
    # Custom Scheduler Function
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,
                                                              decay_steps=10000,
                                                              decay_rate=0.96)
        
    # compile and fit the model; during trainining, augment training images, change batch size to 16
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   optimizer=optimizers.Adam(learning_rate=1e-5),
                   metrics=['accuracy'])
            
    es = EarlyStopping(monitor = 'val_accuracy', 
                   mode = 'max', 
                   patience = 30, 
                   verbose = 1, 
                   restore_best_weights = True)
    
    history = model.fit(resnet50_X_train, y_train, 
                    validation_data=(resnet50_X_val, y_val), 
                    epochs=150, 
                    batch_size=32, 
                    callbacks=[es])
    
    if os.path.exists(f"../raw_data/models/{model_name}/"):
        model.save(f"../raw_data/models/{model_name}")
    else:
        os.makedirs(f"../raw_data/models/{model_name}")
        print(f"The directory for {model_name} is created.")
  
    model.save(f"../raw_data/models/{model_name}")
    print(f"Model {model_name} saved.")
    return model, model_name, history, resnet50_X_train, resnet50_X_val, resnet50_X_test, y_train, y_test, y_val

def evaluate(model, model_name, X_test, y_test, history):
    res = model.evaluate(X_test, y_test)
    res
    test_accuracy = res[-1]
    print(f"test_accuracy = {round(test_accuracy,2)*100} %")
        
    pd.DataFrame(res).to_csv(f"../raw_data/models/{model_name}/res.csv")

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../raw_data/models/' + str(model_name) + '/' + 'accuracy.png')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../raw_data/models/' + str(model_name) + '/' + 'loss.png')
    plt.show()
    y_pred = model.predict(X_test)
    return model, y_pred, y_test

if __name__ == "__main__":
    model, model_name, history, resnet50_X_train, resnet50_X_val, resnet50_X_test, y_train, y_test, y_val = tuner()
    model, y_pred, y_test = evaluate(model, model_name, resnet50_X_test, y_test, history)
    trainer.generate_reports(model, model_name, y_pred, y_test)