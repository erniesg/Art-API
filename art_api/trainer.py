import pandas as pd
from art_api import config, utils, preproc
import os
from IPython.display import Image
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input as preproc_vgg16
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as preproc_vgg19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input as preproc_resnet
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as preproc_inceptionv3
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preproc_inceptionv2
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as preproc_xception
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
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
def load_baseline_func():
    
    func_input = Input(shape = (config.IM_SIZE, config.IM_SIZE, 3), name = "input image")

    rescaling = Rescaling(1./255)(func_input)
    conv2d_1 = Conv2D(filters = 16, kernel_size = 10, activation = 'relu')(rescaling)
    pool1 = MaxPool2D(pool_size = 3)(conv2d_1)

    conv2d_2 = Conv2D(filters = 32, kernel_size = 8, activation = 'relu')(pool1)
    pool2 = MaxPool2D(pool_size = 3)(conv2d_2)

    conv2d_3 = Conv2D(filters = 32, kernel_size = 6, activation = 'relu')(pool2)
    pool3 = MaxPool2D(pool_size = 3)(conv2d_3)

    flatten = Flatten()(pool3)
    dense = Dense(500, activation="relu")(flatten)

    output = Dense(10, activation="sigmoid")(dense)

    baseline_func = Model(inputs=func_input, outputs=output)

    baseline_func.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                    metrics = ['accuracy'])
    print(baseline_func.summary()) 
    
    return baseline_func

def load_baseline_model():

    model = Sequential()
    model.add(Rescaling(1./255, input_shape=(256,256,3)))

    model.add(layers.Conv2D(16, kernel_size=10, activation='relu'))
    model.add(layers.MaxPooling2D(3))
    
    model.add(layers.Conv2D(32, kernel_size=8, activation="relu"))
    model.add(layers.MaxPooling2D(3))

    model.add(layers.Conv2D(32, kernel_size=6, activation="relu"))
    model.add(layers.MaxPooling2D(3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(10, activation='sigmoid'))
    
    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    return model

def load_resnet152_func():
    base_model = ResNet152(weights='imagenet',
                           input_shape=(config.IM_SIZE, config.IM_SIZE, 3),
                           include_top=False) 
    base_model.trainable = False
    
    inputs = Input(shape=(config.IM_SIZE, config.IM_SIZE, 3))
    
    x = layers.RandomFlip()(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)

    x = base_model(inputs, training=False)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(500, activation="relu")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Define a bunch of models for transfer learning below

def set_nontrainable_layers(model):
    
    # Set the first layers to be untrainable
    model.trainable = False
        
    return model

def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
    base_model = set_nontrainable_layers(model)
    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(500, activation='relu')
    prediction_layer = layers.Dense(10, activation='sigmoid')
    
    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer,
        prediction_layer
    ])

    return model

def build_model(model, X_train, X_val, X_test):
    """This function will take a string input for the model to be built
    Args: pass in desired model as a string and X_train, X_val and X_test
        - baseline, vgg16, vgg19, resnet50, resnet152, resnet152v2, inceptionresnetv2, inceptionv3, xception
        - X_train, X_val, X_test
        E.g. model, model_name, X_test, history = build_model("vgg16", X_train, X_val, X_test)
    Returns:
        Model object
        Model name as string
        X_test preprocessed for testing
        History of accuracy and loss
    """
    model_name = str(model)

    if model == "baseline":
        model = load_baseline_model()
        return model, model_name, X_train, X_val, X_test
        
    if model == "vgg16":
        model = VGG16(weights="imagenet", include_top=False, input_shape=(config.IM_SIZE, config.IM_SIZE, 3), classes=10)
        model = add_last_layers(model)

        vgg16_X_train = preproc_vgg16(X_train) 
        vgg16_X_val = preproc_vgg16(X_val)
        vgg16_X_test = preproc_vgg16(X_test)
        return model, model_name, vgg16_X_train, vgg16_X_val, vgg16_X_test

    if model == "vgg19":
        model = VGG19(weights="imagenet", include_top=False, input_shape=(config.IM_SIZE, config.IM_SIZE, 3), classes=10)
        model = add_last_layers(model)

        vgg19_X_train = preproc_vgg19(X_train) 
        vgg19_X_val = preproc_vgg19(X_val)
        vgg19_X_test = preproc_vgg19(X_test)
        return model, model_name, vgg19_X_train, vgg19_X_val, vgg19_X_test
        
    if model == "resnet50":
        model = ResNet50(weights="imagenet", include_top=False, input_shape=(config.IM_SIZE, config.IM_SIZE, 3), classes=10)
        model = add_last_layers(model)

        resnet50_X_train = preproc_resnet(X_train) 
        resnet50_X_val = preproc_resnet(X_val)
        resnet50_X_test = preproc_resnet(X_test)
        
        return model, model_name, resnet50_X_train, resnet50_X_val, resnet50_X_test
        
    if model == "resnet152":
        model = ResNet152(weights="imagenet", include_top=False, input_shape=(config.IM_SIZE, config.IM_SIZE, 3), classes=config.NUM_CLASSES)
        model = add_last_layers(model)
        
        resnet50_X_train = preproc_resnet(X_train) 
        resnet50_X_val = preproc_resnet(X_val)
        resnet50_X_test = preproc_resnet(X_test)
        
        return model, model_name, resnet50_X_train, resnet50_X_val, resnet50_X_test
        
    if model == "resnet152v2":
        model = ResNet152V2(weights="imagenet", include_top=False, input_shape=(config.IM_SIZE, config.IM_SIZE, 3), classes=10)
        model = add_last_layers(model)
        
        return model, model_name, X_train, X_val, X_test

    if model == "inceptionv2":
        model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(config.IM_SIZE, config.IM_SIZE, 3), classes=10)
        model = add_last_layers(model)

        inceptionv2_X_train = preproc_inceptionv2(X_train) 
        inceptionv2_X_val = preproc_inceptionv2(X_val)
        inceptionv2_X_test = preproc_inceptionv2(X_test)
        
        return model, model_name, inceptionv2_X_train, inceptionv2_X_val, inceptionv2_X_test
        
    if model == "inceptionv3":
        model = InceptionV3(weights="imagenet", include_top=False, input_shape=(config.IM_SIZE, config.IM_SIZE, 3), classes=10)
        model = add_last_layers(model)

        inceptionv3_X_train = preproc_inceptionv3(X_train) 
        inceptionv3_X_val = preproc_inceptionv3(X_val)
        inceptionv3_X_test = preproc_inceptionv3(X_test)
        
        return model, model_name, inceptionv3_X_train, inceptionv3_X_val, inceptionv3_X_test
        
    if model == "xception":
        model = Xception(weights="imagenet", include_top=False, input_shape=(config.IM_SIZE, config.IM_SIZE, 3), classes=10)
        model = add_last_layers(model)

        xception_X_train = preproc_xception(X_train) 
        xception_X_val = preproc_xception(X_val)
        xception_X_test = preproc_xception(X_test)
        
        return model, model_name, xception_X_train, xception_X_val, xception_X_test   

    if model == "new_resnet152":
        model = ResNet152(weights="imagenet", include_top=False, input_shape=(config.IM_SIZE, config.IM_SIZE, 3), classes=config.NUM_CLASSES)
        model = add_last_layers(model)
        
        resnet50_X_train = preproc_resnet(X_train) 
        resnet50_X_val = preproc_resnet(X_val)
        resnet50_X_test = preproc_resnet(X_test)                                               

def fit_model(model, model_name, X_train, X_val, X_test):
    print(f'Model {model_name} loaded')
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    model.summary()

    es = EarlyStopping(monitor = 'val_accuracy', 
                   mode = 'max', 
                   patience = 5, 
                   verbose = 1, 
                   restore_best_weights = True)
    
    history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val), 
                    epochs=25, 
                    batch_size=32, 
                    callbacks=[es])
    
    if os.path.exists(f"../raw_data/models/{model_name}/"):
        model.save(f"../raw_data/models/{model_name}")
    else:
        os.makedirs(f"../raw_data/models/{model_name}")
        print(f"The directory for {model_name} is created.")
  
    model.save(f"../raw_data/models/{model_name}")
    print(f"Model {model_name} saved.")
    
    return model, model_name, X_test, history

# Utility functions for generating evaluation reports below

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(class_label)
    
def evaluate(model, model_name, X_test, y_test):
    """
        E.g. model, y_pred = evaluate(model, model_name, X_test)
    """
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
    
def generate_reports(model, model_name, y_pred, y_test):
    """
    Args:
        e.g. generate_reports(model, model_name, y_pred)
    """
    thresholds = [0.5, np.mean(y_pred), np.median(y_pred)]
    
    for i in thresholds:       
        print(f"\n\nWhen using {i} as threshold\n\n")
        predictions = y_pred > i
        report = classification_report(y_test, predictions, output_dict=True, target_names=config.CLASSES)
        classification_df = pd.DataFrame(list(report.items()),columns = ['class','scores']) 
        classification_df.to_csv(f"../raw_data/models/{model_name}/classification_report_{i}.csv")
        predictions_df = pd.DataFrame(predictions)
        predictions_df.columns = config.CLASSES
        print(predictions_df.sum())
        y_true = np.array(y_test)
        mcm = multilabel_confusion_matrix(y_true, predictions) 
        fig, ax = plt.subplots(3, 4, figsize=(12, 7))
        for axes, cfs_matrix, label in zip(ax.flatten(), mcm, config.CLASSES):
            print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
        fig.tight_layout()
        plt.savefig(f'../raw_data/models/{model_name}/confusion_matrix_{i}.png')
        plt.show()
        
    return f"Reports generated and saved to ../raw_data/models/{model_name}"

if __name__ == "__main__":
    """Prepare to load X and y after undersampling
    Run these models: baseline, VGG16, VGG19, ResNet50, ResNet152, ResNet152V2, InceptionV3, InceptionResNetV2, Xception
    """
    imgs, df = utils.init()
    df_sample = preproc.sample(df)
    X, y = utils.load_data(df_sample)
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
    
    print("\n\nRun baseline model\n\n")
    
    model, model_name, X_train, X_val, X_test = build_model("baseline", X_train, X_val, X_test)
    print(f"the shape of X_test after build_model is {X_test.shape}")
    print(f"the shape of y_test after build_model is {y_test.shape}")

    model, model_name, X_test, history = fit_model(model, model_name, X_train, X_val, X_test)
    print(f"the shape of X_test after fit_model is {X_test.shape}")
    print(f"the shape of y_test after fit_model is {y_test.shape}")
    model, y_pred = evaluate(model, model_name, X_test, y_test)
    generate_reports(model, model_name, y_pred)
    
    print("\n\nRun VGG16 model\n\n")
    
    model, model_name, vgg16_X_train, vgg16_X_val, vgg16X_test = build_model("vgg16", X_train, X_val, X_test)
    model, model_name, vgg16_X_test, history = fit_model(model, model_name, vgg16_X_train, vgg16_X_val, vgg16X_test)
    model, y_pred = evaluate(model, model_name, vgg16_X_test, y_test)
    generate_reports(model, model_name, y_pred)

    print("\n\nRun VGG19 model\n\n")
    
    model, model_name, vgg19_X_train, vgg19_X_val, vgg19_X_test = build_model("vgg19", X_train, X_val, X_test)
    model, model_name, vgg19_X_test, history = fit_model(model, model_name, vgg19_X_train, vgg19_X_val, vgg19_X_test)
    model, y_pred = evaluate(model, model_name, vgg19_X_test, y_test)
    generate_reports(model, model_name, y_pred)
    
    print("\n\nRun ResNet50 model\n\n")
    
    model, model_name, resnet50_X_train, resnet50_X_val, resnet50_X_test = build_model("resnet50", X_train, X_val, X_test)
    model, model_name, resnet50_X_test, history = fit_model(model, model_name, resnet50_X_train, resnet50_X_val, resnet50_X_test)
    model, y_pred = evaluate(model, model_name, resnet50_X_test, y_test)
    generate_reports(model, model_name, y_pred)
    
    print("\n\nRun ResNet152 model\n\n")
    
    model, model_name, resnet50_X_train, resnet50_X_val, resnet50_X_test = build_model("resnet152", resnet50_X_train, resnet50_X_val, resnet50_X_test)
    model, model_name, resnet50_X_test, history = fit_model(model, model_name, resnet50_X_train, resnet50_X_val, resnet50_X_test)
    model, y_pred = evaluate(model, model_name, resnet50_X_test, y_test)
    generate_reports(model, model_name, y_pred)

    print("\n\nRun ResNet152V2 model\n\n")
    
    model, model_name, resnet50_X_train, resnet50_X_val, resnet50_X_test = build_model("resnet152v2", resnet50_X_train, resnet50_X_val, resnet50_X_test)
    model, model_name, resnet50_X_test, history = fit_model(model, model_name, resnet50_X_train, resnet50_X_val, resnet50_X_test)
    model, y_pred = evaluate(model, model_name, resnet50_X_test, y_test)
    generate_reports(model, model_name, y_pred)

    print("\n\nRun InceptionResNetV2 model\n\n")
    
    model, model_name, inceptionv2_X_train, inceptionv2_X_val, inceptionv2_X_test = build_model("inceptionv2", X_train, X_val, X_test)
    model, model_name, inceptionv2_X_test, history = fit_model(model, model_name, inceptionv2_X_train, inceptionv2_X_val, inceptionv2_X_test)
    model, y_pred = evaluate(model, model_name, inceptionv2_X_test, y_test)
    generate_reports(model, model_name, y_pred)

    print("\n\nRun InceptionV3 model\n\n")
    
    model, model_name, inceptionv3_X_train, inceptionv3_X_val, inceptionv3_X_test = build_model("inceptionv3", X_train, X_val, X_test)
    model, model_name, inceptionv3_X_test, history = fit_model(model, model_name, inceptionv3_X_train, inceptionv3_X_val, inceptionv3_X_test)
    model, y_pred = evaluate(model, model_name, inceptionv3_X_test, y_test)
    generate_reports(model, model_name, y_pred)

    print("\n\nRun Xception model\n\n")
    
    model, model_name, xception_X_train, xception_X_val, xception_X_test = build_model("xception", X_train, X_val, X_test)
    model, model_name, xception_X_test, history = fit_model(model, model_name, xception_X_train, xception_X_val, xception_X_test)
    model, y_pred = evaluate(model, model_name, xception_X_test, y_test)
    generate_reports(model, model_name, y_pred)