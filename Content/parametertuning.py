import keras_tuner as kt
import numpy as np
import pandas as pd
from tensorflow import keras

# tuning parameters
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
KERNEL_SIZE = (5, 5)
POOL_SIZE = (2, 2)
RELU = 'relu'
SOFTMAX = 'softmax'
PADDING = 'same'
FILTER_MIN = 32
FILTER_MAX = 288
FILTER_STEP = 32
FLOAT_MIN = 0.0
FLOAT_MAX = 0.7
FLOAT_STEP = 0.1
INT_UNITS = [int(x) for x in np.arange(FILTER_MIN, FILTER_MAX, FILTER_STEP)]
FLOAT_UNITS = [float(y) for y in np.arange(FLOAT_MIN, FLOAT_MAX, FLOAT_STEP)]
LEARNING_RATES = [1e-2, 1e-3, 1e-4]
METRICS = [keras.losses.CategoricalCrossentropy(), keras.metrics.CategoricalAccuracy(), keras.metrics.Precision()]

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

def model_cnn(hp):
    """
    Architecture: Convolutional neural network
    """
    model = keras.Sequential()
    # Input layer
    model.add(keras.Input(shape=INPUT_SHAPE))
    filter_names = ['conv_filters1', 'conv_filters2']
    # Create 2 sets of Convolutional and MaxPooling layers
    for i in range(len(filter_names)):
        # Choice of units for the Convolutional layers
        hp_filters = hp.Choice(filter_names[i],
                               values=INT_UNITS)
        model.add(keras.layers.Conv2D(filters=hp_filters,
                                      kernel_size=KERNEL_SIZE,
                                      activation=RELU,
                                      padding=PADDING))
        model.add(keras.layers.MaxPool2D(pool_size=POOL_SIZE))
        if i == 0:
            # Choice of dropout rate
            hp_dropout_rate = hp.Choice('dropout_rate',
                                        values=FLOAT_UNITS)
            # Dropout layer
            model.add(keras.layers.Dropout(hp_dropout_rate))
    # Flatten to 1 dimension
    model.add(keras.layers.Flatten())
    # Choice of units for the fully connected layer
    hp_units = hp.Choice('connected_units',
                         values=INT_UNITS)
    # Fully connected layer
    model.add(keras.layers.Dense(units=hp_units,
                                 activation=RELU))
    # Output layer
    model.add(keras.layers.Dense(units=NUM_CLASSES,
                                 activation=SOFTMAX))
    # Choice of learning rate for the optimiser
    hp_learning_rate = hp.Choice('learning_rate',
                                 values=LEARNING_RATES)
    # Compile the model with the optimiser and the metrics that will be used to train the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=METRICS[0],
                  metrics=[METRICS[1], METRICS[2]])
    return model

def model_fcnn(hp):
    """
    Architecture: Fully connected neural network
    """
    model = keras.Sequential()
    # Input layer
    model.add(keras.Input(shape=INPUT_SHAPE))
    # Choice of units for fully connected layer
    hp_units = hp.Choice('connected_units1',
                         values=INT_UNITS)
    # Fully connected layer
    model.add(keras.layers.Dense(hp_units,
                                 activation=RELU))
    # Flatten layer
    model.add(keras.layers.Flatten())
    connected_names = ['connected_units2', 'connected_units3', 'connected_units4']
    # 3 Fully connected hidden layers
    for i in range(len(connected_names)):
        # Choice of units for fully connected layers
        hp_units2 = hp.Choice(connected_names[i],
                              values=INT_UNITS)
        model.add(keras.layers.Dense(hp_units2,
                                     activation=RELU))
    # Choice of dropout rate
    hp_dropout = hp.Choice('dropout_rate',
                           values=FLOAT_UNITS)
    # Dropout layer
    model.add(keras.layers.Dropout(hp_dropout))
    # Output layer
    model.add(keras.layers.Dense(NUM_CLASSES,
                                 activation=SOFTMAX))
    # Choice of learning rate for the optimiser
    hp_learning_rate = hp.Choice('learning_rate',
                                 values=LEARNING_RATES)
    # Compile the model with the optimiser and the metrics that will be used to train the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=METRICS[0],
                  metrics=[METRICS[1], METRICS[2]])
    return model


if __name__ == "__main__":
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               patience=3)
    names = [('fcnn_arch_testing', 'results/fcnn'), ('cnn_arch_testing', 'results/cnn')]
    models = (model_fcnn, model_cnn)

    for n in range(len(names)):
        tuner = kt.BayesianOptimization(models[n],
                                        objective='val_loss',
                                        max_trials=20,
                                        directory=names[n][0])
        tuner.search(x_train,
                     y_train,
                     validation_split=0.2,
                     batch_size=300,
                     epochs=15,
                     callbacks=[stop_early])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        if n != 0:
            data = {
                'first': f"{best_hps.get('conv_filters1')}",
                'second': f"{best_hps.get('conv_filters2')}",
                'connected': f"{best_hps.get('connected_units')}",
                'dropout': f"{best_hps.get('dropout_rate')}",
                'learning_rate': f"{best_hps.get('learning_rate')}"
            }
        else:
            data = {
                'first': f"{best_hps.get('connected_units1')}",
                'second': f"{best_hps.get('connected_units2')}",
                'third': f"{best_hps.get('connected_units3')}",
                'fourth': f"{best_hps.get('connected_units4')}",
                'dropout': f"{best_hps.get('dropout_rate')}",
                'learning_rate': f"{best_hps.get('learning_rate')}"
            }
        df = pd.DataFrame(data=data, index=['parameters'])
        df.to_csv(names[n][1])
