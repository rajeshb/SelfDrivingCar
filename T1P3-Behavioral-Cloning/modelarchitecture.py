from keras.models import Sequential
from keras.layers import Lambda, Dense, Dropout, Flatten, Conv2D
from keras.layers.convolutional import Cropping2D
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

from keras.optimizers import Adam

import generator
import settings

class EpochCallback(Callback):

    def __init__(self, generator):
        self.generator = generator

    def on_epoch_begin(self, epoch, logs={}):
        #print("Shuffle and Split train and valid data for epoch {}".format(epoch+1))
        self.generator.shuffle_split()

class DataModel():

    def __init__(self, settings):
        self.settings = settings
        self.setup_NVIDIA_model(learning_rate=settings.learning_rate)

    def setup_NVIDIA_model(self, input_shape=(160,320,3), keep_prob=0.5, learning_rate=0.0001):
    
        # initialize keras model
        self.model = Sequential()
    
        # normalization
        self.model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=input_shape))

        # cropping images to match 66x200 input for NVIDIA model
        self.model.add(Cropping2D(cropping=((70,24), (60,60))))
    
        # three 5x5 convolution layers with 2x2 strides
        self.model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2,2)))
        self.model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2,2)))
        self.model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2,2)))

        # dropout
        self.model.add(Dropout(keep_prob))
      
        # two 3x3 convolution layers with 1x1 strides
        self.model.add(Conv2D(64, 3, 3, activation='elu'))
        self.model.add(Conv2D(64, 3, 3, activation='elu'))

        # dropout
        self.model.add(Dropout(keep_prob))
  
        # flatten
        self.model.add(Flatten())
    
        # fully connected layers
        self.model.add(Dense(100, activation='elu'))
        self.model.add(Dense(50, activation='elu'))
        self.model.add(Dense(10, activation='elu'))
        self.model.add(Dense(1))
    
        # adam optimizer with learning rate = 0.001
        self.model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    
        # print model summary
        self.model.summary()

    def save(self, model_json_file='model.json', model_weights_file='model.h5'):
        # save model json
        json_string = self.model.to_json()
        with open(model_json_file, 'w') as f:
            f.write(json_string)

        # save model weights
        self.model.save_weights(model_weights_file, overwrite=True)

    def get(self):
        return self.model

    def fit(self, generator):

        # shuffle and split training and validation data for every epoch, at the beginning of each epoch
        epoch_begin_callback = EpochCallback(generator)
    
        # save both weights and model for the best
        model_checkpoint = ModelCheckpoint(filepath = 'model-best.h5', verbose = 0, save_best_only=True, monitor='val_loss')

        # stop training when validation loss fails to decrease
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

        # fit the model
        self.model.fit_generator(
            generator.train_data_generator(),
            validation_data = generator.valid_data_generator(),
            nb_val_samples = generator.get_valid_samples_count(),
            samples_per_epoch = self.settings.spe,
            nb_epoch = self.settings.epochs,
            verbose = 1,
            callbacks = [model_checkpoint, early_stopping, epoch_begin_callback])
