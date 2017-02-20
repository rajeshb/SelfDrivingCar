import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import settings
import utils
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

# data generator class
class DataGenerator:
    def __init__(self, settings, dataloader):
        self.settings = settings
        self.dataloader = dataloader
        self.shuffle_split()

    def shuffle_split(self, test_split_size=0.2):

        # Random threhold from 3 - 10
        from random import randint
        self.dataloader.filter_zero_steering(randint(3,10))

        # get data
        X, y = self.dataloader.get_data()

        # Shuffle data
        X_input, y_input = shuffle(X, y)

        # Split data
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_input, y_input, test_size=test_split_size)
        print("Train samples count : {} Valid samples count : {}".format(len(self.y_train), len(self.y_valid)))

    def get_test_samples_count(self):
        return len(self.y_train)

    def get_valid_samples_count(self):
        return len(self.y_valid)

    def data_gen(self, X, y, data_augmentation = False):
        while True:
            X_output, y_output = ([],[])
            # Shuffle data
            X_input, y_input = shuffle(X, y)
            for i in range(len(y_input)):

                image_file = self.settings.data_path + self.settings.images_path + X_input[i].strip()
                (img, steering) = utils.get_proceesed_data(image_file, y_input[i], data_augmentation)
                X_output.append(img)
                y_output.append(steering)

                if len(y_output) == self.settings.batch_size:
                    break

            # yield the batch
            yield (np.array(X_output), np.array(y_output))

    def train_data_generator(self):
        return self.data_gen(self.X_train, self.y_train, data_augmentation = True)

    def valid_data_generator(self):
        return self.data_gen(self.X_valid, self.y_valid, data_augmentation = False)
