import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import settings
import utils
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

# data generator class
class DataGenerator:
    def __init__(self, settings, X, y):
        self.settings = settings
        self.X = X
        self.y = y

        print("Samples (before stratified augmentation) : {}".format(len(self.y)))
        single_tokens = [k for k, v in Counter(y).items() if v == 1 ]
        
        if len(single_tokens) > 0:

            single_token_indexes = [i for i,x in enumerate(y) if x in single_tokens]
            print("single_tokens : {} single_tokens_indexes : {}".format(len(single_tokens), len(single_token_indexes)))

            X_tmp = []
            y_tmp = []

            for i in range(len(single_token_indexes)):
                X_tmp.append(X[single_token_indexes[i]])
                y_tmp.append(y[single_token_indexes[i]])

            ## Duplicate for stratified sample split
            self.X = np.concatenate((X,X_tmp), axis=0)
            self.y = np.concatenate((y,y_tmp), axis=0)

        print("Samples (after stratified augmentation) : {}".format(len(self.y)))

        self.shuffle_split()
        print("Train samples count : {} Valid samples count : {}".format(len(self.y_train), len(self.y_valid)))

    def shuffle_split(self, test_split_size=0.2):
        # Shuffle data
        #X_input, y_input = shuffle(self.X, self.y)

        # Split data
        #self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_input, y_input, test_size=test_split_size)

        X_input, y_input = shuffle(self.X, self.y)

        if len(y_input) * test_split_size < len(Counter(y_input)):
            test_split_size = len(Counter(y_input)) / len(y_input)
            print("Override test split to {}".format(test_split_size))

        # split train and validation data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split_size)
        for train_index, valid_index in sss.split(self.X, self.y):
            self.X_train, self.X_valid = self.X[train_index], self.X[valid_index]
            self.y_train, self.y_valid = self.y[train_index], self.y[valid_index]


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

                if data_augmentation == True and abs(steering) > 0.05:
                    img = utils.flip_image(utils.load_image(image_file))
                    steering = steering * -1.0
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
