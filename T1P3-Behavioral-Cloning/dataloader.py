import numpy as np
import utils
import settings
import pandas as pd

# data loader class
class DataLoader():

    def __init__(self, settings):
        self.settings = settings

    def get_csv_data(self):
        col_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        data_file = self.settings.data_path + self.settings.data_file
        return pd.read_csv(data_file, names=col_names, skiprows=1)

    def get_data(self, correction = 0.20):

        # load data
        data = self.get_csv_data()

        center = [utils.get_image_file_name(file_path) for file_path in data['center'].values] 
        left = [utils.get_image_file_name(file_path) for file_path in data['left'].values] 
        right = [utils.get_image_file_name(file_path) for file_path in data['right'].values] 
        steering = data['steering'].values

        X = np.concatenate((center, left, right), axis=0)
        y = np.concatenate((steering, steering + correction, steering - correction), axis=0)

        print("X (number of samples) : {}".format(len(X)))
        return X, y
