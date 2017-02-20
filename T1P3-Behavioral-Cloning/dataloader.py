import numpy as np
import utils
import settings
import pandas as pd
from collections import deque

# data loader class
class DataLoader():

    def __init__(self, settings):
        self.settings = settings
        self.data = self.get_csv_data()

    def get_csv_data(self):
        col_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        data_file = self.settings.data_path + self.settings.data_file
        return pd.read_csv(data_file, names=col_names, skiprows=1)

    def filter_zero_steering(self, threshold=3):
        history = deque([])
        drop_rows = []

        for idx, row in self.data.iterrows():
            steering = row['steering']

        history.append(steering)
        if len(history) > threshold:
            history.popleft()

        if history.count(0.0) == threshold:
            drop_rows.append(idx)

    return self.data.drop(self.data.index[drop_rows])


    def get_data(self, correction = 0.20):

        # Filter excess straight line driving to reduce its influence on the model
        filtered_data = self.filter_zero_steering(self, threshold=3)

        center = [utils.get_image_file_name(file_path) for file_path in filtered_data['center'].values] 
        left = [utils.get_image_file_name(file_path) for file_path in filtered_data['left'].values] 
        right = [utils.get_image_file_name(file_path) for file_path in filtered_data['right'].values] 
        steering = filtered_data['steering'].values

        X = np.concatenate((center, left, right), axis=0)
        y = np.concatenate((steering, steering + correction, steering - correction), axis=0)

        print("X (number of samples) : {}".format(len(X)))
        return X, y
