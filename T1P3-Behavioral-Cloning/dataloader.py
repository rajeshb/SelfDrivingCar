import numpy as np
import utils
import settings
import pandas as pd
from collections import deque

# data loader class
class DataLoader():

    def __init__(self, settings, threshold=5):
        self.settings = settings
        self.data = self.get_csv_data()
        # Filter excess straight line driving to reduce its influence on the model
        self.filter_zero_steering(threshold)

    def get_csv_data(self):
        col_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        data_file = self.settings.data_path + self.settings.data_file
        return pd.read_csv(data_file, names=col_names, skiprows=1)

    def filter_zero_steering(self, threshold=5):
        history = deque([])
        indexes = deque([])
        drop_rows = []

        for idx, row in self.data.iterrows():
            steering = row['steering']

            history.append(steering)
            indexes.append(idx)
            if len(history) > threshold:
                history.popleft()
                indexes.popleft()

            if history.count(0.0) == threshold:
                drop_rows.extend(list(indexes)[1:-1])
                history.clear()
                indexes.clear()
                history.append(steering)
                indexes.append(idx)

        self.filtered_data = self.data.drop(self.data.index[drop_rows])
        print("Samples before (removing zero steering) : {} and after : {} threshold : {}".format(len(self.data), len(self.filtered_data), threshold))
        return

    def get_data(self, correction = 0.15):

        center = [utils.get_image_file_name(file_path) for file_path in self.filtered_data['center'].values] 
        left = [utils.get_image_file_name(file_path) for file_path in self.filtered_data['left'].values] 
        right = [utils.get_image_file_name(file_path) for file_path in self.filtered_data['right'].values] 
        steering = self.filtered_data['steering'].values

        X = np.concatenate((center, left, right), axis=0)
        y = np.concatenate((steering, steering + correction, steering - correction), axis=0)

        print("Samples (after merging left, right camera images) : {}".format(len(X)))
        return X, y
