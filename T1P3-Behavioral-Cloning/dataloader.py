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
        #self.filter_zero_steering(threshold)
        self.normalize_distribution(self.data)

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
        return self.filtered_data

    def normalize_distribution(self, data):
        num_bins = 41
        angles = data['steering'].values
        avg_samples_per_bin = len(angles)/num_bins
        hist, bins = np.histogram(angles, num_bins)
        keep_probs = []
        target = avg_samples_per_bin * .5
        for i in range(num_bins):
            if hist[i] < target:
                keep_probs.append(1.)
            else:
                keep_probs.append(1./(hist[i]/target))
        remove_list = []
        for i in range(len(angles)):
            for j in range(num_bins):
                if angles[i] > bins[j] and angles[i] <= bins[j+1]:
                    if np.random.rand() > keep_probs[j]:
                        remove_list.append(i)
        self.normalized_data = data.drop(data.index[remove_list])
        print("Samples before normalization : {} and after : {}".format(len(data), len(self.normalized_data)))
        return self.normalized_data

    def get_data(self, correction = 0.12):

        center = [utils.get_image_file_name(file_path) for file_path in self.normalized_data['center'].values] 
        left = [utils.get_image_file_name(file_path) for file_path in self.normalized_data['left'].values] 
        right = [utils.get_image_file_name(file_path) for file_path in self.normalized_data['right'].values] 
        steering = self.normalized_data['steering'].values

        X = np.concatenate((center, left, right), axis=0)
        y = np.concatenate((steering, steering + correction, steering - correction), axis=0)

        print("Samples (after merging left, right camera images) : {}".format(len(X)))
        return X, y
