import math

# settings 
class Settings():
    def __init__(self, data_path, data_file, images_path, batch_size, epochs, learning_rate, epoch_sample_size):
        self.data_path = data_path
        self.data_file = data_file
        self.images_path = images_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epoch_sample_size = epoch_sample_size

        # match samples per epoch (spe) to the multiples of batch size
        self.spe = math.ceil(epoch_sample_size/batch_size) * batch_size

    def __repr__(self):
        return "Settings()"

    def __str__(self):
        output = "SETTINGS\n"
        output += "========\n"
        output += "data_path: {}\n".format(self.data_path)
        output += "data_file: {}\n".format(self.data_file)
        output += "images_path: {}\n".format(self.images_path)
        output += "batch_size: {}\n".format(self.batch_size)
        output += "epochs: {}\n".format(self.epochs)
        output += "learning_rate: {}\n".format(self.learning_rate)
        output += "epoch_sample_size: {}\n".format(self.epoch_sample_size)
        output += "samples per epoch: {}\n".format(self.spe)
        return output
