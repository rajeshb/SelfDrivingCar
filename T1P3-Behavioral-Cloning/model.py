import numpy as np
import tensorflow as tf

from settings import Settings
from generator import DataGenerator
from dataloader import DataLoader
from modelarchitecture import DataModel

# set the random seed
np.random.seed(450)

# command line flags
flags = tf.app.flags
FLAGS = flags.FLAGS

# define command line flags & default values
flags.DEFINE_string('data_path', 'data/', 'Path to training data CSV file')
flags.DEFINE_string('data_file', 'driving_log.csv', 'Training data CSV file name')
flags.DEFINE_string('images_path', 'IMG/', 'The directory of the image files')
flags.DEFINE_integer('batch_size', 512, 'Batch size')
flags.DEFINE_integer('epochs', 20, 'The number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('epoch_sample_size', 25000, 'Number of samples per epoch')

# main 
def main(_):

    # initialize settings
    settings = Settings(
        FLAGS.data_path,
        FLAGS.data_file,
        FLAGS.images_path,
        FLAGS.batch_size, 
        FLAGS.epochs,
        FLAGS.learning_rate,
        FLAGS.epoch_sample_size)

    # print settings
    print(settings)
    
    # load, consolidate and augment data
    data_loader = DataLoader(settings)

    # data generator for train and test
    data_generator = DataGenerator(settings, data_loader)

    # setup & fit model
    DataModel(settings).fit(data_generator)

    print("Finally, done!")
    
# main function
if __name__ == '__main__':
    tf.app.run()