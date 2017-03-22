import pickle
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
import pandas as pd
import numpy as np

training_file = 'train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']

n_classes = np.unique(y_train).size

# Load traffic sign names from CSV file
traffic_sign_names = pd.read_csv('signnames.csv', index_col='ClassId')

# TODO: Split data into training and validation sets.
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=450)
for train_index, test_index in sss.split(X_train, y_train):
     X_train, X_validation = X_train[train_index], X_train[test_index]
     y_train, y_validation = y_train[train_index], y_train[test_index]
     
# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))
y = tf.placeholder(tf.int32, (None), name="LabelData")
one_hot_y = tf.one_hot(y, n_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], n_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(n_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
LEARNING_RATE = 0.001
EPOCHS = 2
BATCH_SIZE = 128

with tf.name_scope('Loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)

with tf.name_scope('Train'):
    optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
    training_operation = optimizer.minimize(loss_operation)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# TODO: Train and evaluate the feature extraction model.
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy, loss = sess.run([accuracy_operation, loss_operation] , 
                                  feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_accuracy / num_examples, total_loss / num_examples

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(EPOCHS):
        num_examples = len(X_train)
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
  
        validation_accuracy, validation_loss = evaluate(X_validation, y_validation)
        print("EPOCH {} Duration {} Validation Accuracy = {:.3f} Loss = {:.3f}".format(epoch+1, time.time() - t0, validation_accuracy, validation_loss))
        
    saver.save(sess, 'AlexNetTrainFeature')
    print("Model saved")