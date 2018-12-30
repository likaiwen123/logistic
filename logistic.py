#!/usr/bin/python3

import numpy as np
import tensorflow as tf

filelist = ["train.csv", "test.csv"]

dim = 28
length = dim * dim

feature_names = []
record_defaults = []

for i in range(length + 1):
    feature_names.append('pixel' + str(i))
    record_defaults.append([0])


def decode_csv(line):
    parsed_line = tf.decode_csv(line, record_defaults)
    label = parsed_line[0]
    del parsed_line[0]
    features = tf.stack(parsed_line)
    batch_to_return = features, label
    return batch_to_return


filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(1).map(decode_csv))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(1000)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

training_filenames = ["train.csv"]
test_filenames = ["test.csv"]

train_images = []
train_labels = []
test_images = []
test_labels = []

# Parameters
learning_rate = 0.001
training_epochs = 25
# batch_size = 100
display_step = 1
train_batch = 35

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

# Minimize error using cross entropy
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    number = 0
    batch = 0
    while True:
        try:
            features, label = sess.run(next_element)
            labels = np.zeros((len(label), 10))
            for i in range(len(label)):
                labels[i][label[i]] = 1

            if batch < train_batch:
                train_images.append(features)
                train_labels.append(labels)
            else:
                test_images.extend(features)
                test_labels.extend(labels)
            batch += 1
        except tf.errors.OutOfRangeError:
            print("Out of range error triggered.")
            break

    # training and validating
    epoch = 0
    while epoch < training_epochs:
        avg_cost = 0
        for i in range(train_batch):
            batch_x = train_images[i]
            batch_y = train_labels[i]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / train_batch
        # Display logs per epoch step
        epoch += 1
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(avg_cost))

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: test_images, y: test_labels}))
