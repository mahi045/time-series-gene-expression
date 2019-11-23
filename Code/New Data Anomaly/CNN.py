import numpy as np
from Util import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os


gene_names, gene_types, data = get_patient_data()
X = data
Y = gene_types
X_train, X_test, labels_train,  labels_test = train_test_split(X,Y, test_size=0.2, random_state=None)


# get validation set
X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train,
                                                test_size=0.1, random_state = None)
y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(labels_test)



batch_size = 100       # Batch size
seq_len = 9          # Number of steps
learning_rate = 0.0001
epochs = 100

n_classes = 1   # 1 ta output value, 0 hoile bad, 1 hoile good
n_channels = 1

graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

with graph.as_default():
    # (batch, 9, 9) --> (batch, 64, 18)
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=18, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

    # (batch, 64, 18) --> (batch, 32, 36)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

    # (batch, 32, 36) --> (batch, 16, 72)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

    # (batch, 16, 72) --> (batch, 8, 144)
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

with graph.as_default():
    # Flatten and add dropout
    flat = tf.reshape(max_pool_4, (-1, 8 * 144))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

    # Predictions
    logits = tf.layers.dense(flat, n_classes)

    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

if (os.path.exists('checkpoints-cnn') == False):
    os.makedirs('checkpoints-cnn')

validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1

    # Loop over epochs
    for e in range(epochs):

        # Loop over batches
        for x, y in get_batches(X_tr, y_tr, batch_size):

            # Feed dictionary
            feed = {inputs_: x, labels_: y, keep_prob_: 0.5, learning_rate_: learning_rate}

            # Loss
            loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict=feed)
            train_acc.append(acc)
            train_loss.append(loss)

            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))

            # Compute validation loss at every 10 iterations
            if (iteration % 10 == 0):
                val_acc_ = []
                val_loss_ = []

                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_: x_v, labels_: y_v, keep_prob_: 1.0}

                    # Loss
                    loss_v, acc_v = sess.run([cost, accuracy], feed_dict=feed)
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)

                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))

            # Iterate
            iteration += 1

    saver.save(sess, "checkpoints-cnn/har.ckpt")



