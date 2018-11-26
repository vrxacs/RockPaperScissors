import tensorflow as tf
import numpy as np
from math import floor

# Making sure experiments are reproducible
random_state = 42
np.random.seed(random_state)
tf.set_random_seed(random_state)


def get_data():
    # Load file
    datafile = 'data/data_procc.csv'
    dataset = np.loadtxt(datafile, delimiter=',')
    # Ensure random order of samples
    np.random.shuffle(dataset)
    print("Dataset shape: ", dataset.shape)

    X = dataset[:, 1:16]
    Y = []
    for i in dataset[:, 0]:
        tmp = [0] * 4
        tmp[int(i)] = 1
        Y.append(tmp)
    Y = np.array(Y)
    return X, Y


def split_data(train_size, X, Y):
    train_cnt = floor(X.shape[0] * train_size)
    x_train = X[0:train_cnt]
    y_train = Y[0:train_cnt]
    x_test = X[train_cnt:]
    y_test = Y[train_cnt:]
    return x_train, y_train, x_test, y_test


def multilayer_perceptron(x, keep_prob, n_input, n_hidden, n_output):

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


if __name__ == '__main__':
    X, Y = get_data()

    # Split into training and test set
    train_size = 0.9
    x_train, y_train, x_test, y_test = split_data(train_size, X, Y)

    n_hidden = 30
    n_input = 15
    n_output = 4

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])
    keep_prob = tf.placeholder("float")

    predictions = multilayer_perceptron(x, keep_prob, n_input, n_hidden, n_output)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    training_epochs = 5000
    display_step = 1000
    batch_size = 32

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(x_train) / batch_size)
            x_batches = np.array_split(x_train, total_batch)
            y_batches = np.array_split(y_train, total_batch)
            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                _, c = sess.run([optimizer, cost],
                                feed_dict={
                                    x: batch_x,
                                    y: batch_y,
                                    keep_prob: 0.8
                                })
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0}))