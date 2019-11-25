import tensorflow as tf
import numpy as np
import pickle
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class NeuralNetwork:
    def __init__(self, hiddenNeurons, in_len, out_len):
        self.n_hidden_1 = hiddenNeurons
        self.n_input = in_len
        self.n_classes = out_len
        self.weights = {
            'h1': tf.Variable(tf.random.normal([self.n_input, self.n_hidden_1]), name='h1'),
            'out': tf.Variable(tf.random.normal([self.n_hidden_1, self.n_classes]), name='w_out')
        }
        self.biases = {
            'b1': tf.Variable(tf.random.normal([self.n_hidden_1]), name='b1'),
            'out': tf.Variable(tf.random.normal([self.n_classes]), name='b_out')
        }



    def feedForward(self, x, keep_prob):
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        out_layer = tf.matmul(layer_1, self.weights['out']) + self.biases['out']
        return out_layer

    def load_data(self, dataset_path):
        train_data = pickle.load(open(dataset_path+'/train.pkl', 'rb'))
        valid_data = pickle.load(open(dataset_path+'/valid.pkl', 'rb'))

        return train_data, valid_data

    ## returns a vector of length 'out_len' where the index y is 1 and everything else is 0
    def vectorize(self, y, out_len):
        arr = np.zeros((out_len,1))
        arr[y] = 1.0
        return arr

    ## formats data so it's more convenient to use in our NN algorithm
    def data_wrapper(self, dataset_path, input_shape, out_len):
        train_data, valid_data = self.load_data(dataset_path)

        ## train_data is a tuple(x,y) with x being the input and y a vectorized output
        train_inputs = [row[0] for row in train_data]
        train_results = [self.vectorize(row[1], out_len) for row in train_data]

        valid_inputs = [row[0] for row in valid_data]
        valid_results = [self.vectorize(row[1], out_len) for row in valid_data]

        return train_inputs, train_results, valid_inputs, valid_results

    def test_model(self, test_data, predictions, target):
        x_test, y_test = test_data

        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(target, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        y_test = np.asarray(y_test)
        y_test = y_test.reshape(-1, y_test.shape[1])
        x_test = np.asarray(x_test)
        x_test = x_test.reshape(-1, x_test.shape[1])

        print("Accuracy:", accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0}))

        print(predictions.eval({x: x_test, keep_prob: 1.0}))


    def calc_error(self, X):
        n_true = 0.0
        for row in X:
            x = row[0]
            t = row[1]
            z, a = self.compute_activations(x)
            if (np.argmax(a[-1]) - t) == 0:
                n_true += 1
        return n_true / len(X)


    def SGD(self, epochs, x_train, y_train, x_valid, y_valid, y, batch_size, optimizer, cost, predictions):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            for epoch in range(epochs):
                avg_cost = 0.0
                total_batch = int(len(x_train) / batch_size)
                x_batches = np.array_split(x_train, total_batch)
                y_batches = np.array_split(y_train, total_batch)
                for i in range(total_batch):
                    batch_x, batch_y = x_batches[i], y_batches[i]
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={
                                        x: batch_x.reshape(-1, batch_x.shape[1]),
                                        y: batch_y.reshape(-1, batch_y.shape[1]),
                                        keep_prob: 0.8
                                    })
                    avg_cost += c / total_batch
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
            print("Optimization Finished!")

            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            y_valid = np.asarray(y_valid)
            y_valid = y_valid.reshape(-1, y_valid.shape[1])
            x_valid = np.asarray(x_valid)
            x_valid = x_valid.reshape(-1, x_valid.shape[1])
            print("Accuracy:", accuracy.eval({x: x_valid, y: y_valid, keep_prob: 1.0}))

            model_name = 'model-' + timestr
            saver = tf.train.Saver()
            saver.save(sess, "./models/{0}".format(model_name), global_step=epochs)


        return


if __name__ == "__main__":
    hidden_n = 300
    input_shape = (160*60,1)
    out_layer_length = 4
    NN = NeuralNetwork(hidden_n, input_shape[0], out_layer_length)
    x_train, y_train, x_valid, y_valid = NN.data_wrapper("./warped",input_shape, out_layer_length)
    training_epochs = 10
    display_step = 1
    batch_size = 32

    x = tf.placeholder("float", shape=[None, NN.n_input], name='x')
    y = tf.placeholder("float", shape=[None, NN.n_classes], name='y')
    keep_prob = tf.placeholder("float")

    predictions = NN.feedForward(x, keep_prob)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)


    NN.SGD(training_epochs, x_train, y_train, x_valid, y_valid, y, batch_size, optimizer, cost, predictions)
