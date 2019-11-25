import tensorflow as tf
import numpy as np
import pickle
import time
from PIL import Image
import tensorflow.compat.v1 as tf
from driving_NN_tf import NeuralNetwork
from generate_data_files import extract_data

tf.disable_v2_behavior()

def test_model(test_data, predictions, target):
    x_test, y_test = test_data

    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    y_test = np.asarray(y_test)
    y_test = y_test.reshape(-1, y_test.shape[1])
    x_test = np.asarray(x_test)
    x_test = x_test.reshape(-1, x_test.shape[1])

    print("Accuracy:", accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0}))




if __name__ == "__main__":
    model_name = "models/model-20191124-190659-10"
    with tf.Session() as sess:
        ##extract parameters
        saver = tf.train.import_meta_graph("{0}.meta".format(model_name))
        saver.restore(sess, './{0}'.format(model_name))
        graph = tf.get_default_graph()
        weights = {
            'h1': graph.get_tensor_by_name('h1:0'),
            'out': graph.get_tensor_by_name('w_out:0')
        }
        biases = {
            'b1': graph.get_tensor_by_name('b1:0'),
            'out': graph.get_tensor_by_name('b_out:0')
        }

        ##init network
        hidden_n = 300
        input_shape = (160*60,1)
        out_layer_length = 4
        frozen_nn = NeuralNetwork(hidden_n, input_shape[0], out_layer_length)
        frozen_nn.weights= weights
        frozen_nn.biases = biases
        _, _, x_valid, y_valid = frozen_nn.data_wrapper("./warped", input_shape, out_layer_length)
        test_data = (x_valid, y_valid)
        x = tf.placeholder("float", shape=[None, frozen_nn.n_input], name='x')
        y = tf.placeholder("float", shape=[None, frozen_nn.n_classes], name='y')
        keep_prob = tf.placeholder("float")
        predictions = frozen_nn.feedForward(x, keep_prob)

        ##run inference
        target_img_path = "test_images"
        input = np.asarray(extract_data(target_img_path)[0][0])
        input= input[np.newaxis]
        print(input.shape)
        print(sess.run(tf.argmax(predictions,1), {x: input, keep_prob: 1.0}))
