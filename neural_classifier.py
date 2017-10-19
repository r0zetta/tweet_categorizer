# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from string import printable
import numpy as np
import Queue
import codecs
import random, csv
import sys
import string
import os

trained_model_dir = "trained_model"
training_data_dir = "training_data"
TEST_SET = training_data_dir + "/test_set.csv"
SAVE_PATH = trained_model_dir + '/lstm'
LOGGING_PATH = trained_model_dir + '/log.txt'

TEST_SET_SIZE = 0
test_data = []

emb_alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '
emb_alphabet_extra= ''
DICT = {ch: ix for ix, ch in enumerate(emb_alphabet + emb_alphabet_extra.decode("utf-8"))}
ALPHABET_SIZE = len(DICT)
max_word_length = 30
learning_rate = 0.0001
patience = 10000
EPOCHS = 500
BATCH_SIZE = 1
size = 700
kernels=[1, 2, 3, 4, 5, 6, 7]
kernel_features=[25, 50, 75, 100, 125, 150, 175]
rnn_size=650
dropout=0.0

X = tf.placeholder('float32', shape=[None, None, max_word_length, ALPHABET_SIZE], name='X')
Y = tf.placeholder('float32', shape=[None, 2], name='Y')

def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])
    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b

def linear(input_, output_size, scope=None):
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

def softmax(input_, out_dim, scope=None):
    with tf.variable_scope(scope or 'softmax'):
        W = tf.get_variable('W', [input_.get_shape()[1], out_dim])
        b = tf.get_variable('b', [out_dim])
    return tf.nn.softmax(tf.matmul(input_, W) + b)

def MLP(input_, out_dim, size=128, scope=None):
    assert len(input_.get_shape) == 2, "MLP takes input of dimension 2 only"
    with tf.variable_scope(scope or "MLP"):
        W_h = tf.get_variable("W_hidden", [input_.get_shape()[1], size], dtype='float32')
        b_h = tf.get_variable("b_hidden", [size], dtype='float32')
        W_out = tf.get_variable("W_out", [size, out_dim], dtype='float32')
        b_out = tf.get_variable("b_out", [out_dim], dtype='float32')
    h = tf.nn.relu(tf.matmul(input_, W_h) + b_h)
    out = tf.matmul(h, W_out) + b_out
    return out

def ResBlock(input_, out_dim, size=128, scope=None):
    with tf.variable_scope(scope or "MLP"):
        W_h = tf.get_variable("W_hidden", [input_.get_shape()[1], size], dtype='float32')
        b_h = tf.get_variable("b_hidden", [size], dtype='float32')
        W_h_res = tf.get_variable("W_hidden_res", [input_.get_shape()[1], size], dtype='float32')
        b_h_res = tf.get_variable("b_hidden_res", [size], dtype='float32')

        W_out = tf.get_variable("W_out", [size, out_dim], dtype='float32')
        b_out = tf.get_variable("b_out", [out_dim], dtype='float32')
    h = tf.nn.relu(tf.matmul(input_, W_h) + b_h)
    h_res = tf.nn.relu(tf.matmul(input_, W_h_res) + b_h_res) + h
    out = tf.matmul(h_res, W_out) + b_out
    return out

def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))
            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
            output = t * g + (1. - t) * input_
            input_ = output
    return output

def tdnn(input_, kernels, kernel_features, scope='TDNN'):
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'
    input_ = tf.reshape(input_, [-1, max_word_length, ALPHABET_SIZE])
    input_ = tf.expand_dims(input_, 1)
    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = max_word_length - kernel_size + 1
            conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')
            layers.append(tf.squeeze(pool, [1, 2]))

        if len(kernels) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]
    return output

def create_rnn_cell():
    cell = rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
    if dropout > 0.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1. - dropout)
    return cell


def encode_one_hot(sentence):
    sent = []
    SENT_LENGTH = 0
    encoded_sentence = filter(lambda x: x in (printable  + emb_alphabet_extra), sentence)
    if len(encoded_sentence) > 0:
        for word in word_tokenize(encoded_sentence.decode('utf-8', 'ignore').encode('utf-8')):
            word_encoding = np.zeros(shape=(max_word_length, ALPHABET_SIZE))
            for i, char in enumerate(word):
                try:
                    char_encoding = DICT[char]
                    one_hot = np.zeros(ALPHABET_SIZE)
                    one_hot[char_encoding] = 1
                    word_encoding[i] = one_hot
                except Exception as e:
                    pass
            sent.append(np.array(word_encoding))
            SENT_LENGTH += 1
    return np.array(sent), SENT_LENGTH

def numpy_fillna(data):
    lens = np.array([len(i) for i in data])
    mask = np.arange(lens.max()) < lens[:, None]
    out = np.zeros(shape=(mask.shape + (max_word_length, ALPHABET_SIZE)),
                   dtype='float32')

    out[mask] = np.concatenate(data)
    return out

def make_minibatch(sentences):
    minibatch_x = []
    minibatch_y = []
    max_length = 0
    for sentence in sentences:
        # 0: Negative 1: Positive
        minibatch_y.append(np.array([0, 1]) if sentence[:1] == '0' else np.array([1, 0]))
        one_hot, length = encode_one_hot(sentence[2:-1])
        if one_hot is not None and length > 0:
            if length >= max_length:
                max_length = length
            minibatch_x.append(one_hot)
    minibatch_x = numpy_fillna(minibatch_x)
    return minibatch_x, np.array(minibatch_y)

def load_training_data(filename):
    data_size = 0
    data_set = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            data_size = len(lines)
            random.shuffle(lines)
            data_set = lines
        print "Set at " + filename + " contained: " + str(data_size) + " items."
    else:
        print filename + " doesn't exist. Exiting."
        sys.exit(0)
    return data_size, data_set

if __name__ == '__main__':
    print "Loading training sets"
    TEST_SET_SIZE, test_data = load_training_data(TEST_SET)

    cnn = tdnn(X, kernels, kernel_features)
    cnn = highway(cnn, size)
    cnn = tf.reshape(cnn, [BATCH_SIZE, -1, size])
    with tf.variable_scope('LSTM'):
        cell = create_rnn_cell()
        initial_rnn_state = cell.zero_state(BATCH_SIZE, dtype='float32')
        outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, cnn,
                                                     initial_state=initial_rnn_state,
                                                     dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        last = outputs[-1]
    pred = softmax(last, 2)
    cost = - tf.reduce_sum(Y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))
    predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(predictions, 'float32'))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, SAVE_PATH)
    sess.run(tf.initialize_all_variables())
    accuracy = []
    count = 0
    for line in test_data:
        if len(line) > 0:
            valid_x, valid_y = make_minibatch([line])
            p = sess.run([pred], feed_dict={X: valid_x, Y: valid_y})
            a = sess.run([acc], feed_dict={X: valid_x, Y: valid_y})
            print
            print line
            print('(pos/neg): %.5f/%.5f, prediction: %s' % (p[0][0][0], p[0][0][1], 'pos' if max(p[0][0]) == p[0][0][0] else 'neg'))
            print "Accuracy: " + str(a[0])
            accuracy.append(a)
            count += 1
            if count > 20:
                break
    mean_acc = np.mean(accuracy)
    print
    print "Mean accuracy: " + str(mean_acc)

