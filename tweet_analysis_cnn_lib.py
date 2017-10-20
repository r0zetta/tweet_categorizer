# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from string import printable
from random import randint
import numpy as np
import codecs
import random, csv
import sys
import string
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

trained_model_dir = "trained_model"
training_data_dir = "training_data"

emb_alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '
DICT = {ch: ix for ix, ch in enumerate(emb_alphabet)}
ALPHABET_SIZE = len(DICT)
batch_size = 1
max_word_length = 30

################################
# Tensorflow supporting routines
################################

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
    rnn_size=650
    dropout=0.0
    cell = rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
    if dropout > 0.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1. - dropout)
    return cell

################################
# Input manipulation routines
################################

def encode_one_hot(sentence):
    sent = []
    SENT_LENGTH = 0
    encoded_sentence = filter(lambda x: x in (printable), sentence)
#    print(encoded_sentence)
    for word in word_tokenize(encoded_sentence.decode('utf-8', 'ignore').encode('utf-8')):
        if len(word) > max_word_length:
            continue
        word_encoding = np.zeros(shape=(max_word_length, ALPHABET_SIZE))
        for i, char in enumerate(word):
            if char in DICT:
                char_encoding = DICT[char]
                one_hot = np.zeros(ALPHABET_SIZE)
                one_hot[char_encoding] = 1
                word_encoding[i] = one_hot
            else:
                print
                print char + " not in DICT"
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

def vectorize(sentences):
    vector_x = []
    vector_y = []
    max_length = 0
    for sentence in sentences:
        vector_y.append(np.array([0, 1]) if sentence[:1] == '0' else np.array([1, 0]))
        one_hot, length = encode_one_hot(sentence[2:-1])
        if length >= max_length:
            max_length = length
        vector_x.append(one_hot)
    vector_x = numpy_fillna(vector_x)
    return vector_x, np.array(vector_y)

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

def get_next_training_batch(data, position):
    if position + batch_size > len(data):
        position = 0
    new_position = position+batch_size
    ret = data[position:new_position]
    return ret, new_position

def get_next_test_batch(data):
    ret = []
    for i in range(0, batch_size):
        ret.append(data[randint(0, len(data))])
    return ret

def load_variable(varname):
    ret = 0.0
    filename = trained_model_dir + "/" + varname + ".txt"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            if len(lines[0]) > 0:
                ret = lines[0]
                print "Read: " + varname + " = " + str(ret)
    return ret

def save_variable(varname, value):
    filename = trained_model_dir + "/" + varname + ".txt"
    with open(filename, "w") as f:
        print "Saved: " + varname + " = " + str(value)
        f.write(str(value))

def init_network():
    model_save_path = trained_model_dir + '/lstm'
    learning_rate = 0.0001
    size = 700
    kernels=[1, 2, 3, 4, 5, 6, 7]
    kernel_features=[25, 50, 75, 100, 125, 150, 175]
    X = tf.placeholder('float32', shape=[None, None, max_word_length, ALPHABET_SIZE], name='X')
    Y = tf.placeholder('float32', shape=[None, 2], name='Y')
    cnn = tdnn(X, kernels, kernel_features)
    cnn = highway(cnn, size)
    cnn = tf.reshape(cnn, [batch_size, -1, size])
    with tf.variable_scope('LSTM'):
        cell = create_rnn_cell()
        initial_rnn_state = cell.zero_state(batch_size, dtype='float32')
        outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, cnn,
                                                     initial_state=initial_rnn_state,
                                                     dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        last = outputs[-1]
    pred = softmax(last, 2)
    predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(predictions, 'float32'))
    cost = - tf.reduce_sum(Y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    saver = tf.train.Saver()
    sess = tf.Session()
    if os.path.exists(trained_model_dir + "/lstm.index"):
        print "Restoring previously saved model."
        saver.restore(sess, model_save_path)
    else:
        print "No previously stored model."
    net = {}
    net["X"] = X
    net["Y"] = Y
    net["sess"] = sess
    net["pred"] = pred
    net["acc"] = acc
    net["optimizer"] = optimizer
    net["saver"] = saver
    net["cost"] = cost
    return net

def train_network(net, train_set_filename, valid_set_filename):
    model_save_path = trained_model_dir + '/lstm'
    logging_path = trained_model_dir + '/log.txt'
    train_set_size, training_data = load_training_data(train_set_filename)
    valid_set_size, valid_data = load_training_data(valid_set_filename)
    training_set_position = int(load_variable("training_set_position"))
    saver = net["saver"]
    sess = net["sess"]
    acc = net["acc"]
    optimizer = net["optimizer"]
    cost = net["cost"]
    X = net["X"]
    Y = net["Y"]
    sess.run(tf.global_variables_initializer())
    best_acc = float(load_variable("best_acc"))
    DONE = False
    batch = 0
    log_every = 10
    save_every = 50
    patience = int(load_variable("patience"))
    if patience == 0:
        patience = 10000
    processed = int(load_variable("processed"))
    print "Training..."
    while not DONE:
        loss = 0.0
        current_training_batch, training_set_position = get_next_training_batch(training_data, training_set_position)
        batch += 1
        processed += batch_size
        batch_x, batch_y = vectorize(current_training_batch)
        if batch_x is None:
            continue
        print "Running sample " + str(training_set_position) + "/" + str(train_set_size) + " in training set. (total processed: " + str(processed) + ")"
        _, c, a = sess.run([optimizer, cost, acc], feed_dict={X: batch_x, Y: batch_y})
        loss += c

        if processed % log_every == 0:
            log = open(logging_path, 'a')
            log.write('%s, %6d, %.5f, %.5f \n' % ('train', batch, loss/batch, a))
            log.close()
            print "Log appended"
            print

        if processed % save_every == 0:
            accuracy = []
            test_count = 0
            print
            print "Running accuracy test"
            while test_count <= 100:
                test_batch = get_next_test_batch(valid_data)
                valid_x, valid_y = vectorize(test_batch)
                if valid_x is None:
                    print "ERROR!"
                    print test_batch
                    continue
                a = sess.run([acc], feed_dict={X: valid_x, Y: valid_y})
                accuracy.append(a)
                test_count += 1
                sys.stdout.write("#")
                sys.stdout.flush()
            mean_acc = np.mean(accuracy)
            print
            print "Mean accuracy: " + str(mean_acc) + " (best: " + str(best_acc) + ")"
            print "Patience: " + str(patience)
            if mean_acc > best_acc:
                best_acc = mean_acc
                save_path = saver.save(sess, model_save_path)
                print('Model saved in file: %s' % save_path)
            else:
                patience -= 500
                if patience <= 0:
                    print "Patience ended"
                    DONE = True
                    break
            log = open(logging_path, 'a')
            log.write('%s, %6d, %.5f \n' % ('valid', batch, mean_acc))
            log.close()
            save_variable("training_set_position", training_set_position)
            save_variable("best_acc", best_acc)
            save_variable("processed", processed)
            save_variable("patience", patience)

def predict(net, sentence, verdict):
    sess = net["sess"]
    pred = net["pred"]
    X = net["X"]
    Y = net["Y"]
    sess.run(tf.global_variables_initializer())
    sentence = str(verdict) + "," + sentence
    sentence = [sentence]
    valid_x, valid_y = vectorize(sentence)
    p = sess.run([pred], feed_dict={X: valid_x, Y: valid_y})
    pos = p[0][0][0]
    neg = p[0][0][1]
    final = 0
    if pos > neg:
        final = 1
    return pos, neg, final



################################
# Trainer loop
################################

def load_test_data(filename):
    data_set = []
    verdicts = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines:
                data_set.append(line[2:])
                v = int(line[0])
                if v > 0:
                    v = 1
                verdicts.append(v)
            data_size = len(data_set)
        print "Set at " + filename + " contained: " + str(data_size) + " items."
    else:
        print filename + " doesn't exist. Exiting."
        sys.exit(0)
    return data_set, verdicts

if __name__ == '__main__':
    training_set_filename = training_data_dir + "/train_set.csv"
    valid_set_filename = training_data_dir + "/valid_set.csv"
    print "Initializing network"
    net = init_network()
    print "Running test set"
    test_data, verdicts = load_test_data("training_data/test_set.csv")
    missed = []
    total = 0
    correct = 0
    for x, t in enumerate(test_data):
        verdict = verdicts[x]
        pos, neg, final = predict(net, t, verdict)
        print str(x) + "/" + str(len(test_data)) + " Predicted verdict: " + str(final) + " actual: " + str(verdict)
        if verdict != final:
            missed.append(t)
        else:
            correct += 1
        total += 1
    print "Missed samples:"
    for m in missed:
        print m
    accuracy = float(float(correct)/float(total))*100.00
    print "Correct: " + str(correct) + "/" + str(total) + " (" + str(accuracy) + ")"
    print "Starting training loop"
    train_network(net, training_set_filename, valid_set_filename)




