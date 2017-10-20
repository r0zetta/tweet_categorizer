# coding=utf-8
from string import printable
from random import randint, shuffle
from tweet_analysis_cnn_lib import init_network, predict
import sys
import string
import os

trained_model_dir = "trained_model"
training_data_dir = "training_data"

def load_test_data(filename):
    data_set = []
    verdicts = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            shuffle(lines)
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




