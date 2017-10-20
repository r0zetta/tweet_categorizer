from tweet_analysis_cnn_lib import init_network, train_network

if __name__ == '__main__':
    training_data_dir = "training_data"
    training_set_filename = training_data_dir + "/train_set.csv"
    valid_set_filename = training_data_dir + "/valid_set.csv"
    print "Initializing network"
    net = init_network()
    print "Starting training loop"
    train_network(net, training_set_filename, valid_set_filename)

