# -*- coding: utf-8 -*-
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor
from authentication_keys import get_account_credentials
import os.path
import random
import time
import sys
import re
import os
import io


def write_csv(filename, lines):
    total = len(lines)
    print "Writing: " + filename + " with " + str(total) + " entries."
    handle = io.open(filename, "w", encoding="utf-8")
    for line in lines:
        handle.write(line)
    handle.close

def dump_lines(lines):
    test_num = 200
    count = 0
    lines_test = []
    new_lines = []
    for line in lines:
        if count <= test_num:
            lines_test.append(line)
        else:
            new_lines.append(line)
        count += 1
    valid_perc = 0.05
    lines_train = new_lines[:int(len(new_lines) * (1 - valid_perc))]
    lines_valid = new_lines[int(len(new_lines) * (1 - valid_perc)):]
    write_csv("test_set.csv", lines_test)
    write_csv("train_set.csv", lines_train)
    write_csv("valid_set.csv", lines_valid)

if __name__ == '__main__':
    names = {}
    filename = "user_dump_conf.txt"
    if os.path.exists(filename):
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                classification, name = line.split(" ")
                names[name] = classification
    else:
        print "No config supplied. Exiting"
        sys.exit(0)

    print "Targets:"
    print names

    lines = []
    num_names = len(names.items())
    name_count = 1
    for name, classification in names.iteritems():
        print "Getting tweets for " + name + " (" + str(classification) + ") " + str(name_count) + "/" + str(num_names)
        name_count += 1
        acct_name, consumer_key, consumer_secret, access_token, access_token_secret = get_account_credentials()
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        auth_api = API(auth)
        print "Signing in as: " + auth_api.me().name
        tweet_count = 0
        valid_count = 0
        for status in Cursor(auth_api.user_timeline, id=name).items():
            tweet_count += 1
            text = status.text
            text = text.replace('\n', ' ').replace('\r', '')
            text = re.sub("^RT", "", text)
            text = re.sub(r"http\S+", "", text)
            text = text.strip()
            if len(text.split()) > 7:
                valid_count += 1
                line = unicode(classification) + u",\"" + unicode(text) + u"\"\n"
                lines.append(line)
            sys.stdout.write("#")
            sys.stdout.flush()
        print
        print "Got " + str(valid_count) + "/" + str(tweet_count) + " valid tweets."
        random.shuffle(lines)
        dump_lines(lines)
        print
        time.sleep(300)

