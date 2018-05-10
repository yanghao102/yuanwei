#!/usr/bin/python
#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import urllib
import pandas as pd
import csv

tf.logging.set_verbosity(tf.logging.ERROR)              #日志级别设置成 ERROR，避免干扰
np.set_printoptions(threshold='nan')                    #打印内容不限制长度

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
    
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
            raw = urllib.urlopen(IRIS_TRAINING_URL).read()
            with open(IRIS_TRAINING, "w") as f:
                    f.write(raw)
    
    if not os.path.exists(IRIS_TEST):
            raw = urllib.urlopen(IRIS_TEST_URL).read()
            with open(IRIS_TEST, "w") as f:
                    f.write(raw)   

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING,
            target_dtype=np.int,
            features_dtype=np.float32)
    
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST,
            target_dtype=np.int,
            features_dtype=np.float32)
    
    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    
    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=3,
                                                model_dir="iris1_model")
    # Define the training inputs
    def get_train_inputs():
            x = tf.constant(training_set.data)
            y = tf.constant(training_set.target)
            return x, y
    
    # Fit model.
    classifier.fit(input_fn=get_train_inputs, steps=2000)
    
    # Define the test inputs
    def get_test_inputs():
            x = tf.constant(test_set.data)
            y = tf.constant(test_set.target)
    
            return x, y
    
    # Evaluate accuracy.
    #print(classifier.evaluate(input_fn=get_test_inputs, steps=1))
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
    
    print("nTest Accuracy: {0:f}n".format(accuracy_score))
    
    # Classify any new flower samples.
    with open("iris1.csv", "rt", encoding="utf-8") as vsvfile:
        reader = csv.reader(vsvfile)
        rows = [row for row in reader]
        #print(rows)    
    def new_samples():
        return np.array(rows, dtype=np.float32)
    
    predictions = list(classifier.predict(input_fn=new_samples))
    
    print("New Samples, Class Predictions:    {}n".format(predictions))

if __name__ == "__main__":
        main()

exit(0)

