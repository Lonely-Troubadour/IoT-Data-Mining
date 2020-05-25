# -*- coding: utf-8 -*-
"""Decision Tree python implementation.

Homework of IoT Information processing Lab 1. A simple implementation
of Decision Tree algorithm.

Author: Yongjian Hu
License: MIT License
"""
from collections import defaultdict

import pandas as pd
import random
import array


def read_file(file_path):
    """Read data file from disk.

    Args:
        file_path (str): Path to file on disk.

    Returns:
        df. The data frame object contains the data set.
    """
    col_names = ["x1", "x2", "x3", "x4", "class"]
    df = pd.read_csv(file_path, names=col_names)
    return df


def bootstrap(data, length):
    """Partition the data set to training set and testing set.

    Args:
        data (pandas.DataFrame): Data frame that contains the data set.
        length (int): The length of data set.

    Return:
        training set and testing set.
    """
    index = random.randint(0, length - 1)
    training_set = pd.DataFrame()
    testing_set = pd.DataFrame()
    index_set = set()

    # Select training set
    for _ in range(length):
        index_set.add(index)
        row = data.iloc[index]
        training_set = training_set.append(row)
        index = random.randint(0, length - 1)

    # Let the remaining to be testing set
    for i in range(length):
        if i not in index_set:
            testing_set = testing_set.append((data.iloc[i]))

    return training_set, testing_set


def feature_scaler(x_train, x_test):
    """Feature scaler. Standardize the features.

    Args:
        x_train (pandas.DataFrame): features of training set.
        x_test (pandas.DataFrame): features of testing set.

    Returns:
        Training set and testing set after scaling.
    """
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train + mean) / std
    x_test = (x_test + mean) / std
    return x_train, x_test


class NaiveBayesClassifier:
    """Naive Bayes Classifier.

    Attributes:
        data_set (pandas.DataFrame): Data set.
        # training_set (pandas.DataFrame): Training set.
        # testing_set (pandas.DataFrame): Testing set.
    """

    def __init__(self, x_train, y_train, feature_num, class_num):
        """Initialize naive Bayes classifier.

        Args:
            data (pandas.DataFrame): Data set.
            feature_num (int): No. of features.
            class_num (int): No. of classes.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.feature_num = feature_num
        self.class_num = class_num
        self.prob_y = defaultdict(float)  # array.array('f')
        self.prob_x_cond_c = defaultdict(float)  # array.array('f')
        self.length = x_train.shape[0]
        # self.training_set, self.testing_set = bootstrap(self.data_set, self.data_set.shape[0])

    pass

    def train(self):
        # Get Probability(c), prior prob of each class c.
        class_count = self.y_train.groupby(self.y_train).size()
        for class_, count in class_count.items():
            self.prob_y[class_] = round(count / self.length, 6)

        # Get Probability(x|c), posterior prob of each class c and feature x.

    def predict(self):
        pass


if __name__ == '__main__':

    # Read file
    df = read_file('Iris.csv')

    # Partition
    training_set, testing_set = bootstrap(df, df.shape[0])

    # Separates features and classes
    x_train = training_set.iloc[:, 1:5]
    y_train = training_set.iloc[:, 0]
    x_test = testing_set.iloc[:, 1:5]
    y_test = testing_set.iloc[:, 0]

    # Feature scaling
    x_train, x_test = feature_scaler(x_train, x_test)

    classifier = NaiveBayesClassifier(x_train, y_train, 4, 3)
    classifier.train()
