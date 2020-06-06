# -*- coding: utf-8 -*-
"""Naive Bayes python implementation.

Homework of IoT Information processing Lab 2. A simple implementation
of Naive Bayes algorithm.

Example:
    $ python NaiveBayes.py
    $ python NiaveBayes.py -k num_of_iterations
    $ python NaiveBayes.py -k 25

Author: Yongjian Hu
License: MIT License
"""
from collections import defaultdict

import pandas as pd
import random
import math
import array
import argparse


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
    std = x_train.std(ddof=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test


def calc_accuracy(y_test, prediction):
    """Accuracy of the prediction.

    Args:
        y_test (pandas.DataFrame): Actual classes of test set.
        prediction (list): Predicted classes of test set.

    Returns:
        Accuracy of the prediction.
    """
    count = 0
    length = len(y_test)
    for i in range(length):
        if prediction[i] == y_test.iloc[i]:
            count += 1

    return count / length


class NaiveBayesClassifier:
    """Naive Bayes Classifier.

    Attributes:
        x_train (pandas.DataFrame): Training set.
        y_train (pandas.DataFrame): Classes of training set.
        feature_num (int): Feature number.
        class_num (int): Class number.
        prob_y (dict): prior probability of each class.
        feature_mean (dict): Mean of each feature of each class.
        feature_std (dict): Standard deviation of each feature of each class.
        length (int): Length of training set..
    """

    def __init__(self, x_train, y_train, feature_num, class_num):
        """Initialize naive Bayes classifier.

        Args:
            x_train (pandas.DataFrame): Training set.
            y_train (pandas.DataFrame): Classes of training set.
            feature_num (int): No. of features.
            class_num (int): No. of classes.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.feature_num = feature_num
        self.class_num = class_num
        self.prob_y = defaultdict(float)
        self.feature_mean = defaultdict(array.array)
        self.feature_std = defaultdict(array.array)
        self.length = x_train.shape[0]

    def train(self):
        """Train the Gaussian Naive Bayes model.

        Returns:
            Prior probability of each class, 
            Mean value and standard deviation of each feature of different classes.
        """

        # Get Probability(c), prior prob of each class c.
        class_count = self.y_train.groupby(self.y_train).size()
        for class_, count in class_count.items():
            self.prob_y[class_] = round(count / self.length, 6)
        self.prob_y = dict(self.prob_y)

        # Get mean and std for each feature of each class.
        feature_sum = defaultdict(array.array)
        feature_mean = defaultdict(array.array)
        feature_std = defaultdict(array.array)

        # Initialize array in dict.
        for class_ in self.prob_y.keys():
            feature_sum[class_] = array.array('f', [.0 for _ in range(self.feature_num)])
            feature_mean[class_] = array.array('f', [.0 for _ in range(self.feature_num)])
            feature_std[class_] = array.array('f', [.0 for _ in range(self.feature_num)])

        # Sum.
        for i in range(self.length):
            for j in range(self.feature_num):
                feature_sum[self.y_train.iloc[i]][j] += self.x_train.iloc[i, j]

        # Mean.
        for class_, count in class_count.items():
            for j in range(self.feature_num):
                feature_mean[class_][j] = feature_sum[class_][j] / count

        # Standard deviation.
        for i in range(self.length):
            class_ = self.y_train.iloc[i]
            for j in range(self.feature_num):
                feature_std[class_][j] += (self.x_train.iloc[i, j] - feature_mean[class_][j]) ** 2

        for class_, count in class_count.items():
            for j in range(self.feature_num):
                feature_std[class_][j] = (feature_std[class_][j] / count) ** 0.5

        self.feature_mean = dict(feature_mean)
        self.feature_std = dict(feature_std)

        return self.prob_y, self.feature_mean, self.feature_std

    def gaussian_pdf(self, x, mean, std):
        """Gaussian distribution, probability density function.

        N(x, mu, theta) = ( 1/(2pi)^0.5 theta ) * ( e^-( (x - mu)^2/2 theta^2 ) )

        Args:
            x (float): probability.
            mean (float): mean value.
            std (float): standard deviation.

        Returns:
            Probability.
        """
        prob = math.e ** (-(x - mean) ** 2 / (2 * std ** 2)) / ((2 * math.pi) ** 0.5 * std)
        return prob

    def joint_prob(self, test_data):
        """Calculate joint probability of likelihood and prior probability.

        Args:
            test_data (list): Test data set, contains features of the test data.

        Returns:
            Joint probability of each class.
        """
        joint_probs = defaultdict(float)
        for class_ in self.prob_y.keys():
            likelihood = 1.0

            # Calculate likelihood first.
            for i in range(self.feature_num):
                feature = test_data[i]
                mean = self.feature_mean[class_][i]
                std = self.feature_std[class_][i]
                gaussian_prob = self.gaussian_pdf(feature, mean, std)
                likelihood += gaussian_prob

            # Calculate prior_prob * likelihood.
            prior_prob = self.prob_y[class_]
            joint_probs[class_] = prior_prob * likelihood
        return dict(joint_probs)

    def get_max(self, test_data):
        """Get maximum probability from all joint probabilities,
        and hence predict the class.

        Args:
            test_data (list): Test data set, contains features of test data.

        Returns:
            Predicted class that has the max probability.
        """
        joint_probs = self.joint_prob(test_data)
        max_prob = max(joint_probs, key=joint_probs.get)
        return max_prob

    def predict(self, test_set):
        """Predict on the give test set.

        Args:
            test_set (pandas.DataFrame): Test data set.

        Returns:
            List of predictions.
        """
        prediction = list()
        for row in test_set.values:
            max_prob = self.get_max(row)
            prediction.append(max_prob)
        return prediction


def bootstrap_accuracy(data_set, k=20):
    """Calculate model accuracy using .632 bootstrap.

    Args:
        data_set (pandas.DataFrame): Data set.
        k (int): The number of iterations. Default is 20

    Returns:
        Accuracy of the model.
    """
    acc_sum = 0
    for i in range(k):
        # Partition
        training_set, testing_set = bootstrap(data_set, data_set.shape[0])

        # Separates features and classes
        x_train = training_set.iloc[:, 1:5]
        y_train = training_set.iloc[:, 0]
        x_test = testing_set.iloc[:, 1:5]
        y_test = testing_set.iloc[:, 0]

        # Feature scaling
        x_train, x_test = feature_scaler(x_train, x_test)

        # Train
        classifier = NaiveBayesClassifier(x_train, y_train, 4, 3)
        classifier.train()

        # Predict
        prediction_test = classifier.predict(x_test)
        prediction_train = classifier.predict(x_train)

        # Accuracy
        acc_test = calc_accuracy(y_test, prediction_test)
        acc_train = calc_accuracy(y_train, prediction_train)

        print("Iteration " + str(i) + ": ", end="")
        print("Acc_test = " + str(acc_test) + ", Acc_train = " + str(acc_train))
        acc_sum += 0.632 * acc_test + 0.368 * acc_train

    return acc_sum / k


if __name__ == '__main__':
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", help="Number of iteration, default 20", \
    type=int, default=20)
    args = parser.parse_args()

    # Check k value
    if (args.k <= 0):
        raise Exception("Invalid k. k should be > 0", args.k)

    # Read file
    df = read_file('Iris.csv')

    # Using .632 bootstrap
    accuracy = bootstrap_accuracy(df, args.k)

    # Print model accuracy
    print("Accuracy " + str(accuracy))
