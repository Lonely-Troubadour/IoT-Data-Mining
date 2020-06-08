# -*- coding: utf-8 -*-
"""K-Means clustering python implementation.

Homework of IoT Information processing Lab 3. A simple implementation
of K-Means clustering algorithm.

Example:
    $ python KMeans.py
    $ python KMeans.py -k num_of_clusters
    $ python KMeans.py -k 3

Author: Yongjian Hu
License: MIT License
"""
import argparse
import pandas as pd
import math
import random
from collections import defaultdict


def read_file(file_path):
    """Read data file from disk.

    Args:
        file_path (str): Path to file on disk.

    Returns:
        df. The data frame object contains the data set.
    """
    col_names = ["x1", "x2", "x3", "x4", "class"]
    dataFrame = pd.read_csv(file_path, names=col_names)
    return dataFrame


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


class KMeans:
    """K-Means clustering method.
    
    Attributes:


    """

    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train
        self.feature_num = x_train.shape[1]
        self.length = len(self.x_train)
        self.centroids = list()
        self.clusters = list()
        pass

    def initialize(self):
        """Initialize algorithm. Arbitrarily choose k objects as initial cluster
        centers.
        """
        for _ in range(self.k):
            index = random.randrange(self.length)
            self.centroids.append(x_train.iloc[index, :].values)

    def train(self):
        for i in range(self.length):
            centroid = self.find_cosest_cluster(i)
            self.clusters.append(centroid)
        print(self.clusters)
        pass

    def find_cosest_cluster(self, point):
        """Find the closest cluster head.
        
        Args:
            point (int): The index of data point.

        Returns:
            Closest cluster centroid index.
        """
        distances = dict()
        for i in range(self.k):
            distance = self.calc_euclid_dist(self.centroids[i], self.x_train.iloc[point], self.feature_num)
            distances[i] = distance
        
        sorted_distances = sorted(distances.items(), key=lambda x: (x[1], x[0]))
        print(sorted_distances)
        return sorted_distances[0][0]

    def calc_euclid_dist(self, p1, p2, no_of_attrs):
        """Calculate Euclidean distance between two points.

        Args:
            p1: Point 1
            p2: Point 2
            no_of_attrs: Number of attributes

        Returns:
            Euclidean distance between 2 points.
        """
        dist_sum = 0.0
        for i in range(no_of_attrs):
            dist_sum += ( p1[i] - p2[i]) ** 2

        dist = math.sqrt(dist_sum)
        return dist


if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", help="Number of clusters, default 3", \
        type=int, default=3)
    args = parser.parse_args()

    # Check k value
    if args.k <= 0:
        raise ValueError("Invalid k. k should be > 0", args.k)
    
    # read data
    df = read_file('Iris.csv')

    # Partition
    #training_set, testing_set = bootstrap(df, df.shape[0])
    

    # seperate feature and class values
    # x_train = training_set.iloc[:, 1:5]
    # y_train = training_set.iloc[:, 0]
    # x_test = testing_set.iloc[:, 1:5]
    # y_train = testing_set.iloc[:, 0]
    x_train = df.iloc[:, 0:4]
    y_train = df.iloc[:, 4]
    print(y_train)

    # K-Means
    kmeans = KMeans(3, x_train, y_train)
    kmeans.initialize()
    kmeans.train()
    pass
