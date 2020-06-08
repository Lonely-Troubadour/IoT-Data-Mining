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

    def __init__(self, k, data_set):
        self.k = k
        self.data_set = data_set
        pass

    def initialize(self):
        """Initialize algorithm. Arbitrarily choose k objects as initial cluster
        centers.
        """
        index = list()
        for _ in range(self.k):
            index.append(random.randrange(len(self.data_set)))

        print(index)

        pass

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
            dist_sum += p1[i] ** 2 + p2[i] ** 2

        dist = math.sqrt(dist_sum)
        return dist





if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", help="Number of clusters, default 3", \
                        type=int, default=3)
    args = parser.parse_args()

    # check k value
    if args.k <= 0:
        raise ValueError("Invalid k. k should be > 0", args.k)

    df = read_file('Iris.csv')
    kmeans = KMeans(3, df)
    kmeans.initialize()
    pass
