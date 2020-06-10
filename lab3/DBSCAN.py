# -*- coding: utf-8 -*-
"""Density-Based Spatial Clustering of Applications with Noise (DBSCAN) python implementation.

Homework of IoT Information processing Lab 3. A simple implementation
of DBSCAN algorithm.

Example:
    $ python NaiveBayes.py
    $ python NiaveBayes.py -k num_of_iterations
    $ python NaiveBayes.py -k 25

Author: Yongjian Hu
License: MIT License
"""
import argparse
import random
import math
import pandas as pd
import numpy as np


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


class DBSCAN:
    """DBSCAN clustering algorithm.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.


    """

    def __init__(self, x, y, random_seed=None, eps=0.5, min_pts=5):
        """Initialize DBSCAN algorithm.

        Args:
            x: Training data set.
            y: Classes of data set.
            eps: Epsilon. Minimum distance. Default 0.5.
            min_pts: Minimum points in the neighbor. Defualt 5
        """
        self.X = x
        self.y = y
        self.eps = eps
        self.min_pts = min_pts
        self.length = len(x)
        self.visited = np.zeros(self.length)
        self.noise = np.zeros(self.length)
        pass

    def train(self):
        # Randomly selected an unvisited object p

        pass

    def calc_euclid_distance(self, p1, p2):
        """Calculate Euclidean distance between 2 points.

        Args:
            p1 (numpy.array): Point 1.
            p2 (numpy.array): Point 2.

        Returns:
            Euclidean distance between 2 points.
        """
        return math.sqrt(sum((p1 - p2) ** 2))


if __name__ == "__main__":
    # parse arguement
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", help="Number of clusters, defualt 3", type=int, default=3)
    args = parser.parse_args()

    # check k value
    if args.k <= 0:
        raise ValueError("Invalid k. k should be > 0", args.k)

    df = read_file('Iris.csv')

    x_train = df.iloc[:, 0:4]
    y_train = df.iloc[:, 4]
    # x_test = testing_set.iloc[:, 1:5]
    # y_test = testing_set.iloc[:, 0]

    dbscan = DBSCAN(x_train, y_train)
    print(dbscan.calc_euclid_distance(np.array([1,1,1]), np.array([2,3,4])))

    pass
