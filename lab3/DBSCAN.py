# -*- coding: utf-8 -*-
"""Density-Based Spatial Clustering of Applications with Noise (DBSCAN) python implementation.

Homework of IoT Information processing Lab 3. A simple implementation
of DBSCAN algorithm.

Example:
    $ python DBSCAN.py # Default
    $ python DBSCAN.py -h # Help message
    $ python DBSCAN.py -eps epsilon
    $ python DBSCAN.py -p minimal_Points
    $ python DBSCAN.py -v # Verbosity turned on
    $ python DBSCAN.py -r random_seed

Author: Yongjian Hu
License: MIT License
"""
import argparse
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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

    Attributes:
        x (pandas.DataFrame): Training data set.
        y (pandas.DataFrame): Classes of data set.
        eps (float): Epsilon. Minimum distance. Default 0.5.
        min_pts (int): Minimum points in the neighbor. Defualt 5
        length (int): Length of data set.
        visited (numpy.array): Array of data set visited status. 0 Stands for not visited, 1 opposite.
        noise (numpy.array): Array of noise. 0 stands for normal data, 1 for noise.
        clusters (numpy.array): Array of clusters assignments. 0 Stands for noise.
        cluster_num (int): Number of clusters.
        verbose (bool): Verbosity. Default false.
    """

    def __init__(self, x, y, random_seed=None, eps=0.5, min_pts=5, verbose=False):
        """Initialize DBSCAN algorithm.

        Args:
            x (pandas.DataFrame): Training data set.
            y (pandas.DataFrame): Classes of data set.
            eps (float): Epsilon. Minimum distance. Default 0.5.
            min_pts (int): Minimum points in the neighbor. Defualt 5
        """
        self.X = x
        self.y = y
        self.eps = eps
        self.min_pts = min_pts
        self.length = len(x)
        self.visited = np.zeros(self.length, dtype=int)
        self.noise = np.zeros(self.length, dtype=int)
        self.clusters = np.zeros(self.length, dtype=int)
        self.cluster_num = 0
        self.verbose = verbose

    def train(self):
        """Train. Find clusters.

        Returns:
            None.
        """
        cluster_num = 0
        while (self.visited == 0).any():
            # Randomly selected an unvisited object p
            p = random.randrange(self.length)
            while self.visited[p] == 1:
                p = random.randrange(self.length)

            # Mark p as visited
            self.visited[p] = 1

            # Get neibors of p
            neighbors = self.get_neighbors(p)

            if len(neighbors) >= self.min_pts:
                # New cluster
                cluster_num += 1
                self.clusters[p] = cluster_num
                # For each point in neighborhood N of p
                while len(neighbors) != 0:
                    p_prime = neighbors.pop()
                    if self.visited[p_prime] == 0:
                        # Mark p' as visited
                        self.visited[p_prime] = 1
                        # If p' is core point. Add its neighbors to N.
                        neighbors_prime = self.get_neighbors(p_prime)
                        if len(neighbors_prime) >= self.min_pts:
                            neighbors.extend(neighbors_prime)
                    # If p' is not yet a member of any cluster
                    if self.clusters[p_prime] == 0:
                        self.clusters[p_prime] = cluster_num
            else:
                # Mark noise
                self.noise[p] = 1

        self.cluster_num = cluster_num + 1

    def calc_euclid_distance(self, p1, p2):
        """Calculate Euclidean distance between 2 points.

        Args:
            p1 (numpy.array): Point 1.
            p2 (numpy.array): Point 2.

        Returns:
            Euclidean distance between 2 points.
        """
        return math.sqrt(sum((p1 - p2) ** 2))

    def get_neighbors(self, point):
        """Get neighbors of the point.

        Args:
            point (int): Point index.

        Returns:
            List of neighbor indexes.
        """
        neighbors = list()
        for i in range(self.length):
            if i != point and self.calc_euclid_distance(self.X.iloc[point], self.X.iloc[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def get_labels(self):
        """Get labels of each cluster.

        Returns:
            Labels with corresponding cluster number.
        """
        labels = dict.fromkeys(self.y.unique())
        cluster_labels = dict()
        for key in labels.keys():
            labels[key] = np.zeros(self.cluster_num, dtype=int)

        for i in range(self.length):
            labels[self.y.iloc[i]][self.clusters[i]] += 1

        for key in labels.keys():
            index = np.argmax(labels[key])
            cluster_labels[index] = key

        if self.verbose:
            print("Number of clusters: " + str(self.cluster_num - 1))
            print("Note: cluster num 0 is the set of noise points.")
            print("========labels -> cluster no.========")
            print(labels)

        return cluster_labels


if __name__ == "__main__":
    # parse arguement
    parser = argparse.ArgumentParser()
    parser.add_argument("-eps", help="Epsilon, minimal distance, defualt 0.5", type=float, default=.5)
    parser.add_argument("-p", "--minPts", help="Minimal points in one neighbor, default 5", type=int, default=5)
    parser.add_argument("-v", "--verbose", help="Verbosity", action="store_true")
    parser.add_argument("-r", "--rand", help="Random seed, default None", type=int, default=None)
    args = parser.parse_args()


    print("==========Parameters===========")
    # check args value
    if args.eps <= 0:
        raise ValueError("Invalid Epsilon. Epsilon should be > 0", args.eps)
    else:
        print("- Epsilon: " + str(args.eps))

    if args.minPts <= 0:
        raise ValueError("Invalid minimum points. min_pts should be > 0", args.minPts)
    else:
        print("- MinPts: " + str(args.minPts))

    if args.rand != None and args.rand <= 0:
        raise ValueError("Invalid random seed.", args.rand)
    else:
        print("- Random seed: " + str(args.rand))

    if args.verbose:
        print("- Verbosity turned on")
    
    print("DBSCAN running...")
    print("===============================")
    print()

    # Read data
    df = read_file('Iris.csv')

    # Training set seperated into feautres and classes
    x_train = df.iloc[:, 0:4]
    y_train = df.iloc[:, 4]

    # Start runing algorithm
    dbscan = DBSCAN(x_train, y_train, random_seed=args.rand, eps=args.eps, min_pts=args.minPts, verbose=args.verbose)
    dbscan.train()
    print("========Cluster no. -> class label:======== \n" + str(dbscan.get_labels()))

    # Plot
    cmap = ListedColormap(['r', 'g', 'b', 'k'])
    plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 2], c=dbscan.clusters, edgecolor='k', s=40)
    plt.savefig('figure.png', dpi=300)
