# -*- coding: utf-8 -*-
"""K-Means clustering python implementation.

Homework of IoT Information processing Lab 3. A simple implementation
of K-Means clustering algorithm.

Example:
    $ python KMeans.py
    $ python KMeans.py -k num_of_clusters
    $ python KMeans.py -k 3
    $ python KMeans.py -v

Author: Yongjian Hu
License: MIT License
"""
import argparse
import pandas as pd
import math
import random
from collections import defaultdict
from array import array


def read_file(file_path):
    """Read data file from disk.

    Args:
        file_path (str): Path to file on disk.

    Returns:
        df. The data frame object contains the data set.
    """
    col_names = ["x1", "x2", "x3", "x4", "class"]
    data_frame = pd.read_csv(file_path, names=col_names)
    return data_frame


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
    """K-Means clustering algorithm model.
    
    Attributes:
        k (int): Number of clusters.
        x_train (pandas.DataFrame): Training set.
        y_train (pandas.Series): Classes of training set.
        feature_num (int): Number of features.
        length (int): Total length of data set.
        random_seed (int): Random seed.
        clusters (list): List of cluster number of corresponding data sample.
        labels (list): Corresponding label of each cluster.
        max_iter (int): Maximum number of iterations
        init (str): "Random" for basic random k initialization. "PlusPlus" for KMeans++ k initialization (Default).
        iterations (int): Maximum iterations.
        verbose (bool): Verbose or not.
    """

    def __init__(self, k, x_train, y_train, random_seed=None, init="PlusPlus", iterations=10, verbose=False):
        """Initialize K-Means clustering algorithm.

        Args:
            k (int): Number of clusters.
            x_train (pandas.DataFrame): Training set.
            y_train (pandas.Series): Classes of training set.
            random_seed (int): Random seed.
            init (str): "Random" for basic random k initialization. "PlusPlus" for KMeans++ k initialization (Default).
            iterations (int): Maximum iterations.
            verbose (bool): Verbose or not.
        """
        self.k = k
        self.x_train = x_train
        self.y_train = y_train
        self.feature_num = x_train.shape[1]
        self.length = len(self.x_train)
        self.clusters = [0] * self.length
        if random_seed:
            random.seed(random_seed)
        self.max_iter = iterations
        self.labels = dict()
        self.verbose = verbose
        if init == "Random":
            self.centroids = self.k_initialize()
        elif init == "PlusPlus":
            self.centroids = self.plus_initialize()

    def plus_initialize(self):
        """K-Means++ initialize k centroids.

        Returns:
            Centroids' coordinates
        """
        index = random.randrange(self.length)

        # First random centroid
        centroids = list()
        centroids.append(self.x_train.iloc[index].values)

        # Candidates
        candidates = [0] * self.length
        distances = [0] * self.length

        for i in range(1, self.k):
            # Get distances of all points to closest cluster
            for j in range(self.length):
                candidates[j] = self.find_closest_cluster(self.x_train.iloc[j], centroids)
                distances[j] = self.calc_euclid_dist(self.x_train.iloc[j], centroids[candidates[j]])

            dist_sum = sum(distances)
            probability = [distances[i] / dist_sum for i in range(len(distances))]

            next_rand = random.random()
            prob_sum = 0
            for j in range(len(probability)):
                prob_sum += probability[j]
                if next_rand < prob_sum:
                    centroids.append(self.x_train.iloc[j].values)
                    break

        return centroids

    def k_initialize(self):
        """Initial step of algorithm. Arbitrarily choose k objects as initial cluster
        centers.

        Returns:
            Centroids' coordinates.
        """
        centroids = list()
        for _ in range(self.k):
            index = random.randrange(self.length)
            centroids.append(self.x_train.iloc[index, :].values)

        if self.verbose:
            print("========Initial Centroids========")
            print(centroids)
            print()

        return centroids

    def train(self):
        """Train the K-Means model."""
        i = 0
        while i < self.max_iter:
            if self.verbose:
                print("========Iteration " + str(i + 1) + "========")
                print("Assigning data points to clusters...")
            self.update_clusters()
            if not self.update_centroid():
                break
            i += 1

        self.labels = self.get_labels()

    def get_labels(self):
        """Get labels of each cluster.

        Returns:
            Labels with corresponding cluster number.
        """
        labels = dict.fromkeys(self.y_train.unique())
        clusters_labels = dict()
        for key in labels.keys():
            labels[key] = array('i', [0 for _ in range(self.k)])

        for i in range(self.length):
            labels[self.y_train.iloc[i]][self.clusters[i]] += 1

        for key in labels.keys():
            maximum = max(labels[key])
            for i in range(len(labels[key])):
                if labels[key][i] == maximum:
                    if i in clusters_labels:
                        print("WARNING! Same class for different clusters.")
                        print("Trying to fix...")

                        while i in clusters_labels and len(clusters_labels) != 3:
                            i = (i + 1) % self.k

                        print("Done!")
                    clusters_labels[i] = key
                    break

        if self.verbose:
            print(labels)
            print(clusters_labels)

        return clusters_labels

    def update_clusters(self):
        """Update data points' assignment to clusters"""
        for i in range(self.length):
            centroid = self.find_closest_cluster(self.x_train.iloc[i], self.centroids)
            self.clusters[i] = centroid

        if len(set(self.clusters)) < 3:
            raise Exception("Bad initialization. Clusters fewer than expected", len(self.clusters))

    def update_centroid(self):
        """Update centroids

        Returns:
            True if update successfully, False if fail to update
        """
        count = 0
        result = self.x_train.groupby(self.clusters).mean().values

        for i in range(self.k):
            if (self.centroids[i] != result[i]).any():
                self.centroids[i] = result[i]
            else:
                count += 1

        if self.verbose:
            print("--------New Centroids--------")
            print(self.centroids)
            print()

        if count == self.k:
            return False
        else:
            return True

    def predict(self, data_set):
        """Predict based on trained model and given testing set.

        Returns:
            Prediction result.
        """
        y_predict = list()
        for i in range(len(data_set)):
            centroid = self.find_closest_cluster(data_set.iloc[i], self.centroids)
            y_predict.append(self.labels[centroid])

        return y_predict

    def find_closest_cluster(self, point, centroids):
        """Find the closest cluster head.
        
        Args:
            centroids (list): The list of coordinates of centroids.
            point (array): The features array of data point.

        Returns:
            Closest cluster centroid index.
        """
        distances = dict()
        for i in range(len(centroids)):
            distance = self.calc_euclid_dist(centroids[i], point)
            distances[i] = distance

        sorted_distances = sorted(distances.items(), key=lambda x: (x[1], x[0]))
        closest_centroid = sorted_distances[0]
        return closest_centroid[0]

    def calc_euclid_dist(self, p1, p2):
        """Calculate Euclidean distance between two points.

        Args:
            p1: Point 1
            p2: Point 2

        Returns:
            Euclidean distance between 2 points.
        """
        dist_sum = 0.0
        for i in range(len(p1)):
            dist_sum += (p1[i] - p2[i]) ** 2

        dist = math.sqrt(dist_sum)
        return dist


def calc_accuracy(predict, labels):
    """Calculate accuracy

    Returns:
        Accuracy.
    """
    return sum(predict == labels) / len(labels)


def bootstrap_accuracy(data_set, k=20, verbose=False):
    """Calculate model accuracy using .632 bootstrap.

    Args:
        data_set (pandas.DataFrame): Data set.
        k (int): The number of iterations. Default is 20
        verbose (bool): Turn on verbosity or not.

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
        # x_train, x_test = feature_scaler(x_train, x_test)

        # K-Means train
        kmeans = KMeans(3, x_train, y_train, init="Random", verbose=verbose)
        kmeans.plus_initialize()
        kmeans.train()

        # Predict
        predict_train = kmeans.predict(x_train)
        acc_train = calc_accuracy(predict_train, y_train)
        predict_test = kmeans.predict(x_test)
        acc_test = calc_accuracy(predict_test, y_test)

        # Accuracy
        acc_test = calc_accuracy(predict_train, y_train)
        acc_train = calc_accuracy(predict_test, y_test)

        print("Iteration " + str(i) + ": ", end="")
        print("Acc_test = " + str(acc_test) + ", Acc_train = " + str(acc_train))
        acc_sum += 0.632 * acc_test + 0.368 * acc_train

    return acc_sum / k


if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", help="Number of clusters, default 3", type=int, default=3)
    parser.add_argument("-v", "--verbose", help='Verbose', action="store_true")
    args = parser.parse_args()

    # Check k value
    if args.k <= 0:
        raise ValueError("Invalid k. k should be > 0", args.k)

    if args.verbose:
        print("Verbosity turned on.")

    # read data
    df = read_file('Iris.csv')
    acc = bootstrap_accuracy(df, 20)
    print("Model accuracy is {:.2f}".format(acc))
