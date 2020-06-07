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

def calc_euclid_dist(p1, p2, no_of_attrs):
    sum = 0.0
    for i in range(no_of_attrs):
        sum += p1[i] ** 2 + p2[i] ** 2

    dist = math.sqrt(sum)
    return dist

if __name__ == "__main__":
    # parse arguement
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", help="Number of clusters, defualt 3", \
        type=int, default=3)
    args=parser.parse_args()

    # check k value
    if (args.k <= 0):
        raise ValueError("Invalid k. k should be > 0", args.k)
    
    pass