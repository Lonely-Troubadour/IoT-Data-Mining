# -*- coding: utf-8 -*-
"""Partition Around Medoids K-Medoid python implementation.

Homework of IoT Information processing Lab 3. A simple implementation
of PAM algorithm.

Example:
    $ python NaiveBayes.py
    $ python NiaveBayes.py -k num_of_iterations
    $ python NaiveBayes.py -k 25

Author: Yongjian Hu
License: MIT License
"""
import argparse
import pandas as pd

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