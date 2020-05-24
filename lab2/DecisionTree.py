"""Decision Tree python implementation.

Homework of IoT Information processing Lab 1. A simple implementation
of Decision Tree algorithm.

Author: Yongjian Hu
License: MIT License
"""
import pandas as pd

def read_file(file_path):
    df = pd.read_csv(file_path, header=None)
    print(df)

if __name__ == '__main__':
    read_file('Iris.csv')