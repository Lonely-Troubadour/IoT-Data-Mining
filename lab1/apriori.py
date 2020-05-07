"""Apriori python implementation

Homework of IoT Information processing. A simple implementation
of Apriori algorithm.

Author:
    Hu Yongjian
"""
import pandas as pd
import argparse
import time

class Apriori():
    def __init__(self, itemset, minSupport = 0.3, minConfidence = 0.7):
        print(itemset, minSupport, minConfidence)
        self.itemset = itemset
        self.minSupport = minConfidence
        self.minConfidence = minConfidence

def read_file(file_path):
    df = pd.read_csv(file_path)
    raw_data = df.iloc[:,1].values.tolist()
    itemset = []
    t0 = time.perf_counter()
    for item in raw_data:
        item = item.strip('{}')
        tmp = item.split(",")
        itemset.append(tmp)

    t1 = time.perf_counter()
    print("Read Data Time ", (t1 - t0) * 1000)
    return itemset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="input file path")
    parser.add_argument("-s", "--support", type=float, default=0.3, \
     help="minimal suppythonport")
    parser.add_argument("-c", "--confidence", type=float, default=0.7, \
     help="minimal confidence")
    args = parser.parse_args()
    apriori = Apriori(read_file(args.file), args.support, args.confidence)
    pass
