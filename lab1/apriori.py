"""Apriori python implementation

Homework of IoT Information processing. A simple implementation
of Apriori algorithm.

Author: Hu Yongjian
"""
import pandas as pd
import argparse
import time
from itertools import chain, combinations


class Apriori:
    def __init__(self, transaction_list, min_support=0.03, min_confidence=0.6):
        self.transaction_list = transaction_list
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.t_length = len(transaction_list)
        self.item_set = set()

    def __main__(self):
        self.gen_item_set()
        l1, item_set_1 = self.gen_frequent_item_set(self.item_set)
        # self.gen_candidate_item_set(item_set_1)

    def gen_item_set(self):
        for transaction in self.transaction_list:
            for item in transaction:
                self.item_set.add(frozenset([item]))

    def gen_frequent_item_set(self, items):
        l = dict()
        candidate = dict.fromkeys(items, 0)
        item_set = set()

        for t in self.transaction_list:
            for item in items:
                if item.issubset(t):
                    candidate[item] += 1

        for item, count in candidate.items():
            support = count / self.t_length
            if support >= self.min_support:
                l[item] = support
                item_set.add(item)

        return l, itemset

    def gen_candidate_item_set(self, item_set):
        ck = set()
        for l1 in item_set:
            for l2 in item_set:
                c = l1.union(l2)
                if c not in ck and not self.has_infrequent_subset(c, item_set):
                    ck.add(c)
                    print(c)
        return ck

    def has_infrequent_subset(self, c, item_set):
        subset_list = list(powerset(c))
        for item in subset_list:
            if not item.issubset(item_set):
                return False


def powerset(iterable):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def read_file(file_path):
    itemset = list()
    df = pd.read_csv(file_path)
    raw_data = df.iloc[:, 1].values.tolist()
    t0 = time.perf_counter()
    for item in raw_data:
        item = item.strip('{}')
        tmp = item.split(",")
        tmp = sorted(tmp)
        itemset.append(set(tmp))
    t1 = time.perf_counter()
    print("Read Data Time ", (t1 - t0) * 1000)

    return itemset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="input file path")
    parser.add_argument("-s", "--support", type=float, default=0.1,
                        help="minimal support")
    parser.add_argument("-c", "--confidence", type=float, default=0.6, help="minimal confidence")
    args = parser.parse_args()
    apriori = Apriori(read_file(args.file), args.support, args.confidence)
    apriori.__main__()
    pass
