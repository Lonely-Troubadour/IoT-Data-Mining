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
    def __init__(self, transaction_list, min_support=0.01, min_confidence=0.6):
        self.transaction_list = transaction_list
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.t_length = len(transaction_list)
        self.item_set = set()

    def __main__(self):
        # Initialize. Generate item set from transaction lists
        self.gen_item_set()
        frequent_item_set = list()
        frequent_set = list()

        # Get frequent 1-item set
        tmp_set, tmp_item_set = self.gen_frequent_item_set(self.item_set)

        # Iterates
        k = 2
        while tmp_item_set != set():
            frequent_set.append(tmp_set)
            frequent_item_set.append(tmp_item_set)
            candidate_item_set = self.gen_candidate_item_set(tmp_item_set)
            tmp_set, tmp_item_set = self.gen_frequent_item_set(candidate_item_set)
            frequent_set.append(tmp_set)
            frequent_item_set.append(tmp_item_set)
            k += 1

    def gen_item_set(self):
        for transaction in self.transaction_list:
            for item in transaction:
                self.item_set.add(frozenset([item]))

    def gen_frequent_item_set(self, items):
        """
        Generate frequent item set.

        Args:
            items: The items from which generates frequent item set.

        Returns:
            The frequent item set with support and frequent item set.
        """
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

        return l, item_set

    def gen_candidate_item_set(self, item_set):
        """
        Generates candidate item sets from given items.

        Args:
            item_set: The item set from which generates candidate.

        Returns:
            The set of candidate sets.
        """
        ck = set()
        for l1 in item_set:
            for l2 in item_set:
                if l1 != l2:
                    c = l1.union(l2)
                    if not self.has_infrequent_subset(c, item_set):
                        ck.add(c)
        return ck

    def has_infrequent_subset(self, c, item_set):
        """
        Check if the candidate has subset that is not the subset of item set.

        Args:
            c: The candidate set.
            item_set: The frequent item set.

        Returns:
            False If candidate set c has infrequent subset.
        """
        subset_list = map(frozenset, [item for item in powerset(c)])
        for item in subset_list:
            if not item.issubset(item_set):
                return False


def powerset(iterable):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def read_file(file_path):
    """
    Read csv data file from disk.

    Parameters:
        file_path: The path to the data file.

    Returns:
        The transactions data list.
    """
    transaction_list = list()
    df = pd.read_csv(file_path)
    raw_data = df.iloc[:, 1].values.tolist()
    t0 = time.perf_counter()
    for item in raw_data:
        item = item.strip('{}')
        tmp = item.split(",")
        transaction_list.append(set(tmp))
    t1 = time.perf_counter()
    print("Read Data Time ", (t1 - t0) * 1000)

    return transaction_list


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-f", "--file", help="input file path")
    # parser.add_argument("-s", "--support", type=float, default=0.1,
    #                     help="minimal support")
    # parser.add_argument("-c", "--confidence", type=float, default=0.6, help="minimal confidence")
    # args = parser.parse_args()
    file_path = "Groceries.csv"
    # apriori = Apriori(read_file(args.file), args.support, args.confidence)
    apriori = Apriori(read_file(file_path))
    apriori.__main__()
    pass
