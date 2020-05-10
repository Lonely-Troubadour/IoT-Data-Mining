"""Apriori python implementation

Homework of IoT Information processing. A simple implementation
of Apriori algorithm.

Author: Hu Yongjian
License: GNU GPLv3
"""

import pandas as pd
import argparse
import time
from itertools import chain, combinations

class Apriori:
    """
    Apriori algorithm

    Attributes:
        transaction_list: The list of all transactions.
        min_support: The minimal support.
        min_confidence: The minimal confidence,
        t_length: The length of transactions.
        item_set: All items' set.
        frequent_item_set: List of frequent k-item set
        frequent_seet: List of frequent k-item set with support
        rules: List of association rules.
    """
    def __init__(self, transaction_list, min_support=0.05, min_confidence=0.3):
        """
        Initialize the Apriori algorithm.

        Args:
            transaction_list (:obj:`list` of :obj:`set`): The list of all transactions.
            min_support (float): The minimal relative support.
            min_confidence (float): The minimal confidence.
        """
        self.transaction_list = transaction_list
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.t_length = len(transaction_list)
        self.item_set = set()
        self.frequent_item_set = list()
        self.frequent_set = list()
        self.rules = list()

    def __main__(self):
        # Initialize. Generate item set from transaction lists
        self.gen_item_set()

        # Get frequent 1-item set
        tmp_set, tmp_item_set = self.gen_frequent_item_set(self.item_set)

        # Iterates
        k = 2
        while tmp_item_set != set():
            self.frequent_set.append(tmp_set)
            self.frequent_item_set.append(tmp_item_set)
            candidate_item_set = self.gen_candidate_item_set(tmp_item_set)
            tmp_set, tmp_item_set = self.gen_frequent_item_set(candidate_item_set)
            k += 1

        # Generates association rules
        self.gen_rules()

        # Show results
        self.get_result()

    def gen_item_set(self):
        """Generates full item set from transaction list"""
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
        frequent_set = dict()
        candidate = dict.fromkeys(items, 0)
        item_set = set()

        for t in self.transaction_list:
            for item in items:
                if item.issubset(t):
                    candidate[item] += 1

        for item, count in candidate.items():
            support = count / self.t_length
            if support >= self.min_support:
                frequent_set[item] = support
                item_set.add(item)

        return frequent_set, item_set

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

    def get_support(self, item):
        """
        Get support of specified item from frequent set.

        Args:
            item: The item from which gets support.

        Returns:
            The support of item.
        """
        for frequent_set_dict in self.frequent_set:
            if item in frequent_set_dict:
                return frequent_set_dict[item]

    def gen_rules(self):
        """
        Generate rules from all frequent sets.

        Returns:
            The rules list of tuples
        """
        for i in range(1, len(self.frequent_set)):
            for key, value in self.frequent_set[i].items():
                subset = map(frozenset, [item for item in powerset(key)])
                for item in subset:
                    remaining = key.difference(item)
                    if len(remaining) > 0:
                        confidence = self.frequent_set[i][key] / self.get_support(item)
                        if confidence >= self.min_confidence:
                            self.rules.append(((tuple(item), tuple(remaining)), confidence))

    def get_result(self):
        """Print result, including frequent item set with support and association rules."""
        for i in range(len(self.frequent_item_set)):
            print("The frequent {}-item set is {}".format((i+1), sorted(
                [sorted(list(x)) for x in self.frequent_item_set[i]])))

        sorted_frequent_set = []
        for frequent_set_ in self.frequent_set:
            sorted_frequent_set.extend([(sorted(tuple(key)), value) for key, value in frequent_set_.items()])

        print()
        print("The frequent item set with support:")
        for item, support in sorted(sorted_frequent_set, key=lambda x: (x[1], x[0]), reverse=True):
            print("The {} item set with support: {:.2f}".format(str(item), support))

        print()
        print("The association rules with confidence:")
        for item, confidence in sorted(self.rules, key=lambda x: (x[1], x[0]), reverse=True):
            print("The {} -> {} with confidence: {:.2f}".format(str(item[0]), str(item[1]), confidence))


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
    file_path = "Groceries.csv"
    apriori = Apriori(read_file(file_path))
    apriori.__main__()
