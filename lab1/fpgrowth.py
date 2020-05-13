"""FP growth python implementation

Homework of IoT Information processing Lab1 . A simple implementation
of Fp-growth algorithm.

Author: Hu Yongjian
License: MIT License
"""

import math
from collections import defaultdict
import pandas as pd
import time
from itertools import chain, combinations


def read_file(file_path):
    """
    Read csv data file from disk.

    Args:
        file_path (str): The path to the data file.

    Returns:
        (list[set]) The transactions data list.
    """
    transaction_list = list()
    df = pd.read_csv(file_path)
    raw_data = df.iloc[:, 1].values.tolist()

    for item in raw_data:
        item = item.strip('{}')
        tmp = item.split(",")
        transaction_list.append(set(tmp))

    return transaction_list


def gen_item_support(transaction_list, item_set):
    """
    Generate support for each item in transaction list.

    Args:
        transaction_list (list[set]): Transaction list.
        item_set (set): Item set.

    Returns:
        (dict) Dictionary for each item with support.
    """
    item_set_support = dict.fromkeys(item_set, 0)
    for transaction in transaction_list:
        for item in item_set:
            if item.issubset(transaction):
                item_set_support[item] += 1

    return item_set_support


class FPTree:
    """
    FP Tree class.

    Attributes:
        root (FPNode): Root node.
        header_table (dict[frozenset]): Header table.
        self.node_list (list): List of linked node list.
    """

    def __init__(self, transaction_list, min_support):
        """
        Initialize FPTree.

        Args:
            transaction_list (list[set]): List of transactions.
            min_support (int): Minimal support.
        """
        self.root = FPNode(None)
        self.header_table = self._gen_header_table(transaction_list, min_support)
        self.node_list = self._insert_tree(transaction_list)

    def _gen_header_table(self, transaction_list, min_support):
        """
        Generate header table.

        Args:
            transaction_list (list[set]): List of transactions.
            min_support (int): Minimal support.

        Returns:
            (dict[set]): Header table.
        """
        item_set = self._gen_item_set(transaction_list)
        header_table = gen_item_support(transaction_list, item_set)

        # Remove infrequent items
        for item in list(header_table.keys()):
            if header_table[item] < min_support:
                del (header_table[item])

        header_table = {k: v for k, v in sorted(header_table.items(), key=lambda x: (x[1], list(x[0])), reverse=True)}
        return header_table

    def _gen_item_set(self, transaction_list):
        """
        Generate item sets from transaction list.

        Args:
            transaction_list (list[set]): List of transactions.

        Returns:
            (set): Set of items.
        """
        items = set()
        for transaction in transaction_list:
            for item in transaction:
                items.add(frozenset([item]))

        return items

    def _insert_tree(self, transaction_list, count=1):
        """
        Construct Fp tree, and generates linked list of nodes.

        Args:
            transaction_list (list[set]): List of transactions.
            count (int): Count

        Returns:
            Node list.
        """
        node = self.root
        node_list = defaultdict(list)
        for key, value in node_list.items():
            print(key, value)
        for transaction in transaction_list:
            for item in self.header_table.keys():
                if item.issubset(transaction):
                    if item in node.children:
                        child = node.children[item]
                        child.count += count
                        node = child
                    else:
                        new_child_node = FPNode(item, count, node)
                        node_list[item].append(new_child_node)
                        node.children[item] = new_child_node
                        node = new_child_node

            node = self.root
        return node_list

    def is_path(self):
        """
        Decides if the tree is a path.

        Returns:
            True if it is a path, False otherwise.
        """
        if len(self.root.children) > 1:
            return False
        for item, node in self.node_list.items():
            if len(node) > 1 or len(node[0].children) > 1:
                return False
        return True


# def print_node_list(self, node_list):
#     """
#     This method prints node list from a tree.
#
#     Args:
#         node_list (dict[FPNode]): Node list
#     """
#     for key, value in node_list.items():
#         print("The key is " + str(key))
#         print("The node list is ")
#         for item in value:
#             print(item.item, item.count, end=" -> ")
#         print()


class FPNode:
    """
    FP tree node.

    Attributes:
        item (frozenset)
        count (int)
        parent (FPNode)
        children (dict[FPNode])
    """

    def __init__(self, node_item=None, count=0, parent=None):
        """
        Initialize Fp node.

        Args:
            node_item : Node' s item.
            count (int): The count of node' s number.
            parent (FPNode): The parent node of current node.
        """
        self.item = node_item
        self.count = count
        self.parent = parent
        self.children = defaultdict(FPNode)

    def get_path(self):
        """
        Get path from root to current node.

        Returns:
            Path to current node.
        """
        path = []
        if self.item is None:
            return path

        tmp_node = self.parent
        while tmp_node.item is not None:
            path.append(tmp_node.item)
            tmp_node = tmp_node.parent

        path.reverse()
        return path


def fp_growth(tree, conditional_pattern=None, minimal_support=3):
    """
    Fp growth, recursive algorithm.

    Args:
        tree (FPTree) : Fp tree.
        minimal_support (int): Minimal support
        conditional_pattern (list): List of conditional pattern base

    Returns:
        FP list, a list contains all frequent patterns with support.
    """
    if conditional_pattern is None:
        conditional_pattern = []
    fp_list = list()
    items = tree.header_table.keys()
    length = len(items)
    if tree.is_path():
        # If tree is path, just generate all sub subset of current frequent
        # item set. No need to generate conditional tree.
        for i in range(1, length + 1):
            for item_set in combinations(items, i):
                support = min([tree.node_list[node][0].count] for node in item_set)

                tmp_list = list()
                for x in list(item_set):
                    tmp_list.extend(list(x))

                pattern = conditional_pattern + tmp_list
                fp_list.append((pattern, support[0]))
    else:
        for item in items:
            support = sum([node.count for node in tree.node_list[item]])
            pattern = conditional_pattern + list(item)
            fp_list.append((pattern, support))

            # Construct conditional tree.
            # First, generate sub transaction list.
            subtransaction_list = list()
            tmp_list = list()
            for node in tree.node_list[item]:
                for _ in range(node.count):
                    tmp_list.append(node.get_path())

            # Process the list, including removing empty set,
            # to get final usable sub transaction list
            for i in range(len(tmp_list)):
                tmp_list[i] = [list(x) for x in tmp_list[i]]

            for i in range(len(tmp_list)):
                tmp = set()
                for item in tmp_list[i]:
                    if item != []:
                        tmp.update(item)
                if len(tmp) is not 0:
                    subtransaction_list.append(tmp)

            # After getting complete sub transaction list,
            # generates conditional tree.
            conditional_tree = FPTree(subtransaction_list, minimal_support)

            # Recursive step
            if conditional_tree.header_table:
                fp_list.extend(fp_growth(conditional_tree, pattern, min_support))

    return fp_list


def print_result(fp_list, length):
    """
    Show result, including frequent item set and association rules.
    """
    print()
    print("Frequent item set with support")
    for k, v in sorted(fp_list, key=lambda x: x[1], reverse=True):
        print("{} --:-- {:.2f}".format(k, v / length))

    print()
    print("Association rules with confidence:")
    for item, confidence in sorted(rules, key=lambda x: (x[1], x[0]), reverse=True):
        print("{} --> {} --:-- {:.2f}".format(str(item[0]), str(item[1]), confidence))


def powerset(iterable):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def get_support(item, item_set):
    """
    Get support of specified item.

    Args:
        item (frozenset): The specified item.
        item_set (list[tuple]): The frequent item set with support.

    Returns:
        Support of specified item.
    """
    for key, value in item_set:
        if frozenset(key) == item:
            return value


def get_rules(item_set, min_confidence):
    """
    Generate association rules from frequent item set.

    Args:
        item_set (list[tuple]): Frequent item set.
        min_confidence: Minimal confidence.

    Returns:
        The association rules.
    """
    rules = list()

    for key, value in item_set:
        key = set(key)
        subset = map(frozenset, [item for item in powerset(key)])
        for item in subset:
            remaining = key.difference(item)
            if len(remaining) > 0:
                confidence = value / get_support(item, fp_list)
                if confidence >= min_confidence:
                    rules.append(((tuple(item), tuple(remaining)), confidence))
    return rules


if __name__ == '__main__':
    # Set up parameters
    transaction_list = read_file("Groceries.csv")
    relative_support = 0.05
    min_support = math.ceil(relative_support * len(transaction_list))
    min_confidence = 0.3

    # Start
    t0 = time.perf_counter()
    tree = FPTree(transaction_list, min_support)
    fp_list = fp_growth(tree, None, min_support)
    rules = get_rules(fp_list, min_confidence)
    t1 = time.perf_counter()
    # Finish
    print("Time elapsed: ", (t1 - t0) * 1000)

    # Print result
    print_result(fp_list, len(transaction_list))
