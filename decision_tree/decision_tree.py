import numpy as np
import pandas as pd

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine


class Node:
    def __init__(self):
        self.attribute = None
        self.children = dict()
        self.default = None


class DecisionTree:
    def __init__(self):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.root = Node()

        self.root = Node()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Generates a decision tree for classification

        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y):
        node = Node()

        counts = np.bincount(y)
        node.default = np.argmax(counts)

        if len(np.unique(y)) != 1:
            gains = np.array(
                [self._information_gain_categorical(y, X[attr]) for attr in X.columns]
            )

            max_gain_attr = np.argmax(gains)

            if np.isnan(gains[max_gain_attr]):
                return node

            node.attribute = X.columns[max_gain_attr]

            for attr_val, partition in X.groupby(node.attribute):
                X_subset = partition.drop(columns=node.attribute)
                y_subset = y[X_subset.index]
                node.children[attr_val] = self._build_tree(X_subset, y_subset)

        return node

    def predict(self, X: pd.DataFrame):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.

        Returns:
            A length m vector with predictions
        """
        return np.array([self._predict_instance(inst) for _, inst in X.iterrows()])

    def _predict_instance(self, instance):
        node = self.root
        while node.attribute:
            if instance[node.attribute] in node.children:
                node = node.children[instance[node.attribute]]
            else:
                break
        return node.default

    def get_rules(self):
        """
        Returns the decision tree as a list of rules

        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label

            attr1=val1 ^ attr2=val2 ^ ... => label

        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        return self._generate_rules(self.root, rule=[])

    def _generate_rules(self, node, rule):
        if not node.attribute:
            yield (rule, node.default)
        else:
            for val, child in node.children.items():
                yield from self._generate_rules(child, rule + [(node.attribute, val)])

    def _information_gain(self, y, X_attr):
        counts_total = np.bincount(y)
        entropy_total = entropy(counts_total)

        N = len(y)

        entropy_attr = 0
        for attr_val, bincount in X_attr.groupby(X_attr):
            counts_attr = np.bincount(y[bincount.index])
            entropy_attr += len(bincount) / N * entropy(counts_attr)
        gain = entropy_total - entropy_attr

        return gain


# --- Some utility functions


def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy

    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning

    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0

    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.

    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))
