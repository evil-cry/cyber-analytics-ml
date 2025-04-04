#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
import tqdm

# Supress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# seed value
# (ensures consistent dataset splitting between runs)
SEED = 0

class Node:
    '''
    Represents a node in a decision tree
    '''

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    '''
    Represents a decision tree classifier
    '''

    def __init__(self, max_depth=None, min_node=2):
        self.root = None
        self.max_depth = max_depth
        self.min_node = min_node
        self.node_count = 0

    def fit(self, x, y):
        '''
        Build the decision tree to the training data
        '''

        print("\n===== BUILDING DECISION TREE =====")
        print(f"Data shape: {x.shape}")
        print(f"Max depth: {self.max_depth}")
        print(f"Min samples: {self.min_node}")
        print("================================\n")
        
        self.root = self._grow_tree(x, y)

        print("\n===== TREE CONSTRUCTION COMPLETE =====")
        print(f"Total nodes: {self.node_count}")
        print("=====================================\n")

    def _grow_tree(self, x, y, depth=0):
        '''
        Grow the decision tree recursively
        '''
        n_samples, n_features = x.shape
        n_classes = len(np.unique(y))
        self.node_count += 1
        node_id = self.node_count

        indent = "│  " * depth + "├─"
        print(f"{indent} Node {node_id} (Depth {depth}): {n_samples} samples")

        values, counts = np.unique(y, return_counts=True)
        most_common_i = np.argmax(counts)
        most_common = values[most_common_i]

        leaf = Node(value=most_common)

        # 1. Maximum depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            print(f"Max depth {self.max_depth} reached")
            return leaf
        
        # 2. Minimum samples not reached
        if n_samples < self.min_node:
            print(f"Sample count {n_samples} below min_samples={self.min_node}")
            return leaf
        
        # 3. All samples are of the same class
        if n_classes == 1:
            print("Pure node (single class)")
            return leaf
        
        split_result = split(x, y)
        if split_result[0] is None:
            return leaf  # Return a leaf if no valid split is found
        best_gini, (feature, threshold) = split_result

        feature_names = [f"Feature {i}" for i in range(x.shape[1])]
        feature_name = feature_names[feature]

        print(f"{indent} SPLIT: {feature_name} <= {threshold:.4f} (Gini: {best_gini:.4f})")
 
        # 4. Check if the optimal split results in a group with no samples or
        # if the optimal split has a worse Gini impurity than the parent node
        if feature is None or threshold is None:
            return leaf

        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return leaf
        
        print(f"{indent} Left branch: {np.sum(left_mask)} samples")
        print(f"{indent} Right branch: {np.sum(right_mask)} samples")

        return Node(
            feature = feature, 
            threshold = threshold,
            left = self._grow_tree(x[left_mask], y[left_mask], depth + 1),
            right = self._grow_tree(x[right_mask], y[right_mask], depth + 1)
        )


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    def check_path(parser, x):
        if not os.path.exists(x):
            parser.error("That directory {} does not exist!".format(x))
        else:
            return x
    parser.add_argument('-r', '--root', type=lambda x: check_path(parser, x), 
                        help='The path to the root directory containing feature files.')
    parser.add_argument('-s', '--split', type=float, default=0.7, 
                        help='The percentage of samples to use for training.')

    return parser.parse_args()


def load_data(root, min_samples=20, max_samples=1000):
    """Load json feature files produced from feature extraction.

    The device label (MAC) is identified from the directory in which the feature file was found.
    Returns X and Y as separate multidimensional arrays.
    The instances in X contain only the first 6 features.
    The ports, domain, and cipher features are stored in separate arrays for easier process in stage 0.

    Parameters
    ----------
    root : str
           Path to the directory containing samples.
    min_samples : int
                  The number of samples each class must have at minimum (else it is pruned).
    max_samples : int
                  Stop loading samples for a class when this number is reached.

    Returns
    -------
    features_misc : numpy array
    features_ports : numpy array
    features_domains : numpy array
    features_ciphers : numpy array
    labels : numpy array
    """
    X = []
    X_p = []
    X_d = []
    X_c = []
    Y = []

    port_dict = dict()
    domain_set = set()
    cipher_set = set()

    # create paths and do instance count filtering
    fpaths = []
    fcounts = dict()
    for rt, dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(rt, fname)
            label = os.path.basename(os.path.dirname(path))
            name = os.path.basename(path)
            if name.startswith("features") and name.endswith(".json"):
                fpaths.append((path, label, name))
                fcounts[label] = 1 + fcounts.get(label, 0)

    # load samples
    processed_counts = {label:0 for label in fcounts.keys()}
    for fpath in tqdm.tqdm(fpaths):
        path = fpath[0]
        label = fpath[1]
        if fcounts[label] < min_samples:
            continue
        if processed_counts[label] >= max_samples:
            continue
        processed_counts[label] += 1
        with open(path, "r") as fp:
            features = json.load(fp)
            instance = [features["flow_volume"],
                        features["flow_duration"],
                        features["flow_rate"],
                        features["sleep_time"],
                        features["dns_interval"],
                        features["ntp_interval"]]
            X.append(instance)
            X_p.append(list(features["ports"]))
            X_d.append(list(features["domains"]))
            X_c.append(list(features["ciphers"]))
            Y.append(label)
            domain_set.update(list(features["domains"]))
            cipher_set.update(list(features["ciphers"]))
            for port in set(features["ports"]):
                port_dict[port] = 1 + port_dict.get(port, 0)

    # prune rarely seen ports
    port_set = set()
    for port in port_dict.keys():
        if port_dict[port] > 10:
            port_set.add(port)

    # map to wordbag
    print("Generating wordbags ... ")
    for i in tqdm.tqdm(range(len(Y))):
        X_p[i] = list(map(lambda x: X_p[i].count(x), port_set))
        X_d[i] = list(map(lambda x: X_d[i].count(x), domain_set))
        X_c[i] = list(map(lambda x: X_c[i].count(x), cipher_set))

    return np.array(X).astype(float), np.array(X_p), np.array(X_d), np.array(X_c), np.array(Y)


def classify_bayes(X_tr, Y_tr, X_ts, Y_ts):
    """
    Use a multinomial naive bayes classifier to analyze the 'bag of words' seen in the ports/domain/ciphers features.
    Returns the prediction results for the training and testing datasets as an array of tuples in which each row
      represents a data instance and each tuple is composed as the predicted class and the confidence of prediction.

    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    C_tr : numpy array
           Prediction results for training samples.
    C_ts : numpy array
           Prediction results for testing samples.
    """
    classifier = MultinomialNB()
    classifier.fit(X_tr, Y_tr)

    # produce class and confidence for training samples
    C_tr = classifier.predict_proba(X_tr)
    C_tr = [(np.argmax(instance), max(instance)) for instance in C_tr]

    # produce class and confidence for testing samples
    C_ts = classifier.predict_proba(X_ts)
    C_ts = [(np.argmax(instance), max(instance)) for instance in C_ts]

    return C_tr, C_ts


def do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts):
    """
    Perform stage 0 of the classification procedure:
        process each multinomial feature using naive bayes
        return the class prediction and confidence score for each instance feature

    Parameters
    ----------
    Xp_tr : numpy array
           Array containing training (port) samples.
    Xp_ts : numpy array
           Array containing testing (port) samples.
    Xd_tr : numpy array
           Array containing training (port) samples.
    Xd_ts : numpy array
           Array containing testing (port) samples.
    Xc_tr : numpy array
           Array containing training (port) samples.
    Xc_ts : numpy array
           Array containing testing (port) samples.
    Y_tr : numpy array
           Array containing training labels.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    resp_tr : numpy array
              Prediction results for training (port) samples.
    resp_ts : numpy array
              Prediction results for testing (port) samples.
    resd_tr : numpy array
              Prediction results for training (domains) samples.
    resd_ts : numpy array
              Prediction results for testing (domains) samples.
    resc_tr : numpy array
              Prediction results for training (cipher suites) samples.
    resc_ts : numpy array
              Prediction results for testing (cipher suites) samples.
    """
    # perform multinomial classification on bag of ports
    resp_tr, resp_ts = classify_bayes(Xp_tr, Y_tr, Xp_ts, Y_ts)

    # perform multinomial classification on domain names
    resd_tr, resd_ts = classify_bayes(Xd_tr, Y_tr, Xd_ts, Y_ts)

    # perform multinomial classification on cipher suites
    resc_tr, resc_ts = classify_bayes(Xc_tr, Y_tr, Xc_ts, Y_ts)

    return resp_tr, resp_ts, resd_tr, resd_ts, resc_tr, resc_ts


def gini_impurity(left: list, right: list) -> float:
    '''
    Calculate the Gini impurity of a set of labels.

    @params
    - grps: list 
    - cls: list

    @returns
    - gini: float
            The Gini impurity of the set of labels.
    '''
     
    def gini_index(labels: list) -> float:
        '''
        Calculate the Gini impurity of a sample
        '''
        if len(labels) == 0:
            return 0
        
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        
        gini = 1 - np.sum(p ** 2)
        return gini
     
    n_left = len(left)
    n_right = len(right)
    n_total = n_left + n_right
     
    if n_total == 0:    
        return 0
      
    gini = (n_left / n_total) * gini_index(left) + \
            (n_right / n_total) * gini_index(right)
      
    return gini

def split(samples: list, labels: list) -> int:
    '''
    Split a group of labels into two groups based on the Gini impurity.

    @params
    - grourps: left and right group of samples
    - labels: list of classes/labels

    @returns
    - b: (int, int)
            Tuple of the best feature index and threshold.
    '''
      
    n_samples, n_features = samples.shape
      
    if n_samples == 0:
        return None, 0
      
    best_gini = float("inf")
    b = None

    for i in range(n_features):
        f = samples[:, i]
        u = np.unique(f)

        if len(u) <= 1:
            continue

        split_t = (u[:-1] + u[1:]) / 2 

        for t in split_t:
            l_mask = f <= t
            r_mask = ~l_mask

            left = labels[l_mask]
            right = labels[r_mask]
            
            gini = gini_impurity(left, right)
            
            if gini < best_gini:
                best_gini = gini
                b = (i, t)

    if b is None:
        return None, None

    return best_gini, b

def do_stage_1(X_tr, X_ts, Y_tr, Y_ts):
    """
    Random Forest Classifier
    @params
    - X_tr : numpy array
             Array containing training samples.
    - Y_tr : numpy array
             Array containing training labels.
    - X_ts : numpy array
             Array containing testing samples.
    - Y_ts : numpy array
             Array containing testing labels
    Returns
    -------
    final_preds : numpy array
                  Final predictions on testing dataset.
    """
    # Hyperparameters
    n_trees = 10          # Total number of decision trees in the forest 
    per_data = 0.7        # Use 70% of the training data
    feature_subcount = 3  # Number of features to sample for each tree

    n_samples, n_features = X_tr.shape
    forest = []  # List to hold tuples - (tree, feature_indices)

    # Create each tree in the forest
    for i in range(n_trees):
        
        # Calculate the number of samples in the current tree, then randomly select training data
        sample_size = int(per_data * n_samples)
        sample_indices = np.random.choice(n_samples, size=sample_size, replace=True)
        X_sample = X_tr[sample_indices, :]
        Y_sample = Y_tr[sample_indices]

        # Randomly sample feature indices to use for this tree
        feature_indices = np.random.choice(n_features, size=feature_subcount, replace=True)
        X_sample_sub = X_sample[:, feature_indices]

       # Build the Tree and train it
        print(f"\nBuilding tree {i+1}/{n_trees} using {sample_size} samples and feature subset {feature_indices}")
        tree = DecisionTree(max_depth=10, min_node=2)
        tree.fit(X_sample_sub, Y_sample)
        forest.append((tree, feature_indices))

    # Prediction:
    n_test = X_ts.shape[0]
    votes = np.zeros((n_test, n_trees), dtype=int)
    
    for i, (tree, feat_idx) in enumerate(forest):
        # Restrict test data to the feature subset for only this tree
        X_test_sub = X_ts[:, feat_idx]
        
        tree_preds = []
        for x in X_test_sub:
            # Start at the root node
            node = tree.root
            
            while node.left is not None or node.right is not None:
                
                if node.feature is None or node.threshold is None:
                    break
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            tree_preds.append(node.value)
        
    votes[:, i] = np.array(tree_preds)

    # Majority vote 
    final_preds = []
    
    # For each test sample, get the vote
    for i in range(n_test):
        sample_votes = votes[i, :]
        final_pred = np.bincount(sample_votes).argmax()
        final_preds.append(final_pred)

    # Final predictions
    final_preds = np.array(final_preds)

    return final_preds

def main(args):
    """
    Perform main logic of program
    """
    # load dataset
    print("Loading dataset ... ")
    X, X_p, X_d, X_c, Y = load_data('iot_classification/corpus/iot_data')

    # encode labels
    print("Encoding labels ... ")
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)

    print("Dataset statistics:")
    print("\t Classes: {}".format(len(le.classes_)))
    print("\t Samples: {}".format(len(Y)))
    print("\t Dimensions: ", X.shape, X_p.shape, X_d.shape, X_c.shape)

    # shuffle
    print("Shuffling dataset using seed {} ... ".format(SEED))
    s = np.arange(Y.shape[0])
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, X_p, X_d, X_c, Y = X[s], X_p[s], X_d[s], X_c[s], Y[s]

    # split
    print("Splitting dataset using train:test ratio of {}:{} ... ".format(int(args.split*10), int((1-args.split)*10)))
    cut = int(len(Y) * args.split)
    X_tr, Xp_tr, Xd_tr, Xc_tr, Y_tr = X[cut:], X_p[cut:], X_d[cut:], X_c[cut:], Y[cut:]
    X_ts, Xp_ts, Xd_ts, Xc_ts, Y_ts = X[:cut], X_p[:cut], X_d[:cut], X_c[:cut], Y[:cut]

    # perform stage 0
    print("Performing Stage 0 classification ... ")
    p_tr, p_ts, d_tr, d_ts, c_tr, c_ts = \
        do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts)

    # build stage 1 dataset using stage 0 results
    # NB predictions are concatenated to the quantitative attributes processed from the flows
    X_tr_full = np.hstack((X_tr, p_tr, d_tr, c_tr))
    X_ts_full = np.hstack((X_ts, p_ts, d_ts, c_ts))

    # perform final classification
    print("Performing Stage 1 classification ... ")
    pred = do_stage_1(X_tr_full, X_ts_full, Y_tr, Y_ts)

    # print classification report
    print(classification_report(Y_ts, pred, target_names=le.classes_))


if __name__ == "__main__":
    # parse cmdline args
    args = parse_args()
    main(args)
