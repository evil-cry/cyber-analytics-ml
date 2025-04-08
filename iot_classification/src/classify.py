#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
import tqdm
import itertools
import multiprocessing as mp
from functools import partial

# Supress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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
    def __init__(self, max_depth=None, min_node=2, feature_count=None, printf=True):
        self.root = None
        self.max_depth = max_depth
        self.min_node = min_node
        self.node_count = 0
        self.feature_count = feature_count
        self.printf = printf

    def fit(self, x, y):
        '''
        Build the decision tree to the training data
        '''

        if self.printf:
            print('\n===== BUILDING DECISION TREE =====')
            print(f'Data shape: {x.shape}')
            print(f'Max depth: {self.max_depth}')
            print(f'Min samples: {self.min_node}')
            print('================================\n')
        
        self.root = self._grow_tree(x, y)

        if self.printf:
            print('\n===== TREE CONSTRUCTION COMPLETE =====')
            print(f'Total nodes: {self.node_count}')
            print('=====================================\n')

    def _grow_tree(self, x, y, depth=0):
        '''
        Grow the decision tree recursively
        '''
        n_samples, n_features = x.shape
        n_classes = len(np.unique(y))
        self.node_count += 1
        node_id = self.node_count

        indent = '│  ' * depth + '├─'
        if self.printf:
            print(f'{indent} Node {node_id} (Depth {depth}): {n_samples} samples')

        values, counts = np.unique(y, return_counts=True)
        most_common_i = np.argmax(counts)
        most_common = values[most_common_i]

        leaf = Node(value=most_common)

        # 1. Maximum depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            if self.printf:
                print(f'Max depth {self.max_depth} reached')
            return leaf
        
        # 2. Minimum samples not reached
        if n_samples < self.min_node:
            if self.printf:
                print(f'Sample count {n_samples} below min_samples={self.min_node}')
            return leaf
        
        # 3. All samples are of the same class
        if n_classes == 1:
            if self.printf:
                print('Pure node (single class)')
            return leaf
        
        best_gini, (feature, threshold) = self.split(x, y)
        parent_gini = self.gini_index(y)

        if best_gini is None:
            if self.printf:
                print(f'{indent} Unable to find a valid split')
            return leaf

        # check if the split improves Gini impurity
        # either best gini is better than parent gini
        # or the split has no samples
        if best_gini >= parent_gini or threshold == 0:
            if self.printf:
                print(f'{indent} No improvement in Gini impurity (parent: {parent_gini:.4f}, best: {best_gini:.4f})')
            return leaf

        feature_names = [f'Feature {i}' for i in range(x.shape[1])]
        feature_name = feature_names[feature]

        if self.printf:
            print(f'{indent} SPLIT: {feature_name} <= {threshold:.4f} (Gini: {best_gini:.4f})')
 
        # 4. Check if the optimal split results in a group with no samples or
        # if the optimal split has a worse Gini impurity than the parent node
        if feature is None or threshold is None:
            return leaf

        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return leaf
        
        if self.printf:
            print(f'{indent} Left branch: {np.sum(left_mask)} samples')
            print(f'{indent} Right branch: {np.sum(right_mask)} samples')

        return Node(
            feature = feature, 
            threshold = threshold,
            left = self._grow_tree(x[left_mask], y[left_mask], depth + 1),
            right = self._grow_tree(x[right_mask], y[right_mask], depth + 1)
        )
    
    def predict(self, samples):
        '''
        Predict the class of a sample
        @params:
        sample: numpy array - samples to predict
        
        @returns: 
        predictions: numpy array - predicted classes
        '''
        if len(samples.shape) == 1:
            return self.predict_single(self.root, samples)
        else:
            return np.array([self.predict_single(self.root, sample) for sample in samples])
        
    def predict_single(self, node, sample):
        '''
        Predict the class of a single sample
        '''

        # if the node is a leaf, return its value
        if node.value is not None:
            return node.value
            
        # decide which branch to follow
        if sample[node.feature] <= node.threshold:
            return self.predict_single(node.left, sample)
        else:
            return self.predict_single(node.right, sample)
        
    def gini_index(self, labels: list) -> float:
        '''
        Calculate the Gini impurity of a sample
        '''
        if len(labels) == 0:
            return 0
        
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        
        gini = 1 - np.sum(p ** 2)
        return gini

    def gini_impurity(self, left: list, right: list) -> float:
        '''
        Calculate the Gini impurity of a set of labels.

        @params
        - grps: list 
        - cls: list

        @returns
        - gini: float
                The Gini impurity of the set of labels.
        '''
        
        n_left = len(left)
        n_right = len(right)
        n_total = n_left + n_right
        
        if n_total == 0:    
            return 0
        
        gini = (n_left / n_total) * self.gini_index(left) + \
                (n_right / n_total) * self.gini_index(right)
        
        return gini

    def split(self, samples: list, labels: list) -> int:
        '''
        Split a group of labels into two groups based on the Gini impurity.

        @params
        - grourps: left and right group of samples
        - labels: list of classes/labels
        - feature_count: number of features to select randomly

        @returns
        - b: (int, int)
                Tuple of the best feature index and threshold.
        '''
        
        n_samples, n_features = samples.shape
        
        if n_samples == 0:
            return None, (0, 0)
        
        # start best gini as infinity, since best gini is the lowest
        best_gini = float('inf')
        index_plus_threshold = None

        # randomly select features
        if self.feature_count and self.feature_count < n_features:
            feature_indices = np.random.choice(n_features, size=self.feature_count, replace=True)
        else:
            feature_indices = np.arange(n_features)

        # iterate through features to find the best split
        for index in feature_indices:
            feature = samples[:, index]
            unique = np.unique(feature)

            if len(unique) <= 1:
                continue

            split_t = (unique[:-1] + unique[1:]) / 2 

            for threshold in split_t:
                l_mask = feature <= threshold
                r_mask = ~l_mask

                left = labels[l_mask]
                right = labels[r_mask]
                
                gini = self.gini_impurity(left, right)
                
                if gini < best_gini:
                    best_gini = gini
                    index_plus_threshold = index, threshold

        if index_plus_threshold is None:
            return None, (0, 0)

        return best_gini, index_plus_threshold


def parse_args():
    '''
    Parse arguments.
    '''
    parser = argparse.ArgumentParser()

    def check_path(parser, x):
        if not os.path.exists(x):
            parser.error('That directory {} does not exist!'.format(x))
        else:
            return x
    parser.add_argument('-r', '--root', type=lambda x: check_path(parser, x), 
                        help='The path to the root directory containing feature files.')
    parser.add_argument('-s', '--split', type=float, default=0.7, 
                        help='The percentage of samples to use for training.')

    return parser.parse_args()


def load_data(root, min_samples=20, max_samples=1000):
    '''Load json feature files produced from feature extraction.

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
    '''
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
            if name.startswith('features') and name.endswith('.json'):
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
        with open(path, 'r') as fp:
            features = json.load(fp)
            instance = [features['flow_volume'],
                        features['flow_duration'],
                        features['flow_rate'],
                        features['sleep_time'],
                        features['dns_interval'],
                        features['ntp_interval']]
            X.append(instance)
            X_p.append(list(features['ports']))
            X_d.append(list(features['domains']))
            X_c.append(list(features['ciphers']))
            Y.append(label)
            domain_set.update(list(features['domains']))
            cipher_set.update(list(features['ciphers']))
            for port in set(features['ports']):
                port_dict[port] = 1 + port_dict.get(port, 0)

    # prune rarely seen ports
    port_set = set()
    for port in port_dict.keys():
        if port_dict[port] > 10:
            port_set.add(port)

    # map to wordbag
    print('Generating wordbags ... ')
    for i in tqdm.tqdm(range(len(Y))):
        X_p[i] = list(map(lambda x: X_p[i].count(x), port_set))
        X_d[i] = list(map(lambda x: X_d[i].count(x), domain_set))
        X_c[i] = list(map(lambda x: X_c[i].count(x), cipher_set))

    return np.array(X).astype(float), np.array(X_p), np.array(X_d), np.array(X_c), np.array(Y)


def classify_bayes(X_tr, Y_tr, X_ts, Y_ts):
    '''
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
    '''
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
    '''
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
    '''
    # perform multinomial classification on bag of ports
    resp_tr, resp_ts = classify_bayes(Xp_tr, Y_tr, Xp_ts, Y_ts)

    # perform multinomial classification on domain names
    resd_tr, resd_ts = classify_bayes(Xd_tr, Y_tr, Xd_ts, Y_ts)

    # perform multinomial classification on cipher suites
    resc_tr, resc_ts = classify_bayes(Xc_tr, Y_tr, Xc_ts, Y_ts)

    return resp_tr, resp_ts, resd_tr, resd_ts, resc_tr, resc_ts

def do_stage_1(X_tr, X_ts, Y_tr, Y_ts, printf = True, **kwargs):
    '''
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
    '''
    # Tree Hyperparameters
    max_depth = kwargs.get('max_depth', 10)             # maximum depth of the tree
    min_node = kwargs.get('min_node', 2)                # minimum number of samples in a node
    feature_count = kwargs.get('feature_count', None)   # number of features to sample for each tree

    # Forest Hyperparameters
    n_trees = kwargs.get('n_trees', 10)             # number of decision trees in the forest 
    per_data = kwargs.get('per_data', 0.7)          # percentage of data to use for training

    n_samples, n_features = X_tr.shape
    forest = []  # List to hold tuples - (tree, feature_indices)

    # Create each tree in the forest
    for i in range(n_trees):
        
        # Calculate the number of samples in the current tree, then randomly select training data with replacement
        sample_size = int(per_data * n_samples)
        sample_indices = np.random.choice(n_samples, size=sample_size, replace=True)
        X_sample = X_tr[sample_indices, :]
        Y_sample = Y_tr[sample_indices]

        # Build the Tree
        if printf:
            print(f'\nBuilding tree {i+1}/{n_trees} using {sample_size} samples')
        tree = DecisionTree(max_depth, min_node, feature_count, printf)

        # fit the tree
        tree.fit(X_sample, Y_sample)
        forest.append(tree)

    # Prediction:
    n_test = X_ts.shape[0]
    votes = np.zeros((n_test, n_trees), dtype=int)
    
    for i, tree in enumerate(forest):
        predictions = tree.predict(X_ts)
        votes[:, i] = predictions

    # majority vote 
    final_preds = []
    
    for i in range(n_test):
        sample_votes = votes[i, :]
        # get the count of each unique vote value
        vote_counts = np.bincount(sample_votes)
        # get the index of the highest count
        final_pred = vote_counts.argmax()
        final_preds.append(final_pred)

    # Final predictions
    final_preds = np.array(final_preds)

    return final_preds

def evaluate_hyperparameters(params, X_tr_full, X_ts_full, Y_tr, Y_ts, target_names):
    '''
    Evaluate a single set of hyperparameters
    This still builds a forest! Just uses the default value.
    '''
    max_depth, min_node, feature_count = params
    
    kwargs = {
        'max_depth': max_depth,
        'min_node': min_node,
        'feature_count': feature_count
    }
    
    # the accuracy wobbles, so take the average 
    num_runs = 4
    accuracies = []
    
    for run in range(num_runs):
        prediction = do_stage_1(X_tr_full, X_ts_full, Y_tr, Y_ts, False, **kwargs)
        report = classification_report(Y_ts, prediction, target_names=target_names, output_dict=True)
        accuracies.append(report.get('accuracy', 0))
    
    avg_accuracy = sum(accuracies) / len(accuracies)
    return (max_depth, min_node, feature_count, avg_accuracy)

def evaluate_tree_count(n_trees, best_params, X_tr_full, X_ts_full, Y_tr, Y_ts, target_names):
    '''
    Evaluate tree count
    '''
    kwargs = best_params.copy()
    kwargs['n_trees'] = n_trees
    
    # Run multiple times and average the results
    num_runs = 4
    accuracies = []
    
    for run in range(num_runs):
        prediction = do_stage_1(X_tr_full, X_ts_full, Y_tr, Y_ts, False, **kwargs)
        report = classification_report(Y_ts, prediction, target_names=target_names, output_dict=True)
        accuracies.append(report.get('accuracy', 0))
    
    avg_accuracy = sum(accuracies) / len(accuracies)
    return (n_trees, avg_accuracy)

def tune_hyperparameters(X_tr_full, X_ts_full, Y_tr, Y_ts, target_names):
    '''
    Tune hyperparameters
    '''
    print('Tuning tree hyperparameters...')
    
    # default hyperparameters
    kwargs = {
        'max_depth': 10,
        'min_node': 2,
        'feature_count': None,
    }
    
    # these were the original test values
    '''
    max_depth_values = [None, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    min_node_values = [1, 2, 4, 8, 16, 32, 64, 100]
    feature_count_values = [None, 3, 5, 10, 20, 30, 50, 75, 100]
    '''

    # these are changed based on the results of the original tests
    # it seems that max_depth values above 100 are not giving improvements (perhaps even less than 100 but 100 doesn't take too much computing)
    max_depth_values = [100, 125, 150, 200, 250, 300, 400, 500, 1000]
    # quadruple the tests
    max_depth_values.extend(max_depth_values)
    max_depth_values.extend(max_depth_values)

    min_node_values = [1]
    feature_count_values = [10]
    
    # all combinations of hyperparameters
    param_combinations = list(itertools.product(
        max_depth_values,
        min_node_values,
        feature_count_values
    ))
    
    print(f'Testing {len(param_combinations)} hyperparameter combinations...')
    
    # this creates a copy of the function with these parameters preset and static
    # we are still testing a forest, not a single tree, just using the default tree count
    worker_fn = partial(
        evaluate_hyperparameters,
        X_tr_full=X_tr_full,
        X_ts_full=X_ts_full,
        Y_tr=Y_tr,
        Y_ts=Y_ts,
        target_names=target_names
    )
    
    # USE (almost) MAXIMUM POWER
    # https://tenor.com/view/spongebob-maximum-power-power-spongebob-squarepants-full-power-gif-22777727
    cores = max(1, mp.cpu_count() - 1)
    
    best_accuracy = 0
    best_params = kwargs.copy()
    
    with mp.Pool(processes=cores) as pool:
        for max_depth, min_node, feature_count, accuracy in pool.imap_unordered(worker_fn, param_combinations):
            params = {
                'max_depth': max_depth,
                'min_node': min_node,
                'feature_count': feature_count
            }
            print(f'\tHyperparameters: {params}, Accuracy: {accuracy:.4f}')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params.copy()
    
    print(f'Found best tree hyperparameters: {best_params} with accuracy {best_accuracy:.4f}')
    
    print('Tuning forest hyperparameters using best tree hyperparameters...')

    #tree_counts = [1, 5, 10, 20, 50, 75, 100]
    # found that tree counts above 25 are not giving improvements (perhaps even less than 25 but that doesn't take too much computing)
    tree_counts = [25, 50, 75, 100] #, 125, 150, 200, 250, 300]
    
    tree_worker = partial(
        evaluate_tree_count,
        best_params=best_params,
        X_tr_full=X_tr_full,
        X_ts_full=X_ts_full,
        Y_tr=Y_tr,
        Y_ts=Y_ts,
        target_names=target_names
    )
    
    best_tree_count = 1
    
    with mp.Pool(processes=cores) as pool:
        for n_trees, accuracy in pool.imap_unordered(tree_worker, tree_counts):
            print(f'\tTree count {n_trees}: Average accuracy {accuracy:.4f}')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_tree_count = n_trees
    
    best_params['n_trees'] = best_tree_count
    print(f'Found best tree count: {best_params['n_trees']} with accuracy {best_accuracy:.4f}')
    
    return best_params

def main(args):
    '''
    Perform main logic of program
    '''
    # load dataset
    print('Loading dataset ... ')
    X, X_p, X_d, X_c, Y = load_data('iot_classification/corpus/iot_data')

    # encode labels
    print('Encoding labels ... ')
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)

    print('Dataset statistics:')
    print('\t Classes: {}'.format(len(le.classes_)))
    print('\t Samples: {}'.format(len(Y)))
    print('\t Dimensions: ', X.shape, X_p.shape, X_d.shape, X_c.shape)

    # shuffle
    print('Shuffling dataset using seed {} ... '.format(SEED))
    s = np.arange(Y.shape[0])
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, X_p, X_d, X_c, Y = X[s], X_p[s], X_d[s], X_c[s], Y[s]

    # split
    print('Splitting dataset using train:test ratio of {}:{} ... '.format(int(args.split*10), int((1-args.split)*10)))
    cut = int(len(Y) * args.split)
    X_tr, Xp_tr, Xd_tr, Xc_tr, Y_tr = X[cut:], X_p[cut:], X_d[cut:], X_c[cut:], Y[cut:]
    X_ts, Xp_ts, Xd_ts, Xc_ts, Y_ts = X[:cut], X_p[:cut], X_d[:cut], X_c[:cut], Y[:cut]

    # perform stage 0
    print('Performing Stage 0 classification ... ')
    p_tr, p_ts, d_tr, d_ts, c_tr, c_ts = \
        do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts)

    # build stage 1 dataset using stage 0 results
    # NB predictions are concatenated to the quantitative attributes processed from the flows
    X_tr_full = np.hstack((X_tr, p_tr, d_tr, c_tr))
    X_ts_full = np.hstack((X_ts, p_ts, d_ts, c_ts))

    # perform final classification
    print('Performing Stage 1 classification ... ')
    
    # found using tune_hyperparameters
    kwargs = {
        'max_depth': 100, # might be even less than 100
        'min_node': 1, # 1 is the best
        'feature_count': 10, # 10 is the best
        'n_trees': 25, # didn't change much above 25
        'per_data': 0.7 # 70% of the data is used for training
    }

    prediction = do_stage_1(X_tr_full, X_ts_full, Y_tr, Y_ts, True, **kwargs)

    # print the report
    string_report = classification_report(Y_ts, prediction, target_names=le.classes_, output_dict=False)
    print(string_report)

    # tune hyperparameters (uncomment to use)
    #hyperparameters = tune_hyperparameters(X_tr_full, X_ts_full, Y_tr, Y_ts, le.classes_)
    #print('Best hyperparameters: ', hyperparameters)

if __name__ == '__main__':
    # parse cmdline args
    args = parse_args()
    main(args)
