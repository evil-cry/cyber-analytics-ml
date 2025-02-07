import numpy as np
from collections import Counter
import utils

class _Classifier():
    '''
    Classifier class. This is an abstract class that should not be instantiated directly.
    @params:
        file_path (str): The path to the file containing the data.
        name (str): The name of the classifier.
        stop_words_per_mille (int): The bottom per mille of stop words to remove.
        min_count (int): The minimum number of times a word must appear to be included in the vocabulary.
        parameters (dict): A dictionary of parameters for the classifier.
    '''
    def __init__(self, file_path: str, stop_words_per_mille: int, min_count: int, parameters: dict):
        self.file_path = file_path
        self.stop_words_per_mille = stop_words_per_mille
        self.min_count = min_count
        self.parameters = parameters

        self.stop_words = utils.find_stop_words(file_path, stop_words_per_mille, min_count)
        self.train_data, self.test_data = utils.get_data(file_path, self.stop_words)

        self.train_labels = np.array([doc[0] for doc in self.train_data])
        self.test_labels = np.array([doc[0] for doc in self.test_data])

        self.processed_test_data = None
        self.name = "Abstract Classifier"

        self.model = {}
        self.train()
    
    def train(self) -> dict:
        '''
        Train a classifier. Sets self.model to the trained model dictionary.
        '''
        raise NotImplementedError()
    
    def classify(self, sample) -> str:
        '''
        Classify test messages as spam or ham.
        @param:
            sample - sample object. Uses different classes for different classifiers, check the correct one for your use case. 
        @returns:
            str: Prediction, either 'spam' or 'ham'
        ''' 
        raise NotImplementedError()

    def test(self) -> tuple:
        '''
        Test a classifier.
        @returns:
            tuple: A tuple containing four integers:
                - tp (int): True positives (correctly classified as spam).
                - tn (int): True negatives (correctly classified as ham).
                - fp (int): False positives (incorrectly classified as spam).
                - fn (int): False negatives (incorrectly classified as ham).
        '''
        tp = fp = tn = fn = 0

        for sample, label in zip(self.processed_test_data, self.test_labels):
                prediction = self.classify(sample)

                if prediction == 'spam' and label == 'spam':
                    tp += 1
                elif prediction == 'spam' and label == 'ham':
                    fp += 1
                elif prediction == 'ham' and label == 'ham':
                    tn += 1
                elif prediction == 'ham' and label == 'spam':
                    fn += 1

        return tp, fp, tn, fn
    
    def evaluate(self) -> tuple:
        '''
        Calculate performance statistics.
        @returns:
        tuple: A tuple containing statistics as floats:
            - accuracy (float): The accuracy of the model
            - precision (float): The precision of the model
            - recall (float): The recall of the model
            - f1 (float): The F1 score of the model
        @exceptions:
            If a ZeroDivisionError occurs during calculation, all statistics will be set to 0.
        '''
        tp, fp, tn, fn = self.test()

        try:
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall) 

        except ZeroDivisionError:
            accuracy = 0
            precision = 0
            recall = 0
            f1 = 0

        return accuracy, precision, recall, f1

class KNN(_Classifier):
    def __init__(self, *args, **kwargs):
        super(KNN, self).__init__(*args, **kwargs)

        self.k = self.parameters['k'] if 'k' in self.parameters else 7
        self.name = "K-Nearest Neighbors"

        # Get Vectors from Vecotrize and create the labels
        self.train_vectors, self.test_vectors = utils.vectorize(self.train_data, self.test_data)
        
        # Converts to Numpy arrays to make faster
        self.train_vectors = np.array(self.train_vectors)
        self.test_vectors = np.array(self.test_vectors)

        self.processed_test_data = self.test_vectors

    def train(self):
        return {}
    
    def classify(self, sample) -> str:
        # Calculates the similarity between two vectors
        def cosine_sim(vec1, vec2):
            # Calculate dot product 
            dproduct = np.dot(vec1,vec2)
            
            # Caculate Magnitudes
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # Stops "scalar divide" error from occuring
            if norm1 == 0 or norm2 == 0:
                return 0
            
            final = dproduct/(norm1*norm2)
            return final

        # Considers k neighbors
        # Compute distances between test_vector and all train_vectors
        distances = []
        for i, train_v in enumerate(self.train_vectors):
            sim = cosine_sim(sample, train_v)
            distances.append((sim, self.train_labels[i]))  
        
        # Sort by descending order of similarity, selects the top (k) neighbors
        distances.sort(reverse=True, key=lambda x: x[0])
        neighbors = distances[:self.k]
        
        # Majority vote
        spam_votes = sum(1 for _, label in neighbors if label == 'spam')
        ham_votes = self.k - spam_votes
        
        # Decide whether it is spam or ham
        if spam_votes > ham_votes:
            return 'spam'
        else:
            return 'ham'

class NB(_Classifier):
    '''
    Naive Bayes Classifier
    '''
    def __init__(self, *args, **kwargs):
        super(NB, self).__init__(*args, **kwargs)

        self.s = self.parameters['s'] if 's' in self.parameters else 4
        self.name = "Naive Bayes"
        self.processed_test_data = [message[1] for message in self.test_data]

    def train(self):
        spam_count = 0
        ham_count = 0
        spam_corpus = []
        ham_corpus = []

        # Determine the total number of spam and ham messages
        for document in self.train_data:
            classification = document[0]
            if classification == 'spam':
                spam_count += 1
                spam_corpus.append(document[1])

            elif classification == 'ham':
                ham_count += 1
                ham_corpus.append(document[1])

            else:
                raise ValueError('Unknown classification')

        # Seperate the spam and ham messages into individual collections
        w_spam = [word for msg in spam_corpus for word in msg]
        w_ham = [word for msg in ham_corpus for word in msg]

        vocab = set(w_spam + w_ham)

        # Get count of each word in spam and ham messages
        w_spam_count = Counter(w_spam)
        w_ham_count = Counter(w_ham)

        # Instead of setting to 0, set to the log of the class prior
        # Class prior is the base probability of each class
        p_sample_spam = np.log(spam_count / len(self.train_data))
        p_sample_ham = np.log(ham_count / len(self.train_data))

        # Calculate the probability of the sample being spam or ham
        self.model = {
            'w_spam': w_spam,
            'w_ham': w_ham,
            'vocab': vocab,
            'w_spam_count': w_spam_count,
            'w_ham_count': w_ham_count,
            'p_sample_spam': p_sample_spam,
            'p_sample_ham': p_sample_ham
        }
    
    def classify(self, sample):
        w_spam = self.model['w_spam']
        w_ham = self.model['w_ham']
        vocab = self.model['vocab']
        w_spam_count = self.model['w_spam_count']
        w_ham_count = self.model['w_ham_count']
        p_sample_spam = self.model['p_sample_spam']
        p_sample_ham = self.model['p_sample_ham']

        for word in sample:        
            # Calculate the P(W_i|Spam) and P(W_i|Ham)
            # Apply smoothing = add s to the numerator and 2 * s to the denominator
            # Multiply by 2 because there are two categories
            p_word_spam = ((w_spam_count[word]) + self.s) / (len(w_spam) + self.s * len(vocab))
            p_word_ham = ((w_ham_count[word]) + self.s) / (len(w_ham) + self.s * len(vocab))

            # Use log instead
            log_p_word_spam = np.log(p_word_spam)
            log_p_word_ham = np.log(p_word_ham)

            # Add to total log probabilities
            p_sample_spam += log_p_word_spam
            p_sample_ham += log_p_word_ham

        if p_sample_spam > p_sample_ham:
            return 'spam'
        else:
            return 'ham'