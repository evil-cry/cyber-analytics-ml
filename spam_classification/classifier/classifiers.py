import numpy as np
from collections import Counter

class KNN():
    def __init__(self, stop_words_per_mille: int = 500, min_count: int = 0, parameters: dict = {'k': 7}):
        self.stop_words_per_mille = stop_words_per_mille
        self.min_count = min_count
        self.parameters = parameters
        
    def knn(self, train_vectors: np.array, train_labels: np.array, test_vector: np.array, k: int = 7) -> str:
        '''
        K-Nearest Neighbors classifier that determines if a given message is spam or ham..
        @params:
            train_vectors (np.array): Array of training vectors.
            train_labels (np.array): Array of labels corresponding to the training vectors.
            test_vector (np.array): The vector to classify.
            k (int): The number of nearest neighbors to consider. Defaults to 7.
        @returns:
            str: Prediction, either 'spam' or 'ham'
        '''
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
        for i, train_vec in enumerate(train_vectors):
            sim = cosine_sim(test_vector, train_vec)
            distances.append((sim, train_labels[i]))  
        
        # Sort by descending order of similarity, selects the top (k) neighbors
        distances.sort(reverse=True, key=lambda x: x[0])
        neighbors = distances[:k]
        
        # Majority vote
        spam_votes = sum(1 for _, label in neighbors if label == 'spam')
        ham_votes = k - spam_votes
        
        # Decide whether it is spam or ham
        if spam_votes > ham_votes:
            return 'spam'
        else:
            return 'ham'
        
    def test_knn(self, train_data: list, test_data: list, k=5) -> tuple:
        '''
        Tests a K-Nearest Neighbor classifier.
        @params:
            train_data (list): A list of training data where each element is a tuple containing the classification ('spam' or 'ham') and the message.
            test_data (list): A list of test data where each element is a tuple containing the classification ('spam' or 'ham') and the message.
        @returns::
            tuple: A tuple containing four integers:
                - tp (int): True positives (correctly classified as spam).
                - tn (int): True negatives (correctly classified as ham).
                - fp (int): False positives (incorrectly classified as spam).
                - fn (int): False negatives (incorrectly classified as ham).
        '''
        tp = tn = fp = fn = 0
        
        # Get Vectors from Vecotrize and create the labels
        train_vectors, test_vectors = utils.vectorize(train_data, test_data)
        train_labels = np.array([doc[0] for doc in train_data])
        test_labels = np.array([doc[0] for doc in test_data])
        
        # Converts to Numpy arrays to make faster
        train_vectors = np.array(train_vectors)
        test_vectors = np.array(test_vectors)
        
        # Classify each instance
        for test_v, label in zip(test_vectors, test_labels):
            prediction = self.knn(train_vectors, train_labels, test_v, k)
            
            if prediction == 'spam' and label == 'spam':
                tp += 1
            elif prediction == 'spam' and label == 'ham':
                fp += 1
            elif prediction == 'ham' and label == 'ham':
                tn += 1
            elif prediction == 'ham' and label == 'spam':
                fn += 1

        return tp, tn, fp, fn
        
def nb(corpus: list, sample: str, s: float) -> str:
    '''
    Naive Bayes classifier that determines if a given message is spam or ham.
    @params
        corpus (list): A list of tuples with the classification ('spam' or 'ham') and the message.
        sample (str): The message to be classified.
        s (float): laplace smoothing alpha.

    @returns:
        str: Prediction - 'spam', 'ham', or 'unknown'.
    '''
    spam_count = 0
    ham_count = 0
    spam_corpus = []
    ham_corpus = []

    # Determine the total number of spam and ham messages
    for document in corpus:
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
    p_sample_spam = np.log(spam_count / len(corpus))
    p_sample_ham = np.log(ham_count / len(corpus))

    # Calculate the probability of the sample being spam or ham
    for word in sample:        
        # Calculate the P(W_i|Spam) and P(W_i|Ham)
        # Apply smoothing = add s to the numerator and 2 * s to the denominator
        # Multiply by 2 because there are two categories
        p_word_spam = ((w_spam_count[word]) + s) / (len(w_spam) + s * len(vocab))
        p_word_ham = ((w_ham_count[word]) + s) / (len(w_ham) + s * len(vocab))

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