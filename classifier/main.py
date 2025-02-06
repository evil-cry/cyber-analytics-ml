from collections import Counter
import numpy as np
import pandas as pd
import string
import re

def vectorize(train_data: list, test_data: list):
    '''
    Vectorizes training and testing data using TF-IDF (Term Frequency-Inverse Document Frequency).
    @params:
        train_data (list): A list of tuples where each tuple contains a document identifier and a list of words in the document.
        test_data (list): A list of tuples where each tuple contains a document identifier and a list of words in the document.
    @returns:
        tuple: A tuple containing two lists:
            - train_feature_vectors (list): A list of TF-IDF feature vectors for the training data.
            - test_feature_vectors (list): A list of TF-IDF feature vectors for the test data.
    '''
    # Build vocabulary from training data
    vocabulary = set()
    for document in train_data:
        vocabulary.update(document[1])

    # Sort the vocab list for constistency
    vocab_list = sorted(vocabulary)
    
    # Calculate number of texts with each term (frequency)
    doc_freq = {term: 0 for term in vocab_list}
    for doc in train_data:
        unique_words = set(doc[1])
        for term in unique_words:
            doc_freq[term] += 1
            
    # Calculate IDF 
    N = len(train_data)
    idf = {}
    for term in vocab_list:
        df = doc_freq[term]
        idf[term] = np.log(N / (df + 1)) + 1
        
    # Create TF-IDF vectors for train data
    train_feature_vectors = []
    for doc in train_data:
        words = doc[1]
        term_counts = Counter(words)
        doc_len = len(words)
        feature_vector = []
        for term in vocab_list:
            tf = term_counts.get(term, 0) / doc_len if doc_len > 0 else 0
            tf_idf = tf * idf[term]
            feature_vector.append(tf_idf)
        train_feature_vectors.append(feature_vector)

    # Create TF-IDF vectors for test data
    test_feature_vectors = []
    for doc in test_data:
        words = doc[1]
        term_counts = Counter(words)
        doc_len = len(words)
        feature_vector = []
        for term in vocab_list:
            tf = term_counts.get(term, 0) / doc_len if doc_len > 0 else 0
            tf_idf = tf * idf[term]
            feature_vector.append(tf_idf)
        test_feature_vectors.append(feature_vector)
        
    return train_feature_vectors, test_feature_vectors

def find_stop_words(data: str, bottom: float, min_count: int = 0):
    '''
    Finds stop words based on how many times they appear in data
    @params:
        data (str): The path to the data file.
        bottom (float): The bottom per mille of stop words that will be removed. (we are removing uncommon words)
        min_count (int): Words with count less than this will be removed. Default is 2.
    @returns:
        set: A set of stop words.
    '''
    # Generate the stop words here - the file wrapper breaks if done in the tokenizer
    stop_words = None
    with open(data, 'r', encoding='utf-8') as file:
        actual_words = {}
        for line in file:
            words = re.split('[^a-zA-Z]', line)

            for word in words:
                if word:
                    if word in actual_words:
                        actual_words[word] += 1
                    else:
                        actual_words[word] = 1

        sorted_words = sorted(actual_words.items(), key=lambda item: item[1], reverse=True)
        num_words = len(sorted_words)
        bottom_mille = int(num_words * (bottom / 1000))
        stop_words = [word for word, count in actual_words.items()  if count > bottom_mille and count > min_count]
        
        return set(stop_words) 

def get_data(data_file: str, stop_words: set) -> tuple:
    '''
    Reads a data file, tokenizes it and splits into training and testing datasets.

    @params:
        data_file (str): The path to the data file.
        stop_words (set): A set of stop words to be removed during tokenization.

    @returns:
        tuple: a tuple of two lists:
            - train_data (list): The training dataset.
            - test_data (list): The testing dataset.
    '''
    with open(data_file, 'r', encoding='utf-8') as file:
        corpus = tokenize(file, stop_words)

        data = pd.DataFrame(corpus)
        train_data = data.sample(frac=0.8, random_state=69)
        test_data = data.drop(train_data.index)

        train_data = train_data.to_numpy().tolist()
        test_data = test_data.to_numpy().tolist()

        return train_data, test_data
    
def tokenize(corpus: object, stop_words: set) -> list:
    '''
    Tokenizes a corpus, removing stop words and punctuation.
    @params:
        corpus (object): A collection of documents, where each document is a string.
        stop_words (set): A set of stop words to be removed from the documents. If not provided, a default set of common English stop words is used.
    @returns:
        list: A list of tuples, where each tuple contains the classification label and a list of tokens for each document.
    '''
    if not stop_words:
        stop_words = set(["a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "on", "in", "with", "as", "by", "at", "to", "from", "up", "down", "out", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

    tokenized_corpus = []
    for document in corpus:
        tokens = document.split('\t', 1) 
        classification = tokens[0]
        tokens = tokens[1].split(' ') # Remove classification column

        # Cleaning tokens
        tokens = [token.lower() for token in tokens] # Lowercase all tokens
        tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens] # Remove punctuation
        # tokens = [token for token in tokens if token.isalpha()] # Remove non-alphabetic tokens
        tokens = [token for token in tokens if token not in stop_words] # Remove stop words
        
        tokenized_corpus.append((classification, tokens))

    return tokenized_corpus

def calculate_statistics(tp:int=0, tn:int=0, fp:int=0, fn:int=0) -> tuple:
    '''
    Calculate performance statistics for a classification model.
    @params:
        tp (int): True Positives. Default is 0.
        tn (int): True Negatives. Default is 0.
        fp (int): False Positives. Default is 0.
        fn (int): False Negatives. Default is 0.
    @returns:
    tuple: A tuple containing statistics as a turple of strings formatted to three decimal places:
        - accuracy (str): The accuracy of the model as a percentage.
        - precision (str): The precision of the model as a percentage.
        - recall (str): The recall of the model as a percentage.
        - f1 (str): The F1 score of the model as a percentage.
    @exceptions:
        If a ZeroDivisionError occurs during calculation, all statistics will be set to 0.
    '''
    try:
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall) 
        
        accuracy = f"{accuracy * 100:.3f}%"
        precision = f"{precision * 100:.3f}%"
        recall = f"{recall * 100:.3f}%"
        f1 = f"{f1 * 100:.3f}%"
    except ZeroDivisionError:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0

    return accuracy, precision, recall, f1

def knn(train_vectors: np.array, train_labels: np.array, test_vector: np.array, k: int = 5) -> string:
    '''
    K-Nearest Neighbors classifier that determines if a given message is spam or ham..
    @params:
        train_vectors (np.array): Array of training vectors.
        train_labels (np.array): Array of labels corresponding to the training vectors.
        test_vector (np.array): The vector to classify.
        k (int): The number of nearest neighbors to consider. Defaults to 5.
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

    vocabluary = set(w_spam + w_ham)

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
        p_word_spam = ((w_spam_count[word]) + s) / (len(w_spam) + s * len(vocabluary))
        p_word_ham = ((w_ham_count[word]) + s) / (len(w_ham) + s * len(vocabluary))

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
    
def test_knn(train_data: list, test_data: list, k=5) -> tuple:
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
    train_vectors, test_vectors = vectorize(train_data, test_data)
    train_labels = np.array([doc[0] for doc in train_data])
    test_labels = np.array([doc[0] for doc in test_data])
    
    # Converts to Numpy arrays to make faster
    train_vectors = np.array(train_vectors)
    test_vectors = np.array(test_vectors)
    
    # Classify each instance
    for test_v, label in zip(test_vectors, test_labels):
        prediction = knn(train_vectors, train_labels, test_v, k)
        
        if prediction == 'spam' and label == 'spam':
            tp += 1
        elif prediction == 'spam' and label == 'ham':
            fp += 1
        elif prediction == 'ham' and label == 'ham':
            tn += 1
        elif prediction == 'ham' and label == 'spam':
            fn += 1

    return tp, tn, fp, fn

def test_nb(train_data: list, test_data: list, s: float) -> tuple:
    '''
    Tests a Naive Bayes classifier.
    @params:
        train_data (list): A list of training data where each element is a tuple containing the classification ('spam' or 'ham') and the message.
        test_data (list): A list of test data where each element is a tuple containing the classification ('spam' or 'ham') and the message.
        s (float): laplace smoothing alpha.
    @returns::
        tuple: A tuple containing four integers:
            - tp (int): True positives (correctly classified as spam).
            - tn (int): True negatives (correctly classified as ham).
            - fp (int): False positives (incorrectly classified as spam).
            - fn (int): False negatives (incorrectly classified as ham).
    '''
    tp = tn = fp = fn = 0

    for document in test_data:
        classification = document[0]
        message = document[1]

        result = nb(train_data, message, s)

        if (result == classification) and (classification == 'spam'):
            tp += 1
                
        elif (result != classification) and (classification == 'spam'):
            fp += 1

        if (result == classification) and (classification == 'ham'):
            tn += 1

        elif (result != classification) and (classification == 'ham'):
            fn += 1

    return tp, tn, fp, fn

def run_test(data_file: str, method: callable, name: str, stop_word_bottom_mille: float, min_count: int, parameters: dict) -> None:
    '''
    Runs a test for a classifier on given data using a specified classification method and parameters.
    @params
        data_file (str): Path to the data file.
        method (callable): The classifier test method.
        name (str): Name of the classifier.
        stop_word_bottom_mille (int): The top per mille of stop words that will be removed.
        parameters (dict): A str:int dictionary of additional parameters.
    '''
    if callable(method):
        stop_words = find_stop_words(data_file, stop_word_bottom_mille, min_count)
        print(f'Stop words removed - bottom {stop_word_bottom_mille}‰. Removed words with less than {min_count} occurrences.')
        if parameters:
            print(parameters)

        train_data, test_data = get_data(data_file, stop_words)

        tp, tn, fp, fn = method(train_data, test_data, **parameters)
        accuracy, precision, recall, f1 = calculate_statistics(tp, tn, fp, fn)

        result = f"{name}: \nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\n\n"

        print(f'{result}')
        with open("classifier/results.txt", 'a') as results:
            results.write(result)
        
    else:
        print(f"{method} not found.")

def run_tests(data: str, classifiers: dict) -> None:
    '''
    Runs tests for classifiers.
    @params:
        data (str): Path to the data file
        classifiers (dict): A dictionary of classifiers to be tested.
        The dictionary must be in the format {name: (method, int, optional_dict)
        Here, name is the name of the classifier, method is the test method for that classifier, int is the top per mille words that will be removed, and optional_dict is a dictionary of parameters.
        optional_dict is a dictionary of str:int, where str is the parameter name and int is the parameter value. 
    '''
    for name, classifier in classifiers.items():
        # Unpack the classifiers with optional parameters, such as for knn
        method, stop_word_bottom_mille, min_count, *parameters = classifier
        parameters = parameters[0] if parameters else {}

        run_test(data, method, name, stop_word_bottom_mille, min_count, parameters)

def _evaluate_configuration(method: callable, train_data: list, test_data: list, stop_words: set, min_count: int, params: dict) -> str:
    '''
    Actually runs the classification test method.
    @params:
        method (callable): Classifier test method to be ran. Must be return a tuple of 4 ints - (tp, tn, fp, fn).
        train_data (list): Training data
        test_data (list): Test data
        stop_words (set): a set of stop words
        params (dict): str:int dictionary of parameters 
    @returns:
        str: A string in the format method(stop_word_bottom_mille, {params}): f1%.
        Here, method is the method name, both parenthesis and the percentage sign are literal characters. stop_word_bottom_mille is an integer, params is a dictionary, and f1 is a float.
        Params is a dictionary of str:int, where str is the parameter name and int is the parameter value. 
    '''
    # Execute the method with the given parameters and return the f1 score
    tp, tn, fp, fn = method(train_data, test_data, **params)
    accuracy, precision, recall, f1 = calculate_statistics(tp, tn, fp, fn)
    result = f"{method.__name__}({stop_words}, {min_count}, {params}): {f1}\n"
    print(result,end='')
    return result

def find_value(data_file: str, method: callable, stop_word_bottom_mille: range, min_count: range, parameters: dict, max_processes: int = 6) -> None:
    '''
    Uses the provided classifer test method to evaluate the F1 score using given parameters.
    @params:
        data_file (str): Path to the data file
        method (callable): Classifier test method
        stop_word_bottom_mille (range): A range of stop_word_bottom_mille values. If not the one being tested, make it a range that doesn't iterate, e.g range(0, 1)
        parameters (dict): A str:range dictionary of parameters.
        max_processes (int, optional): The maximum number of parallel processes to use. Defaults to 6.
    @exceptions:
        All exceptions are caught and printed to the console.
    '''
    import concurrent.futures
    import copy

    all_tasks = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        for stop_word in stop_word_bottom_mille:
            stop_word = float(stop_word)
            for mc in min_count:
                mc = float(mc)
                stop_words = find_stop_words(data_file, stop_word, mc)
                train_data, test_data = get_data(data_file, stop_words)

                def recursive_call(train_data, test_data, stop_words, params):
                    if not params.keys():
                        future = executor.submit(_evaluate_configuration, method, train_data, test_data, stop_word, mc, params) # Run the process
                        all_tasks.append(future)
                        return
                    
                    current_key = list(params.keys())[0]

                    if not isinstance(params[current_key], range) and not isinstance(params[current_key], np.ndarray):
                        # Base case
                        params_copy = copy.deepcopy(params)  # Isolate parameters for each process
                        future = executor.submit(_evaluate_configuration, method, train_data, test_data, stop_word, mc, params_copy)
                        all_tasks.append(future)
                        return

                    for value in params[current_key]:
                        value = float(value)
                        params_copy = copy.deepcopy(params) 
                        # Change the parameter to be a single value for method
                        params_copy[current_key] = value
                        # Recurse with remaining keys.
                        recursive_call(train_data, test_data, stop_words, params_copy)

                recursive_call(train_data, test_data, stop_words, parameters)

        for task in concurrent.futures.as_completed(all_tasks):
            result = task.result()
            with open("classifier/values.txt", 'a') as results:
                results.write(result)
    
    with open("classifier/values.txt", 'a') as results:
        results.write('\n')

def main() -> None:
    data = "corpus/SMSSpamCollection"

    with open("classifier/results.txt", 'w') as results:
        results.write("")
        
    # Experiment with different values using find_value()
    # See values.txt for results - these parameters are most optimal
    classifiers = {"Naive Bayes": (test_nb, 8, 0, {'s': 0.7}), "K-Nearest Neighbor": (test_knn, 500, 0, {'k':7})}
    
    #find_value(data, test_knn, range(500,501), range(0, 21), {'k':range(7,8)})
    #find_value(data, test_nb, range(8, 9), range(0, 1), {'s': np.arange(0.1, 1, 0.1)})

    run_tests(data, classifiers) 

if __name__ == "__main__":
    main()