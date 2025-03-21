import numpy as np
import pandas as pd
from collections import Counter
import string
import re

def vectorize(train_data: list, test_data: list) -> tuple:
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

def find_stop_words(data: str, bottom: float, min_count: int = 0) -> set:
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

# NO LONGER WORKS AFTER REFACTORING
"""
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
            with open("spam_classification/docs/values.txt", 'a') as results:
                results.write(result)
    
    with open("spam_classification/docs/values.txt", 'a') as results:
        results.write('\n')
"""