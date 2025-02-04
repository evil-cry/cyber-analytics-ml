from collections import Counter
import numpy as np
import pandas as pd
import string
import re

def vectorize(train_data: list, test_data: list):
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

def find_stop_words(data: str, top: float):
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
        top_20_percent = int(num_words * (top / 1000))
        stop_words = [word for word, count in actual_words.items()  if count > top_20_percent]
        
        return set(stop_words) 

def get_data(data_file: str, stop_words: set) -> tuple:
    with open(data_file, 'r', encoding='utf-8') as file:
        corpus = tokenize(file, stop_words)

        data = pd.DataFrame(corpus)
        train_data = data.sample(frac=0.8, random_state=69)
        test_data = data.drop(train_data.index)

        train_data = train_data.to_numpy().tolist()
        test_data = test_data.to_numpy().tolist()

        return train_data, test_data
    
def tokenize(corpus: object, stop_words: set) -> list:
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
    
def nb(corpus: list, sample: str) -> str:
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

    # Get count of each word in spam and ham messages
    w_spam_count = Counter(w_spam)
    w_ham_count = Counter(w_ham)

    # Get vocabulary and alpha
    w = set(w_spam + w_ham)
    a = 1

    # Calculate the probability of the sample being spam or ham
    p_sample_spam = 0
    p_sample_ham = 0

    for word in sample:        
        # Calculate the P(W_i|Spam) and P(W_i|Ham)
        p_word_spam = (w_spam_count[word]) / (len(w_spam) + a)
        p_word_ham = (w_ham_count[word]) / (len(w_ham) + a)

        # Calculate the P(Spam) and P(Ham)
        p_spam = spam_count / len(corpus) 
        p_ham = ham_count / len(corpus)

        # Calculate the P(Spam|W_i) and P(Ham|W_i)
        p_spam_word = p_spam * p_word_spam
        p_ham_word = p_ham * p_word_ham

        # Add to total P(W|Spam) and P(W|Ham)
        p_sample_spam += p_spam_word
        p_sample_ham += p_ham_word

    if p_sample_spam > p_sample_ham:
        return 'spam'

    elif p_sample_spam < p_sample_ham:
        return 'ham'

    else: 
        return 'unknown'
    
def test_knn(train_data: list, test_data: list, k=5) -> tuple:
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

def test_nb(train_data: list, test_data: list) -> tuple:
    tp = tn = fp = fn = count = 0

    for document in test_data:
        classification = document[0]
        message = document[1]

        result = nb(train_data, message)

        if (result == classification) and (classification == 'spam'):
            tp += 1
                
        elif (result != classification) and (classification == 'spam'):
            fp += 1

        if (result == classification) and (classification == 'ham'):
            tn += 1

        elif (result != classification) and (classification == 'ham'):
            fn += 1

        count += 1
        if count > 200:
                break

    return tp, tn, fp, fn

def run_test(data_file: str, method: callable, name: str, stop_word_top_mille: float, parameters: dict) -> None:
    if callable(method):
        stop_words = find_stop_words(data_file, stop_word_top_mille)
        print(f'Stop words removed - top {stop_word_top_mille}‰')
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
    for name, classifier in classifiers.items():
        # Unpack the classifiers with optional parameters, such as for knn
        method, stop_word_top_mille, *parameters = classifier
        parameters = parameters[0] if parameters else {}

        run_test(data, method, name, stop_word_top_mille, parameters)

def _evaluate_configuration(method, train_data, test_data, stop_words, params):
    # Execute the method with the given parameters and return the f1 score
    tp, tn, fp, fn = method(train_data, test_data, **params)
    accuracy, precision, recall, f1 = calculate_statistics(tp, tn, fp, fn)
    result = f"{method.__name__}({stop_words}, {params}): {f1}%\n"
    print(result,end='')
    return result

def find_value(data_file: object, method: callable, stop_word_top_mille: range, parameters: dict, max_processes: int = 6) -> None:
    import concurrent.futures
    import copy

    if callable(method) and isinstance(stop_word_top_mille, range) and isinstance(parameters, dict):
        all_tasks = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
            for stop_word in stop_word_top_mille:
                stop_words = find_stop_words(data_file, stop_word)
                train_data, test_data = get_data(data_file, stop_words)

                def recursive_call(train_data, test_data, stop_words, params):
                    if not params.keys():
                        future = executor.submit(_evaluate_configuration, method, train_data, test_data, stop_word, params) # Run the process
                        all_tasks.append(future)
                        return
                    
                    current_key = list(params.keys())[0]

                    if not isinstance(params[current_key], range):
                        # Base case
                        params_copy = copy.deepcopy(params)  # Isolate parameters for each process
                        future = executor.submit(_evaluate_configuration, method, train_data, test_data, stop_word, params_copy)
                        all_tasks.append(future)
                        return

                    for value in params[current_key]:
                        params_copy = copy.deepcopy(params) 
                        # Change the parameter to be a single value for method
                        params_copy[current_key] = value
                        # Recurse with remaining keys.
                        recursive_call(train_data, test_data, stop_words, params_copy)

                recursive_call(train_data, test_data, stop_words, parameters)

            for task in concurrent.futures.as_completed(all_tasks):
                try:
                    result = task.result()
                    with open("classifier/values.txt", 'a') as results:
                        results.write(result)
                except Exception as e:
                    print(f"{e}")
    else:
        print("Something went wrong in find_value.")
    
    with open("classifier/values.txt", 'a') as results:
        results.write('\n')

def main() -> None:
    data = "corpus/SMSSpamCollection"
    # Experiment with different values using find_value()
    # See values.txt for results - these parameters are most optimal
    classifiers = {"Naive Bayes": (test_nb, 5), "K-Nearest Neighbor": (test_knn, 500, {'k':7})}
    
    #find_value(data, test_knn, range(500, 601, 10), {'k':range(7,8)})
    #find_value(data, test_nb, range(0, 10), {})

    run_tests(data, classifiers) 

if __name__ == "__main__":
    main()