import numpy as np
import utils
import classifiers

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
    train_vectors, test_vectors = utils.vectorize(train_data, test_data)
    train_labels = np.array([doc[0] for doc in train_data])
    test_labels = np.array([doc[0] for doc in test_data])
    
    # Converts to Numpy arrays to make faster
    train_vectors = np.array(train_vectors)
    test_vectors = np.array(test_vectors)
    
    # Classify each instance
    for test_v, label in zip(test_vectors, test_labels):
        prediction = classifiers.knn(train_vectors, train_labels, test_v, k)
        
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
        stop_words = utils.find_stop_words(data_file, stop_word_bottom_mille, min_count)
        print(f'Stop words removed - bottom {stop_word_bottom_mille}‰. Removed words with less than {min_count} occurrences.')
        if parameters:
            print(parameters)

        train_data, test_data = utils.get_data(data_file, stop_words)

        tp, tn, fp, fn = method(train_data, test_data, **parameters)
        accuracy, precision, recall, f1 = utils.calculate_statistics(tp, tn, fp, fn)

        result = f"{name}: \nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\n\n"

        print(f'{result}')
        with open("spam_classification/docs/results.txt", 'a') as results:
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