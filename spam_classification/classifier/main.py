import classifier_tests as tests
import utils

def main() -> None:
    data = "spam_classification/corpus/SMSSpamCollection"

    with open("spam_classification/docs/results.txt", 'w') as results:
        results.write("")
        
    # Experiment with different values using find_value()
    # See values.txt for results - these parameters are most optimal
    classifiers = {"Naive Bayes": (tests.test_nb, 8, 0, {'s': 4}), "K-Nearest Neighbor": (tests.test_knn, 500, 0, {'k':7})}

    ''' Example of value search
    import numpy as np
    utils.find_value(data, tests.test_knn, range(500,501), range(0, 21), {'k':range(7,8)})
    utils.find_value(data, tests.test_nb, range(8, 9), range(0, 1), {'s': np.arange(4, 5, 0.1)})
    '''
    
    tests.run_tests(data, classifiers) 

if __name__ == "__main__":
    main()