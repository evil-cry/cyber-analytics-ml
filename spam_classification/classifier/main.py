import classifiers
import utils

def main() -> None:
    data = "spam_classification/corpus/SMSSpamCollection"

    with open("spam_classification/docs/results.txt", 'w') as results:
        results.write("")
        
    # Experiment with different values using find_value()
    # See values.txt for results - these parameters are most optimal
    c = [classifiers.NB(data, "Naive Bayes", 8, 0, {'s': 4}), classifiers.KNN(data, "K-Nearest Neighbor", 500, 0, {'k':7})]
    
    for classifier in c:
        accuracy, precision, recall, f1 = classifier.evaluate()
        result = f"{classifier.name}: \nAccuracy: {accuracy * 100:.3f}%\nPrecision: {precision * 100:.3f}%\nRecall: {recall * 100:.3f}%\nF1: {f1 * 100:.3f}%\n\n"

        print(result)
        with open("spam_classification/docs/results.txt", 'a') as results:
            results.write(result)

if __name__ == "__main__":
    main()