from collections import Counter
import numpy as np
import pandas as pd
import string

def vectorize(train_data: list, test_data: list) -> None:
    # Build vocabulary from training data
    vocabulary = set()
    for document in train_data:
        vocabulary.update(document[1])

    terms = Counter([term for document in train_data for term in document[1]])
    terms += Counter([term for document in test_data for term in document[1]])
    document_length = len(train_data)

    # Calculate inverse document frequency for each term
    idf = {}
    for word, count in terms.items():
        idf[word] = np.log(document_length / (count + 1)) + 1

    print(idf)

    # Calculate term frequency for each term in the training data set
    train_feature_vectors = []

    for document in train_data:
        tf = {}

        words = document[1]
        document_frequency = Counter(words)
        document_word_total = len(words)

        for word in set(words):
            tf[word] = document_frequency[word] / document_word_total if document_word_total > 0 else 0

        feature_vector = []

        for word in set(words):
            tf_idf = tf[word] * idf[word]
            feature_vector.append(tf_idf)

        train_feature_vectors.append(feature_vector)

    # Calculate term frequency for each term in the testing data set
    test_feature_vectors = []
    for document in test_data:
        tf = {}

        words = document[1]
        document_frequency = Counter(words)
        document_word_total = len(words)

        for word in set(words):
            tf[word] = document_frequency[word] / document_word_total if document_word_total > 0 else 0

        feature_vector = []

        for word in set(words):
            tf_idf = tf[word] * idf[word]
            feature_vector.append(tf_idf)

        print(feature_vector)

        test_feature_vectors.append(feature_vector)

    print(test_feature_vectors)
    print(train_feature_vectors)
    
# Calculates the similarity between two vectors
def cosine_distance(vec1, vec2):
    # Calculate dot product 
    dproduct = np.dot(vec1,vec2)
    
    # Caculate Magnitudes
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    final = dproduct/(norm1*norm2)
    return 

def k_nn(corpus: list) -> None:
    
    pass

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

    # Get vocabulary, vocabulary size, and alpha
    w = set(w_spam + w_ham)
    s = len(w)
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

def test_nb(train_data: list, test_data: list) -> None:
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    count = 0

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

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    accuracy = f"{accuracy * 100:.6f}%"
    precision = f"{precision * 100:.6f}%"
    recall = f"{recall * 100:.6f}%"
    f1 = f"{f1 * 100:.6f}%"

    print(f'accuracy: {accuracy}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')

def tokenize(corpus) -> list:
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

def main() -> None:
    with open("Corpus/SMSSpamCollection", 'r') as file:
        corpus = tokenize(file)

        data = pd.DataFrame(corpus)
        train_data = data.sample(frac=0.8, random_state=69)
        test_data = data.drop(train_data.index)

        train_data = train_data.to_numpy().tolist()
        test_data = test_data.to_numpy().tolist()  

        test_nb(train_data, test_data)
        # vectorize(train_data, test_data)

if __name__ == "__main__":
    main()