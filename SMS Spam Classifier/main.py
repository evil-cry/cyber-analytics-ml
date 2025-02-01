from collections import Counter
import numpy as np
import pandas as pd
import string

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
    
# Calculates the similarity between two vectors
def cosine_sim(vec1, vec2):
    # Calculate dot product 
    dproduct = np.dot(vec1,vec2)
    
    # Caculate Magnitudes
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    final = dproduct/(norm1*norm2)
    return final

def k_nn(train_vectors, train_labels, test_vector, k=5):
    # Considers 5 neighbors
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
    
    
def test_knn(train_data: list, test_data: list, k=5) -> None:
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    count = 0
    
    # Get Vectors from Vecotrize and create the labels
    train_vectors, test_vectors = vectorize(train_data, test_data)
    train_labels = [doc[0] for doc in train_data]
    test_labels = [doc[0] for doc in test_data]
    
    # Classify each instance
    for test_v, label in zip(test_vectors, test_labels):
        prediction = k_nn(train_vectors, train_labels, test_v, k)
        
        if prediction == 'spam' and label == 'spam':
            tp += 1
        elif prediction == 'spam' and label == 'ham':
            fp += 1
        elif prediction == 'ham' and label == 'ham':
            tn += 1
        elif prediction == 'ham' and label == 'spam':
            fn += 1
    
    # Final Calculations
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall) 
    
    # Output Final Calculations
    print('\n\nK_nn Model\n')
    print(f'accuracy: {accuracy}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1: {f1}')
    
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