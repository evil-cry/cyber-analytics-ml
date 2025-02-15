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