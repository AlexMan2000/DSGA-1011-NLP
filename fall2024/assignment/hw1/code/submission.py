import json
import collections
import argparse
import random
import numpy as np
from collections import defaultdict, Counter

from numpy.linalg import LinAlgError

from util import *
import ssl  # For compatibility and smooth download from nltk, necessary for local dev

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

random.seed(42)


def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW featurs of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    # BEGIN_YOUR_CODE
    boW = defaultdict(int)
    for word in ex["sentence1"] + ex["sentence2"]:
        boW[word] += 1

    # for word in ex["sentence1"]:
    #     boW[word] += 1
    return boW
    # END_YOUR_CODE


def extract_ngram_features(n=1, filtered=False):
    """Return n-grams in the hypothesis and the premise.

    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
        n : int, optional (default=1)
            The 'n' in n-grams. If n=1, it extracts unigrams (Bag-of-Words). If n=2, it extracts bigrams, and so on.

    Returns:
        A dictionary of n-gram features (BoW features of x).

    Example:
        Unigram (n=1):
        "I love it", "I hate it" --> {"I": 2, "it": 2, "hate": 1, "love": 1}
        Bigram (n=2):
        "I love it", "I hate it" --> {"I love": 1, "love it": 1, "I hate": 1, "hate it": 1}
    """

    from nltk.corpus import stopwords
    nltk.download('stopwords')
    # Define stopwords
    stop_words = set(stopwords.words('english'))

    def inner_func(ex):
        # BEGIN_YOUR_CODE
        boW = defaultdict(int)

        # Combine both sentences
        hypothesis = ex["sentence1"]
        premise = ex["sentence2"]
        combined_sentence = ex["sentence1"] + ex["sentence2"]
        # combined_sentence = list(map(lambda word: word.lower(), combined_sentence))
        # for i in range(len(hypothesis) - n + 1):
        #     ngram = " ".join(hypothesis[i:i + n])  # Create n-grams as space-separated strings
        #     boW[ngram] += 1
        #
        # for i in range(len(premise) - n + 1):
        #     ngram = " ".join(premise[i:i + n])  # Create n-grams as space-separated strings
        #     boW[ngram] += 1

        if filtered:
            combined_sentence = list(filter(lambda word: word.lower() not in stop_words, combined_sentence))

        # # Loop over the sentence and extract n-grams
        for i in range(len(combined_sentence) - n + 1):
            ngram = " ".join(combined_sentence[i:i + n])  # Create n-grams as space-separated strings
            boW[ngram] += 1

        return boW

    return inner_func
    # END_YOUR_CODE


def stable_sigmoid(x):
    """
    Helper function to compute stable sigmoid 1 / (1 + np.exp(- x))
    Args:
        x: a float number
    Returns: The float value of sigmoid(x)
    """
    # Use different formulations of the sigmoid function based on the value of x to prevent overflow/underflow
    epsilon = 1e-10  # A small value to prevent log(0)
    if x >= 0:
        z = np.exp(-x)
        return np.clip(1 / (1 + z), epsilon, 1 - epsilon)
    else:
        z = np.exp(x)
        return np.clip(z / (1 + z), epsilon, 1 - epsilon)


def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    # BEGIN_YOUR_CODE
    weights = defaultdict(float)

    # Should change mode
    train_model(train_data, valid_data, weights, feature_extractor, learning_rate, num_epochs, mode="tf-idf")

    return weights
    # END_YOUR_CODE


def train_model(train_data, valid_data, weights, feature_extractor, learning_rate, num_epochs, mode="bow"):
    for epoch in range(1, num_epochs + 1):
        training_loss = 0
        for data_point in train_data:
            features = feature_extractor(data_point)
            # Expand the weight vector
            for k, v in features.items():
                if k not in weights:
                    weights[k] = 0

            # SGD
            label = data_point["gold_label"]
            predict_prob = stable_sigmoid(dot(weights, features))
            training_loss += - label * np.log(predict_prob) - (1 - label) * np.log(1 - predict_prob)

            # Gradient (sigmoid(w * f(x)) - y) * f(x), update
            increment(weights, features, (- learning_rate) * (predict_prob - label))

        print(f"Epoch: {epoch}, training_loss: {training_loss / len(train_data)}")


def valid_model(valid_data, weights, feature_extractor, epoch):
    validation_loss = 0
    validation_data = []
    for data_point in valid_data:
        features = feature_extractor(data_point)
        label = data_point["gold_label"]
        validation_data.append((features, label))
        # SGD
        prediction = stable_sigmoid(dot(weights, features))
        validation_loss += - label * np.log(prediction) - (1 - label) * np.log(1 - prediction)
    validation_err = evaluate_predictor(validation_data, lambda ex: 1 if dot(weights, features) > 0 else 0)

    print(f"Epoch: {epoch}, validation_loss: {validation_loss}")
    print(f"Epoch: {epoch}, validation_err: {validation_err}")


def extract_custom_features_tf_idf(train_data, filter=False, ngram=1):
    """Design your own features with TF-IDF and stopwords removal using closure.

    Parameters:
        train_data : list of dict
            Each dict contains 'gold_label' (optional), 'sentence1', and 'sentence2'.

    Returns:
        A function `compute_tfidf(ex)` that computes the TF-IDF for a given sentence.
        The format should be {"word1": tf_idf_1, "word2": tf_idf_2, ...}
        >>> train_data = [
        ...     {"sentence1": ["I", "love", "cats"], "sentence2": ["I", "hate", "dogs"]},
        ...     {"sentence1": ["Dogs", "are", "great"], "sentence2": ["Cats", "are", "cute"]}
        ... ]
        >>> compute_tfidf = extract_custom_features_tf_idf(train_data)
        >>> test_point = {"sentence1": ["I", "love", "cats"], "sentence2": ["I", "hate", "dogs"]}
        >>> result = compute_tfidf(test_point)
        >>> "i" not in result # Since I is considered as a stopword in mltk
        True
    """
    # Download stopwords if not already available
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    # Define stopwords
    stop_words = set(stopwords.words('english'))

    # Preprocess the sentences: Tokenize and remove stopwords
    def preprocess(sentences_, stop_words_):
        preprocessed_sentences = []
        for sentence in sentences_:
            # Remove stopwords and lowercase words
            filtered_words = [word.lower() for word in sentence]
            if filter:
                filtered_words = [word.lower() for word in sentence if word.lower() not in stop_words_]
            ngrams = [" ".join(filtered_words[i: i + ngram]) for i in range(len(filtered_words) - ngram + 1)]
            preprocessed_sentences.append(ngrams)
        return preprocessed_sentences

    # Calculate Inverse Document Frequency (IDF)
    def compute_idf(processed_sentences_):
        N = len(processed_sentences_)  # Total number of documents
        idf_dic_ = defaultdict(int)
        for sentence in processed_sentences_:
            for word in set(sentence):
                idf_dic_[word] += 1

        idf_dic_ = {word: math.log(N / count) for word, count in idf_dic_.items()}
        return idf_dic_

    # Process all sentences (sentence1 + sentence2 for each data point)
    sentences = [obj["sentence1"] + obj["sentence2"] for obj in train_data]
    processed_sentences = preprocess(sentences, stop_words)
    idf_dic = compute_idf(processed_sentences)

    # Define a closure that computes TF-IDF for a given sentence pair, optimized
    def compute_tfidf(ex):
        combined_sentence = ex["sentence1"] + ex["sentence2"]
        # Compute Term Frequency (TF) for one sentence which is formatted as {"gold_label": {0,1}, "sentence1":[str]
        # , "sentence2":[str]}
        tf_dict = Counter([word.lower() for word in combined_sentence])
        if filter:
            tf_dict = Counter([word.lower() for word in combined_sentence if word.lower() not in stop_words])
        total_words = len(tf_dict)
        if total_words > 0:
            tf_dict = {word: count / total_words for word, count in tf_dict.items()}
        # Compute TF-IDF
        tfidf_dict = {word: tf_val * idf_dic.get(word, 0) for word, tf_val in tf_dict.items()}

        return tfidf_dict

    # Return the TF-IDF computation function, with closure
    return compute_tfidf
    # END_YOUR_CODE

def extract_custom_features(ex):
    """

    Args:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)

    Returns:
        The tf-idf dict of this sentence
    """
    pass


def count_cooccur_matrix(tokens, window_size=4):
    """Compute the co-occurrence matrix given a sequence of tokens.
    For each word, n words before and n words after it are its co-occurring neighbors.
    For example, given the tokens "in for a penny , in for a pound",
    the neighbors of "penny" given a window size of 2 are "for", "a", ",", "in".
    Parameters:
        tokens : [str]
        window_size : int
    Returns:
        word2ind : dict
            word (str) : index (int)
        co_mat : np.array
            co_mat[i][j] should contain the co-occurrence counts of the words indexed by i and j according to the
            dictionary word2ind.
    Examples:
        >>> test_data = ["i", "wish", "i", "had", "it", "would", "pass", "the", "exam"]
        >>> word2ind, co_mat = count_cooccur_matrix(test_data, 2)
        >>> co_mat[1, 2] == co_mat[2, 1]
        True
        >>> co_mat[1, 3] == co_mat[3, 1]
        True
        >>> co_mat[2, 3] == co_mat[3, 2]
        True
    """
    # BEGIN_YOUR_CODE
    # Get unique words and their indices
    vocab = sorted(set(tokens))
    word2ind = {word: idx for idx, word in enumerate(vocab)}

    # Initialize co-occurrence matrix with dimensions based on vocabulary size
    vocab_size = len(vocab)
    co_mat = np.zeros((vocab_size, vocab_size), dtype=np.int64)

    # Compute co-occurrence counts
    for i, word in enumerate(tokens):
        cur_word_index = word2ind[word]

        # Define the window range
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)

        # Iterate through the window
        for j in range(start, end):
            if i != j:  # Avoid self-co-occurrence
                outer_word_index = word2ind[tokens[j]]
                co_mat[cur_word_index, outer_word_index] += 1
                co_mat[outer_word_index, cur_word_index] += 1

    return word2ind, co_mat
    # END_YOUR_CODE


def cooccur_to_embedding(co_mat, embed_size=50):
    """Convert the co-occurrence matrix to word embedding using truncated SVD. Use the np.linalg.svd function.
    Parameters:
        co_mat : np.array
            vocab size x vocab size
        embed_size : int
    Returns:
        embeddings : np.array
            vocab_size x embed_size
    """
    # BEGIN_YOUR_CODE
    try:
        U, Sigma, Vh = np.linalg.svd(co_mat, full_matrices=False)
        U_k = U[:, :embed_size]
        Sigma_k = np.diag(Sigma[:embed_size])
        return np.dot(U_k, Sigma_k)
    except LinAlgError:
        print("SVD computation does not converge.")
    # END_YOUR_CODE


def top_k_similar(word_ind, embeddings, word2ind, k=10, metric='dot'):
    """Return the top k most similar words to the given word (excluding itself).
    You will implement two similarity functions.
    If metric='dot', use the dot product.
    If metric='cosine', use the cosine similarity.
    Parameters:
        word_ind : int
            index of the word (for which we will find the similar words)
        embeddings : np.array
            vocab_size x embed_size
        word2ind : dict
        k : int
            number of words to return (excluding self)
        metric : 'dot' or 'cosine'
    Returns:
        topk-words : [str]
    """
    # BEGIN_YOUR_CODE
    # Create a reverse mapping from index to word
    ind2word = {index: word for word, index in word2ind.items()}

    if metric == "dot":
        # Dot product similarity
        dot_product_score = embeddings @ embeddings[word_ind].reshape(-1, 1)
        similarity_scores = dot_product_score.squeeze()

    elif metric == "cosine":
        # Normalize the embeddings (handle zero norm by adding a small epsilon)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # Add epsilon to prevent division by zero
        normalized_embeddings = embeddings / norms

        # Calculate cosine similarity
        cosine_score = normalized_embeddings @ (normalized_embeddings[word_ind].reshape(-1, 1))
        similarity_scores = cosine_score.squeeze()

    else:
        raise RuntimeError(f"Unsupported metric {metric}! Please choose between 'dot' and 'cosine'.")
    top_k_indices = np.argsort(similarity_scores)[::-1][1:k + 1]
    top_k_words = [ind2word[idx] for idx in top_k_indices]

    return top_k_words
    # END_YOUR_CODE


if __name__ == "__main__":
    import doctest

    # doctest.run_docstring_examples(extract_custom_features_tf_idf, globals(), verbose=True)
    doctest.run_docstring_examples(count_cooccur_matrix, globals(), verbose=True)
