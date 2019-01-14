import numpy as np
import os
from collections import Counter
import math
import scipy

def process_file(filepath):
    """Given a file, returns a list of tokens for that file"""
    x = []
    with open(filepath, 'r') as f:
        for l in f:
            # Filter lines which consist only of new line operator
            if l == '\n':
                continue
            
            token, pos_tagging = l.split('\t')
            x.append(token)
    return x

def preprocess_data(datapath, sentiment='POS'):
    idx = 0
    X = []
    y = []
    sentiment_value = 1 if sentiment == 'POS' else 0
    
    # For file in the folder
    current_path = datapath + sentiment
    for f in sorted(os.listdir(current_path)):
        x = process_file(current_path + '/' + f)
        X.append(x)
        y.append(sentiment_value)

    return X, y

def get_unigram_dictionary(X, cutoff=1):
    token_counter = Counter(np.concatenate(X))
    idx = 0
    token_to_idx = {}
    
    for x in X:
        for token in x:
            if token_counter[token] >= cutoff and token not in token_to_idx:
                token_to_idx[token] = idx
                idx += 1
                
    return token_to_idx

def get_bigram_dictionary(X, cutoff=1, token_to_idx={}):
    X_bigram = []
    for x in X:
        X_bigram += [(x[i], x[i + 1]) for i, _ in enumerate(x) if i < len(x) - 1 ]

    token_counter = Counter(X_bigram)
    idx = len(token_to_idx)
    
    for x in X:
        x_bigram = [(x[i], x[i + 1]) for i, _ in enumerate(x) if i < len(x) - 1 ]
        for token in x_bigram:
            if token_counter[token] >= cutoff and token not in token_to_idx:
                token_to_idx[token] = idx
                idx += 1
                
    return token_to_idx

def get_dictionary(X, unigram_cutoff=1, bigram_cutoff=1, unigram=True, bigram=False):
    """
    Returns a dictionary which maps each token to its index in the feature space.
    Tokens which appear less times than specified by the cutoff are discarded
    """
    token_to_idx = {}
    if unigram:
        token_to_idx = get_unigram_dictionary(X, unigram_cutoff)
    if bigram:
        token_to_idx = get_bigram_dictionary(X, bigram_cutoff, token_to_idx)
    
    print("Generated {} features".format(len(token_to_idx)))
                    
    return token_to_idx

def featurize_data(X, token_to_idx):
    """Convert each sample from a list of tokens to a multinomial bag of words representation"""
    X_unigram_and_bigram = []
    for x in X:
        X_unigram_and_bigram.append(x + [(x[i], x[i + 1]) for i, _ in enumerate(x) if i < len(x) - 1 ])
        
    X_feat = []
    for x in X_unigram_and_bigram:
        x_feat = np.zeros((len(token_to_idx)))
        for token in x:
            if token in token_to_idx:
                x_feat[token_to_idx[token]] += 1
        X_feat.append(x_feat)
    
    return X_feat

