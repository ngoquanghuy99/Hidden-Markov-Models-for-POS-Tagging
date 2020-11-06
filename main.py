import io
import json
import pickle
import numpy as np
from numpy import save
from numpy import load
import argparse
from nltk import word_tokenize
from hmm import build_vocab2idx, create_dictionaries, create_transition_matrix, create_emission_matrix, initialize, viterbi_forward, viterbi_backward
from utils import processing
from hmm import training_data

corpus_path = "WSJ_02-21.pos"
alpha = 0.001

def load_data():
    vocab2idx = build_vocab2idx(corpus_path)
    f = open('vocab.pkl', 'wb')
    pickle.dump(vocab2idx, f)
    f.close()
    
    training_corpus = training_data(corpus_path)
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab2idx)
    states = sorted(tag_counts.keys())
    alpha = 0.001
    A = create_transition_matrix(transition_counts, tag_counts, alpha)
    B = create_emission_matrix(emission_counts, tag_counts, list(vocab2idx), alpha)
    save('A.npy', A)
    save('B.npy', B)

if __name__ == "__main__":
    load_data()
