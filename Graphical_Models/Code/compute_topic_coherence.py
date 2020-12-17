"""
NPMI ranges from −1 to 1, with −1 indicating that the pair of tags never occurred
together, 0 indicating that the tag occurrences were independent of each other,
and 1 indicating that the tags co-occurred perfectly with each other.
"""

import argparse
import numpy as np
from collections import defaultdict
from data import build_dtm

eps = 1e-5

def get_args():
    parser = argparse.ArgumentParser(description='Computes the observed coherence for a given topic and word-count file.')
    parser.add_argument('--datadir', type=str, help="data directory")
    parser.add_argument('--predictions-file', help="file that contains the topics")
    parser.add_argument('--num-documents', type=int, help="Number of documents to include in document-word matrix.", default=0)
    parser.add_argument("--max-words", type=int, help="Max number of words to include from document.", default=1000)
    args = parser.parse_args()
    return args


def compute_word_count(X, w):
    docs, words = X.row, X.col
    count = 0
    for d in range(X.shape[0]):
        d_words = words[np.where(docs == d)[0]]
        if w in d_words:
            count += 1
    return count


def compute_word_pair_count(X, w1, w2):
    docs, words = X.row, X.col
    count = 0
    for d in range(X.shape[0]):
        d_words = words[np.where(docs == d)[0]]
        if set([w1, w2]).issubset(set(d_words)):
            count += 1
    return count


def main():
    args = get_args()
    X, vocab = build_dtm(args.datadir, args.num_documents, args.max_words)
    topics = defaultdict(list)
    with open(args.predictions_file, 'r') as f:
        for line in f.readlines():
            topic_id = int(line.split(': ')[0].split()[1])
            words = line.split(': ')[1].split()
            topics[topic_id] = words
    
    scores = []
    for topic, words in topics.items():
        topic_score = []
        for i in range(len(words) - 1):
            for j in range(i + 1, len(words)):
                w1, w2 = np.argwhere(vocab == words[i]).item(), np.argwhere(vocab == words[j]).item()
                w1_count = compute_word_count(X, w1)
                w2_count = compute_word_count(X, w2)
                w1w2_count = compute_word_pair_count(X, w1, w2)

                pmi = np.log((w1w2_count * args.num_documents) / ((w1_count * w2_count) + eps) + eps)
                npmi = pmi / (-np.log((w1w2_count / (args.num_documents + eps) + eps)))
                topic_score.append(npmi)
        scores.append(np.mean(topic_score))

    score = np.around(np.mean(scores), 5)
    print('NPMI: %f' % score)
    

if __name__ == '__main__':
    main()
