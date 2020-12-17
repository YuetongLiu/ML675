import numpy as np
import scipy.sparse as sparse
import os
import nltk
from collections import defaultdict

if not nltk.data.find('corpora/stopwords.zip'):
    nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

def clean_text(text, max_words):
    """
    Remove stopwords, punctuation, and numbers from text.

    Args:
        text: article text
        max_words: number of words to keep after processing

    Returns:
        Space-delimited and cleaned string
    """
    # tokenize text
    tokens = nltk.word_tokenize(text)

    # remove stopwords
    tokens = [token.lower().strip() for token in tokens if token.lower() not in stopwords]

    # remove tokens without alphabetic characters (i.e. punctuation, numbers)
    tokens = [token for token in tokens if any(t.isalpha() for t in token)]

    # skipping first ~20 words, which are often introductory
    return tokens[20:20+max_words]


def build_dtm(datadir, num_docs, max_words):
    """
    Build document-word matrix.

    Args:
        datadir: directory where speeches are located
        num_docs: number of documents to include
        max_words: number of words to keep after processing

    Returns:
        A compressed sparse row matrix of floats with shape
            [num_examples, num_features].
    """
    print('Building document-word matrix...')
    files = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
    if num_docs == 0:
        num_docs = len(files)

    docs = defaultdict(list)
    for i, file in enumerate(files[len(files)-num_docs:], start=1):
        with open(os.path.join(datadir, file)) as f:
            docs['doc' + str(i)] = clean_text(' '.join(f.readlines()), max_words)

    n_nonzero = 0
    vocab = set()
    for docterms in docs.values():
        unique_terms = set(docterms)
        vocab |= unique_terms
        n_nonzero += len(unique_terms)

    docnames = np.array(list(docs.keys()))
    vocab = np.array(list(vocab))  

    vocab_sorter = np.argsort(vocab)
    ndocs, nvocab = len(docnames), len(vocab)

    data = np.empty(n_nonzero, dtype=np.intc)
    rows = np.empty(n_nonzero, dtype=np.intc)
    cols = np.empty(n_nonzero, dtype=np.intc)

    ind = 0
    for docname, terms in docs.items():
        term_indices = vocab_sorter[np.searchsorted(vocab, terms, sorter=vocab_sorter)]

        uniq_indices, counts = np.unique(term_indices, return_counts=True)
        n_vals = len(uniq_indices)
        ind_end = ind + n_vals

        data[ind:ind_end] = counts
        cols[ind:ind_end] = uniq_indices
        doc_idx = np.where(docnames == docname)
        rows[ind:ind_end] = np.repeat(doc_idx, n_vals)

        ind = ind_end

    dtm = sparse.coo_matrix((data, (rows, cols)), shape=(ndocs, nvocab), dtype=np.intc)
    print('Document-word matrix built.')
    return dtm, vocab
