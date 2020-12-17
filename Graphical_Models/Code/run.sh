#!/bin/bash -e

DATADIR="speeches/"
NUM_DOCUMENTS=0 # this means use all documents
MAX_WORDS=1000
# TODO: specify other command line flags here

echo "Running inference..."
python3 main.py \
    --datadir $DATADIR \
    --num-documents $NUM_DOCUMENTS \
    --max-words $MAX_WORDS \
    --predictions-file "docs$NUM_DOCUMENTS.words$MAX_WORDS.predictions" \

echo "Computing topic coherence..."
python3 compute_topic_coherence.py \
    --datadir $DATADIR \
    --num-documents $NUM_DOCUMENTS \
    --max-words $MAX_WORDS \
    --predictions-file "docs$NUM_DOCUMENTS.words$MAX_WORDS.predictions" \