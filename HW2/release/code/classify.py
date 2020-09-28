""" Main file. This is the starting point for your code execution. 

You shouldn't need to change much of this code, but it's fine to as long as we
can still run your code with the arguments specified!
"""

import os
import argparse
import pickle
import numpy as np

import models
from data import load_data


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your models.")

    parser.add_argument("--data", type=str, required=True, help="The data file to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create (for training) or load (for testing).")
    parser.add_argument("--algorithm", type=str,
                        choices=['perceptron', 'logistic'],
                        help="The name of the algorithm to use. (Only used for training; inferred from the model file at test time.)")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create. (Only used for testing.)")

    # TODO: Add optional command-line arguments as necessary (learning rate and training iterations).

    args = parser.parse_args()

    return args


def check_args(args):
    mandatory_args = {'data', 'mode', 'model_file', 'algorithm', 'predictions_file'}
    if not mandatory_args.issubset(set(dir(args))):
        raise Exception('Arguments that we provided are now renamed or missing. If you hand this in, you will get 0 points.')

    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--model should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--predictions-file should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")

def test(args):
    """ 
        Make predictions over the input test dataset, and store the predictions.
    """
    # load dataset and model
    X, _ = load_data(args.data)
    model = pickle.load(open(args.model_file, 'rb'))

    # predict labels for dataset
    y_hat = model.predict(X)
    invalid_label_mask = (y_hat != 0) & (y_hat != 1)
    if any(invalid_label_mask):
        raise Exception('All predictions must be 0 or 1, but found other predictions.')

    # output model predictions
    np.savetxt(args.predictions_file, y_hat, fmt='%d')


def train(args):
    """ Fit a model's parameters given the parameters specified in args.
    """
    X, y = load_data(args.data)

    # build the appropriate model
    if args.algorithm == "perceptron":
        model = models.Perceptron(nfeatures=X.shape[1])
    elif args.algorithm == "logistic":
        model = models.LogisticRegression(nfeatures=X.shape[1])
    else:
        raise Exception("Algorithm argument not recognized")

    # Run the training loop
    for epoch in range(args.online_training_iterations):
        model.fit(X=X, y=y, lr=args.online_learning_rate)

    # Save the model
    pickle.dump(model, open(args.model_file, 'wb'))


if __name__ == "__main__":
    args = get_args()
    check_args(args)

    if args.mode.lower() == 'train':
        train(args)
    elif args.mode.lower() == 'test':
        test(args)
    else:
        raise Exception("Mode given by --mode is unrecognized.")