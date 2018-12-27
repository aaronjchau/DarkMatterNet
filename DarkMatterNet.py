from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
import argparse
import math
import os

import DMN_Data


"""
INPUTS {FEATURES}:      Gas Metallicity
                        Stellar Mass
                        B1000

OUPUT {LABEL}:          Halo Mass

DATA:                   Illustris_1 Simulation - Group Catalog from Snap 135
"""



parser = argparse.ArgumentParser()

# Specify batch size - number of halos fed into NN during each step
parser.add_argument('--batch_size',
                        default=8192,
                        type=int,
                        help='batch size')

# Specify number of steps
parser.add_argument('--train_steps',
                        default=500,
                        type=int,
                        help='number of training steps')



def from_dataset(ds):
    return lambda: ds.make_one_shot_iterator().get_next()



def main(argv):
    args = parser.parse_args(argv[1:])

    # Use DMN_Data.load_data() to load Illustris data and split into Train / Test Sets:
    #   1. Reads in Illustris data [pre-processing: halos with 0 stellar mass removed; no randomization]
    #   2. Removes halos with stellar mass < 10^8 Mstar
    #   3. Splits Illustris data randomly into a Train Set [80%] and a Test Set [20%]
    #   4. For Train Set and Test Set, generates a pair of dataframes: Features, Label
    #           Features dataframe: includes all columns from Illustris data, except Halo Mass
    #           Label dataframe: only includes column for Halo Mass
    (train_features, train_label), (test_features, test_label) = DMN_Data.load_data()



    # Use DMN_Data.make_dataset() which harnesses the tf.data.Dataset API for the Train Set input pipline:
    #   1. Takes in a pair of dataframes, Features dataframe and Label dataframe
    #   2. Generates a "TF_Dataset", where each element corresponds to 1 halo and contains 2 items: Features, Label
    #   3. Shuffles the elements [halos] of the TF_Dataset so the data are randomized; buffer size must be > dataframe length
    #   4. Batch elements [halos] as they are fed into the neural network during each step
    #   5. Continue shuffling and batching elements as long as the neural network runs
    train = (DMN_Data
                .make_dataset(train_features, train_label)
                .shuffle(30000)
                .batch(args.batch_size)
                .repeat()
            )


    # Use DMN_Data.make_dataset() which harnesses the tf.data.Dataset API for the Test Set input pipline:
    #   NOTE: The Test Set does not need to be shuffled because it will have no effect on training
    test = (DMN_Data
                .make_dataset(test_features, test_label)
                .batch(args.batch_size)
            )



    # Store mean, std dev, etc. of Train Set for normalization
    train_stats = train_features.describe()

    # Select input Features by defining them as a list of TF Feature Columns; specify normalization function based on Train Set stats
    feature_cols = [tf.feature_column.numeric_column(
                                        key='SubhaloGasMetallicity',
                                        dtype=tf.float64,
                                        normalizer_fn=lambda x: (x - train_stats.SubhaloGasMetallicity['mean'])
                                                                    / train_stats.SubhaloGasMetallicity['std']),
                    tf.feature_column.numeric_column(
                                        key='SubhaloStellarPhotometricsMassInRad',
                                        dtype=tf.float64,
                                        normalizer_fn=lambda x: (x - train_stats.SubhaloStellarPhotometricsMassInRad['mean'])
                                                                    / train_stats.SubhaloStellarPhotometricsMassInRad['std']),
                    tf.feature_column.numeric_column(
                                        key='B1000',
                                        dtype=tf.float64,
                                        normalizer_fn=lambda x: (x - train_stats.B1000['mean'])
                                                                    / train_stats.B1000['std'])
                    ]



    # Instantiate a high-level TF Estimator [Deep Neural Network] and define the hyperparameters:
    #   NOTE: The loss function is mean squared error (MSE) which is (true_halo_mass - predicted_halo_mass)^2
    #   1. Hidden Units refer to the number of neurons in each layer of the NN
    #   2. Feature Columns define the input features to be used by the NN
    #   3. Model Directory points to the path where checkpoint files are written
    #   4. Optimizer refers to the gradient descent algoirthm chosen from the list of tf.Optimizer's
    #   5. Activation Function refers to the function that is applied to each layer of the NN
    #   6. Dropout refers to the probability that a given coordinate will be removed
    #   7. Loss Reduction describes how to reduce training loss over the batch, by minimizing scalar sum of weighted losses
    regressor = tf.estimator.DNNRegressor(
                        hidden_units=[4,8,4],
                        feature_columns=feature_cols,
                        model_dir="...",
                        optimizer='Adam',
                        activation_fn=tf.nn.relu,
                        dropout=None,
                        #loss_reduction=losses.Reduction.SUM
                        )


    # Call the Train method of the TF Estimator to train the NN on the Train Set for the specificed number of steps
    regressor.train(
                input_fn=from_dataset(train),
                steps=args.train_steps
                )



    # Call the Evaluate method to generate metrics of the trained NN on the Train Set:
    #   NOTE: Since the output layer has only 1 neuron [halo mass], the loss [mean squared error] is just the squared error
    #   NOTE: The from_dataset(train) includes .repeat() so an endpoint must be set; 2 steps = 2 batches = 2*batch_size halos
    #   1. Loss refers to mean loss per mini-batch [mean squared error for batch of halos]
    #   2. Average Loss refers to the mean loss per sample [mean squared error for entire Train Set]
    train_eval_result = regressor.evaluate(
                                input_fn=from_dataset(train),
                                steps=2,
                                name='TRAIN_SET'
                                )

    # Store the Average Loss [mean squared error for Train Set] value for printing
    train_average_loss = train_eval_result["average_loss"]

    # Calculate the Root Mean Squared Error of the Train Set for printing
    train_rmse = math.sqrt(train_average_loss)

    # Print the Mean Squared Error and Root Mean Squared Error for a selection of the Train Set
    print("\n\n\nTrain Set MSE = {0:f} \nTrain Set RMSE = {1:f} 10^10 Mstar"
                            .format(train_average_loss, train_rmse))

    # Calculate the total number of halos that were evaluated from the Train Set
    train_eval_total = args.batch_size * 2

    # Print note on total halos in the Train Set
    print("\nNOTE: Train Set has {0:d} total halos."
                            .format(train_label.shape[0]))

    # Print note on number of halos evaluated from the Train Set
    print("NOTE: MSE and RMSE were calculated for 2 batches of {0:d}, which is {1:d} halos from the Train Set.\n\n\n"
                            .format(args.batch_size, train_eval_total))



    # Call the Evaluate method to generate metrics of the trained NN on the Test Set ("unseen data")
    test_eval_result = regressor.evaluate(
                                input_fn=from_dataset(test),
                                name='TEST_SET'
                                )

    # Store the Average Loss [mean squared error for Test Set] value for printing
    test_average_loss = test_eval_result["average_loss"]

    # Calculate the Root Mean Squared Error of the Test Set for printing
    test_rmse = math.sqrt(test_average_loss)

    # Print the Mean Squared Error of the Test Set
    print("\n\n\nTest Set MSE = {0:f} \nTest Set RMSE = {1:f} 10^10 Mstar"
                            .format(test_average_loss, test_rmse))

    # Print note on total halos in the Test Set
    print("\nNOTE: Test Set has {0:d} total halos."
                            .format(test_label.shape[0]))

    # Print note on number of halos evaluted from Test Set
    print("NOTE: MSE and RMSE were calculated for all {0:d} halos from the Test Set.\n\n\n"
                            .format(test_label.shape[0]))



    # Call the Predict method to use the trained NN to make predictions on the Test Set ("unseen data")
    #   NOTE: The Predict method will ignore the second item (label) of the tuple (features, label)
    test_predict_result = regressor.predict(
                                input_fn=from_dataset(test)
                                )

    # Store the predictions as a dataframe for printing
    test_predict_result_df = pd.DataFrame.from_dict(test_predict_result, dtype=float)


    # Print the first 10 predictions by the NN on the Test Set
    print("\n\nTest Set Predictions (First 10 Halos):\n")
    print(*(list(test_predict_result_df[:10].values.flatten())),sep='\n')


    # Print the first 10 actual values of the Test Set for comparison
    print("\n\n\nTest Set Actual Values (First 10 Halos):\n")
    print(*(list(test_label[:10].values.flatten())),sep='\n')
    print("\n\n")



    # Generate a plot of True Halo Mass vs Predicted Halo Mass for easier visualization
    plt.scatter(test_label, test_predict_result_df)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,100])
    plt.ylim([0,100])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()





    # EXTRA NOTE: To call Tensorboard: 'tensorboard --logdir=...'
    # EXTRA NOTE: To turn on TensorFlow logging, uncomment the phrase below

if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
