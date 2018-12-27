import numpy as np
import pandas as pd
import tensorflow as tf



# [Reminder] List of column names assigned when the Illustris dataset is loaded
CSV_COLUMN_NAMES = ['Ignored_Index',
                    'SubhaloBHMass',
                    'SubhaloBHMdot',
                    'SubhaloGasMetallicity',
                    'SubhaloGasMetallicityHalfRad',
                    'SubhaloGasMetallicityMaxRad',
                    'SubhaloGasMetallicitySfr',
                    'SubhaloGasMetallicitySfrWeighted',
                    'SubhaloGrNr',
                    'SubhaloHalfmassRad',
                    'SubhaloIDMostbound',
                    'SubhaloLen',
                    'SubhaloMass',
                    'SubhaloMassInHalfRad',
                    'SubhaloMassInMaxRad',
                    'SubhaloMassInRad',
                    'SubhaloParent',
                    'SubhaloSFR',
                    'SubhaloSFRinHalfRad',
                    'SubhaloSFRinMaxRad',
                    'SubhaloSFRinRad',
                    'SubhaloStarMetallicity',
                    'SubhaloStarMetallicityHalfRad',
                    'SubhaloStarMetallicityMaxRad',
                    'SubhaloStellarPhotometricsMassInRad',
                    'SubhaloStellarPhotometricsRad',
                    'SubhaloVelDisp',
                    'SubhaloVmax',
                    'SubhaloVmaxRad',
                    'SubhaloWindMass',
                    'SubhaloSublinkID',
                    'SubhaloHalfmassRadType0',
                    'SubhaloHalfmassRadType1',
                    'SubhaloHalfmassRadType2',
                    'SubhaloHalfmassRadType3',
                    'SubhaloHalfmassRadType4',
                    'SubhaloHalfmassRadType5',
                    'SubhaloMassInHalfRadType0',
                    'SubhaloMassInHalfRadType1',
                    'SubhaloMassInHalfRadType2',
                    'SubhaloMassInHalfRadType3',
                    'SubhaloMassInHalfRadType4',
                    'SubhaloMassInHalfRadType5',
                    'SubhaloMassInMaxRadType0',
                    'SubhaloMassInMaxRadType1',
                    'SubhaloMassInMaxRadType2',
                    'SubhaloMassInMaxRadType3',
                    'SubhaloMassInMaxRadType4',
                    'SubhaloMassInMaxRadType5',
                    'SubhaloMassInRadType0',
                    'SubhaloMassInRadType1',
                    'SubhaloMassInRadType2',
                    'SubhaloMassInRadType3',
                    'SubhaloMassInRadType4',
                    'SubhaloMassInRadType5',
                    'SubhaloMassType0',
                    'SubhaloMassType1',
                    'SubhaloMassType2',
                    'SubhaloMassType3',
                    'SubhaloMassType4',
                    'SubhaloMassType5',
                    'SubhaloStellarPhotometricsU',
                    'SubhaloStellarPhotometricsB',
                    'SubhaloStellarPhotometricsV',
                    'SubhaloStellarPhotometricsK',
                    'SubhaloStellarPhotometricsg',
                    'SubhaloStellarPhotometricsr',
                    'SubhaloStellarPhotometricsi',
                    'SubhaloStellarPhotometricsz',
                    'SubhaloID',
                    'B300',
                    'B1000',
                    'TotalSFRMass']




def raw_dataframe():
    """Function which loads entire Illustris dataset"""

    # NOTE: Illustris_V2.csv pre-filtering: halos with 0 stellar mass (PhotometricsMassInRad) are removed
    # Load data from Illustris_V2.csv and reassign column names
    df = pd.read_csv("Illustris_V2.csv",
                        header=0,
                        names=CSV_COLUMN_NAMES,
                        dtype=np.float64
                    )

    return df



def load_data(label_name='SubhaloMassInRad', train_fraction=0.8, seed=None):
    """Function which loads the Illustris dataset and returns Train Set and Test Set dataframes"""

    # Use raw_dataframe() to load entire Illustris dataset
    data = raw_dataframe()

    # Remove halos with stellar mass < 10^8 Mstar
    data_stellar_cut = data.drop(data[data.SubhaloStellarPhotometricsMassInRad < 0.01].index)

    # Split Illustris dataframe randomly into a Train Set (80% of data) and a Test Set (20% of data)
    np.random.seed(seed)
    train_features = data_stellar_cut.sample(frac=train_fraction, random_state=seed)
    test_features = data_stellar_cut.drop(train_features.index)

    # Remove Halo Mass (label) from Train and Test Set and assign to new dataframes
    train_label = train_features.pop(label_name)
    test_label = test_features.pop(label_name)

    # Return a pair of dataframes (features only, label only) for Train Set, Test Set
    return (train_features, train_label), (test_features, test_label)



def make_dataset(features, label=None):
    """Function which takes in {features} and/or {label} dataframes and returns a TF_Dataset"""

    # Make a dictionary with column names as keys and all following rows as values
    features = dict(features)

    # Convert values (pd.Series) to np.arrays
    for key in features:
        features[key] = np.array(features[key])

    # Make a list with the dict of features as the first item
    items = [features]

    # If there are labels (i.e. for training data, NOT prediction data), add them as the second item
    if label is not None:
        items.append(np.array(label, dtype=np.float64))

    # Create a TF_Dataset, where each element contains {features} and {the label} for a single halo
    return tf.data.Dataset.from_tensor_slices(tuple(items))
