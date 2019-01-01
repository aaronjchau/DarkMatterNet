import numpy as np
import pandas as pd
import tensorflow as tf


# [Reminder] List of column names assigned when the Illustris dataset is loaded
ILLUSTRIS_COLUMN_NAMES = ['Pre_Drop_Index',
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

# [Reminder] List of column names assigned when the NYU dataset is loaded
# NOTE: NYU column names must match Illustris column names
NYU_COLUMN_NAMES = ['Pre_Drop_Index',
                    'SubhaloStellarPhotometricsMassInRad',
                    'SubhaloGasMetallicity',
                    'B300',
                    'B1000']



def raw_dataframe():
    """Function which loads Illustris and NYU datasets"""

    # Load data from Illustris_V2.csv and reassign column names
    #   NOTE: Illustris_V2.csv pre-filtering: halos with 0 stellar mass (PhotometricsMassInRad) were dropped
    iDF = pd.read_csv("Illustris_V2.csv",
                        header=0,
                        names=ILLUSTRIS_COLUMN_NAMES,
                        dtype=np.float64)

    # Load data from NYU.csv and reassign column names
    #   NOTE: NYU.csv pre-filtering: halos with 0 stellar mass were dropped
    nDF = pd.read_csv("NYU.csv",
                        header=0,
                        names=NYU_COLUMN_NAMES,
                        dtype=np.float64)

    return iDF, nDF



def load_data(label_name='SubhaloMassInRad', train_fraction=0.8, seed=None):
    """Function which loads Illustris, NYU datasets; returns filtered Train & Test Set for Illustris, Predict Set for NYU"""

    # Use raw_dataframe() to load Illustris and NYU datasets
    iData, nData = raw_dataframe()

    # Convert NYU stellar mass to units of 10^10 Mstar, like in Illustris
    nData.SubhaloStellarPhotometricsMassInRad /= (10**10)

    # Illustris: remove halos with stellar mass < 10^8 Mstar
    iData_stell_cut = iData.drop(iData[iData.SubhaloStellarPhotometricsMassInRad < 0.01].index)

    # NYU: remove halos with stellar mass < 10^8 Mstar
    nData_stell_cut = nData.drop(nData[nData.SubhaloStellarPhotometricsMassInRad < 0.01].index)

    # Split Illustris dataframe randomly into a Train Set (80% of data) and a Test Set (20% of data)
    np.random.seed(seed)
    train_features = iData_stell_cut.sample(frac=train_fraction, random_state=seed)
    test_features = iData_stell_cut.drop(train_features.index)

    # Remove Halo Mass (label) from Train and Test Set and assign to new dataframes
    train_label = train_features.pop(label_name)
    test_label = test_features.pop(label_name)

    # Clean name
    NYU_features = nData_stell_cut

    # Return a pair of dataframes (features only, label only) for Train Set, Test Set
    return (train_features, train_label), (test_features, test_label), NYU_features



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



def predict_input_fn(features, labels):
    """Modified input function for compatability with new Predict method from TF.Estimator; will combine into make_dataset()"""
    features = dict(features)

    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    return tf.data.Dataset.from_tensor_slices(inputs)
