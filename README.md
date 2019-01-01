# DarkMatterNet
"DarkMatterNet" is a neural network which predicts the mass of dark matter surrounding a galaxy when given data on three observable features of that galaxy. DarkMatterNet was trained on data from the Illustris-1 Hydrodynamical simulation. After tuning the hyperparameters to ensure our model was robust, we used DarkMatterNet to predict the dark matter mass of the galaxies (with stellar mass >10<sup>8</sup>) in the NYU Value-Added Catalogue.

**Input Features**
* Gas Metallicity
* Stellar Mass
* B1000 [proportion of star formation which occurred in the past 1 billion years]

**Output**
* Dark Matter Halo Mass

**Training Data: Illustris Simulation**
* 22,070 dark matter halos, all with stellar mass >10<sup>8</sup>
  * ~18,000 for training
  * ~4,000 for evaluation

**Prediction Data: NYU Value-Added Galaxy Catalogue**
* 1,002,145 galaxies, all with stellar mass >10<sup>8</sup>


## Datasets
For the training and evaluation of DarkMatterNet, we obtained the data on dark matter halos from Illustris-1. Gas Metallicity and Stellar Mass were obtained from Snapshot 135 (z=0). B1000 was obtained from Snapshots 129-135. The compiled data can be accessed and downloaded with a UCI Google Account.

[Illustris_V2.csv](https://drive.google.com/a/uci.edu/file/d/1gg0eeuadNAWKssdDmyB84M4uXk9Igo0i/view?usp=sharing) | [Source](http://www.illustris-project.org/data/downloads/Illustris-1/)

To test the predictive ability of DarkMatterNet, we used galaxies from the NYU Value-Added Galaxy Catalogue (collision_type: none, flux_type: SDSS model magnitudes, band_shift: 0). These K-Corrections included Gas Metallicity, Stellar Mass, and B1000. The dataset can be accessed and downloaded with a UCI Google Account.

[NYU.csv](https://drive.google.com/a/uci.edu/file/d/1B2RFeulePGP9NfdEODWOQ0bixO-vv3zv/view?usp=sharing) | [Source](http://cosmo.nyu.edu/blanton/vagc/kcorrect.html)

## Installation
1. Install virtualenv:
```bash
pip install virtualenv
```
2. Set up a new virtual environment:
```bash
virtualenv --system-site-packages -p python2.7 /path/to/new/virtual/env
```
3. Clone this repository:
```bash
git clone https://github.com/chauaj1/DarkMatterNet.git
```
4. Change directory to repo:
```bash
cd DarkMatterNet
```
5. Install necessary dependencies (assuming Python 2.7):
```bash
pip install tensorflow
pip install matplotlib
pip install pandas
pip install numpy
```

## Training, Testing, and Predicting
1. Download the 2 CSV files listed in the Datasets section.
2. Place those 2 files where the repo was cloned.
3. In `DarkMatterNet.py`, specify the Model Directory where the TensorFlow checkpoint files should be written:
>     regressor = tf.estimator.DNNRegressor(
          hidden_units=[4,8,4],
          feature_columns=feature_cols,
          model_dir="/path/to/the/correct/directory",
          optimizer='Adam',
          activation_fn=tf.nn.relu,
          dropout=None,
          loss_reduction=tf.losses.Reduction.SUM
>      )
4. Run `DarkMatterNet.py` to initialize and train a new neural network. Performance metrics of the neural network on the Test Set will be printed. Predictions will also be generated based on the newly trained model for all galaxies in the NYU.csv file. Run the script:
```bash
python DarkMatterNet.py
```

## Retraining
After a complete run of `DarkMatterNet.py`, the model checkpoint files will remain in the Model Directory. To continue training on the existing model, leave these files. To train a brand new neural network, delete these files before running `DarkMatterNet.py` again.
