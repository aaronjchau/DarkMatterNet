# DarkMatterNet
"DarkMatterNet" is a neural network which predicts the mass of dark matter surrounding a galaxy when given data on three observable features of that galaxy. DarkMatterNet was trained on data from the Illustris-1 Hydrodynamical simulation. After tuning the hyperparameters to ensure our model was robust, we used DarkMatterNet to predict the dark matter mass of the galaxies (with stellar mass > 10<sup>8</sup> M⊙) in the NYU Value-Added Catalogue.

#### Input Features
* Gas Metallicity
* Stellar Mass
* B1000 [proportion of star formation which occurred in the past 1 billion years]

#### Output
* Dark Matter Halo Mass

#### Training Data: Illustris Simulation
* 22,070 dark matter halos, all with stellar mass > 10<sup>8</sup> M⊙
  * ~18,000 for training
  * ~4,000 for evaluation

#### Prediction Data: NYU Value-Added Galaxy Catalogue
* 1,002,145 galaxies, all with stellar mass > 10<sup>8</sup> M⊙


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
3. Clone this repo:
```bash
git clone https://github.com/chauaj1/DarkMatterNet.git
```
4. Change directory to repo location:
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
2. Place those 2 files in repo location.
3. In `DarkMatterNet.py`, specify the Model Directory where the TensorFlow checkpoint files should be written:
```python
regressor = tf.estimator.DNNRegressor(
                      hidden_units=[4,8,4],
                      feature_columns=feature_cols,
                      model_dir="/path/to/the/chosen/directory",
                      optimizer='Adam',
                      activation_fn=tf.nn.relu,
                      dropout=None,
                      loss_reduction=tf.losses.Reduction.SUM
                      )
```

4. Run `DarkMatterNet.py` to initialize and train a new model of the neural network. Performance metrics of the neural network on the Test Set will be printed. Predictions will also be generated based on the newly trained model for all galaxies in the NYU.csv file. Run the script:
```bash
python DarkMatterNet.py
```

5. The checkpoint files of the new model will be in the specified Model Directory. When `DarkMatterNet.py` is run again, training can continue on the existing model if the checkpoint files are left alone. To train a new model, delete the checkpoint files before running `DarkMatterNet.py` again.


## Our Results
### Hyperparameter Tuning
The hyperparameters of a neural network are highly sensitive to the data and are not able to be determined through a formula. The optimal hyperparameter values are determined through trials of random configurations. The "best" hyperparameters result in a neural network which has the lowest Mean Squared Error (MSE) for the Train Set and a similar MSE for the Evaluation Set. If the Evaluation MSE is significantly higher than the Train MSE, the neural network is overfitting. Here are the hyperparameters we tuned:  

###### Hidden Layer Architectures
  * Tried 2, 3, 4 layers
  * Tried various combinations of layers with 3, 4, 5, 6, 7, 8 neurons
  * All layers fully connected
  * *BEST: 3 layers of 8, 8, 4 neurons*

###### Gradient Descent Algorithms
  * Tried Adam, Adagrad, Ftrl, RMSProp (TensorFlow Optimizers)
  * *BEST: Adam*

###### Activation Functions
  * Tried RELU, ELU, SELU, Sigmoid, tanH functions
  * *BEST: RELU (Rectified Linear Unit)*

###### Dropout
  * Tried none, 0.01, 0.1, 0.2, 0.5, 0.6
  * *BEST: none*

###### Batch Sizes
* Tried 1024, 2048, 4096, 8192, 16384
* *BEST: 8192*



### Best Model Performance
The mean squared error was used as the loss function. Since the neural network only has 1 output (Halo Mass), the loss per halo is simply the squared error. The average loss is the average squared error for all the halos in dataset. Below, we refer to the average loss as the Mean Squared Error. It is important to note that this refers to the average loss across all halos in the dataset.

###### Train Set
* Mean Squared Error = 151.582275
* Root Mean Squared Error = 12.311875

###### Evaluation (Test) Set
* Mean Squared Error = 114.741638
* Root Mean Squared Error = 10.711752


### Sample Predictions

The Evaluation (Test) Set consists of ~4,000 dark matter halos from Illustris-1, which have never been "seen" by the neural network. The predictions were generated by the best model of DarkMatterNet.

Eval Set: Sample # | Gas Metallicity | Stellar Mass (10<sup>10</sup> M⊙) | B1000 | Predicted Halo Mass (10<sup>10</sup> M⊙)
:---: | :---: | :---: | :---: | :---:
1 | 0.053083 | 5.878160 | 0.034669 | 16.391450882  [Actual: 17.420477]
2 | 0.036112 | 4.168655 | 0.000014 | 13.1923856735  [Actual: 8.716017]
3 | 0.000000 | 5.255516 | 0.000000 | 1.95225942135  [Actual: 8.600364]
4 | 0.047815 | 1.999660 | 0.007655 | 5.72952461243  [Actual: 9.483207]
5 | 0.034940 | 2.078924 | 0.039930 | 8.37151622772   [Actual: 7.4421744]

The NYU Set consists of ~1,000,000 galaxies, which have never been "seen" by the neural network. The predictions were generated by the best model of DarkMatterNet.

NYU Set: Sample # | Gas Metallicity | Stellar Mass (10<sup>10</sup> M⊙) | B1000 | Predicted Halo Mass (10<sup>10</sup> M⊙)
:---: | :---: | :---: | :---: | :---:
1 | 0.023889 | 1.955420 | 0.053680 | 9.49759101868
2 | 0.032711 | 5.491253 | 0.240013 | 21.3492698669
3 | 0.026060 | 3.680842 | 0.095491 | 15.1347465515
4 | 0.046434 | 4.414605 | 0.657735 | 9.39928245544
5 | 0.024088 | 13.619882 | 0.053322 | 45.4226417542



### True vs Predicted  Plot
The plot below helps to visualize where DarkMatterNet struggles with predictions (lower mass halos) and where DarkMatterNet has more success (higher mass halos). The plot contains all the predictions for the Evaluation (Test) Set.   

![Best Model Ture vs Predicted Plot](https://imgur.com/foGXEpc.png)
