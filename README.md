# MLBD-Neutrinos
Master's Thesis for the MRes on Machine Learning and Big Data in the Physical Sciences at Imperial College London

Hadronic Tensor Regression with XGBoost
----------------------------------------------

Project Overview
----------------
This project is focused on implementing a regression model to predict the real and imaginary components of the hadronic tensor in neutrino-nucleus interactions, using XGBoost. The primary goal is to predict the tensor components based on the kinematic variables of the interaction.

The script provided is modular, containing two main classes:
1. DataPrep: Handles data preprocessing and prepares it for training, validation, and testing.
2. Boosted_Regression: Implements the XGBoost regression model, including training, evaluation, and visualization.

Data
----
Input Data

The input data comes from code simulating the theoretical nuclear models of neutrino-nucleus interactions(RMF, ED-RMF, RPWIA, ...) and is loaded from a `.out` file (e.g., `/Hmunu.out`). The data represents various hadronic tensor components across different kinematic variables.

- Columns in the input file:
  - T: Nucleon kinetic energy
  - θ: Nucleon scattering angle
  - p3: Missing momentum
  - E: Missing energy
  - ij: Independent tensor component index (0-9)
  - Re: Real part of the tensor component
  - Im: Imaginary part of the tensor component

Output

The model predicts the real (Re) and imaginary (Im) parts of each tensor component, and outputs various plots and statistics to assess model performance.

Script Structure
-----------------
1. DataPrep Class
This class handles all the data preprocessing steps, including:
  - Data splitting: It divides the dataset into the real and imaginary parts of the tensor and combines them with the kinematic variables for further processing.
  - Train/test/validation split: The data is split into 70% training, 15% validation, and 15% test sets.
  - Scaling/Transformation: Uses Yeo-Johnson transformation to scale the target variables if needed.
  - Contour Plots: Visualizes the tensor components across different kinematic variables in 2D space.

2. Boosted_Regression Class
This class implements the regression model using XGBoost:
  - Model Training: Trains the XGBoost model on the training data.
  - Model Evaluation: Evaluates the model's performance on training, validation, and test sets.
  - Model Saving/Loading: Trained models are saved and loaded for future use.
  - Hyperparameter Tuning: Uses GridSearch to find the best hyperparameters for the model.
  - Learning Curves: Visualizes how RMSE evolves with the number of trees.
  - Residual Analysis: Plots the residuals and identifies outliers.
  - Prediction Visualization: Compares predicted vs actual values for each tensor component.
  - Error Visualization: Plots absolute errors of the model's predictions.

Usage
-----
Pre-requisites
- Python 3.x
- Required Python libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - sklearn
  - xgboost

Running the Script
------------------
1. Data Loading: Update the paths in the script to point to your dataset files (Hmunu.out).
2. Data Preprocessing: The script will preprocess the data using the DataPrep class.
3. Model Training and Evaluation:
   - Train the model using the train() method.
   - Save or load the trained model using save() and load().
4. Evaluate the Model: Use evaluation() to evaluate the model on different datasets (e.g., test or validation sets).
5. Visualization:
   - Plot learning curves using learning_curves().
   - Visualize RMSE using plot_rmse() or R² scores with plot_r2().
   - Plot model predictions for a specific kinematic variable using plot_predictions().
   - Analyze the residuals and outliers using residuals() and errors().
  
Example Commands
----------------
```python
# Preparing the data
data = DataPrep(table).sets().split(scaling=True)

# Initializing the model
model = Boosted_Regression(data, 'Oxygen', 7, 'low')

# Training the model
model.train()

# Saving the model
model.save()

# Loading the model
model.load()

# Evaluating the model on test set
model.evaluation('test')

# Visualizing r2 and residuals
model.plot_r2()
model.residuals('Re', 2)
```

Customization
-------------

1. Path Configurations:
   * Update the paths in the Boosted_Regression class to specify where models and data files are stored.
   * Modify the table file paths according to where tables are stored.

2. Hyperparameters:
   * Modify the hyperparameters for XGBoost in the train() function or perform hyperparameter tuning using hp_tuning().

3. Outlier Removal:
   * There is a built-in option to remove outliers based on large jumps in certain tensor components. This can be customized for different datasets.

4. Scaling/Transforming:
   * The script currently uses the Yeo-Johnson transformation. This can be changed to a different scaling method in the split() function of the DataPrep class.

Visualization
-------------

The script produces various types of visualizations, including:
   * Contour plots of tensor components over phase space.
   * Learning curves for RMSE.
   * Bar plots for RMSE and R².
   * Plots of predicted vs actual tensor values.
   * Residual and outlier analysis.

Proposed Future Work
----------------------

1. Current Model Improvement:
   * Combine low and high TN tables.
   * Tune hyper-parameters according to combined dataset.
   * Incorporate a weighting scheme for outliers based on their residuals.

2. Unified Model:
   * Combine all tables together and train a singular model on them.
   * Singular ML model can help in the incorporation of systematic uncertainties.
