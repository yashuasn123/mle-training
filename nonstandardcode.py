import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import tarfile
from six.moves import urllib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV
import yaml


# Define the root URL for downloading the dataset and set up paths for storing it
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Fetch the housing data from the provided URL and extract it to the specified path.

    This function downloads the housing dataset from the given URL, and extracts it 
    into the specified directory (housing_path). If the directory doesn't exist, it 
    creates the directory.

    Args:
    housing_url (str): The URL of the housing dataset to be downloaded.
    housing_path (str): The local directory where the dataset will be stored.

    Returns:
    None
    """
    os.makedirs(housing_path, exist_ok=True)  # Create the directory if it doesn't exist
    tgz_path = os.path.join(housing_path, "housing.tgz")  # Path for the .tgz file
    urllib.request.urlretrieve(housing_url, tgz_path)  # Download the .tgz file
    housing_tgz = tarfile.open(tgz_path)  # Open the .tgz file
    housing_tgz.extractall(path=housing_path)  # Extract all contents of the file
    housing_tgz.close()  # Close the .tgz file

def load_housing_data(housing_path=HOUSING_PATH):
    """
    Load housing data from a CSV file into a pandas DataFrame.

    This function reads a CSV file containing housing data located at the specified
    path and returns it as a pandas DataFrame. The default path is set to 
    the global variable `HOUSING_PATH`.

    Args:
    housing_path (str): The directory path where the 'housing.csv' file is located.
                        Defaults to the value of `HOUSING_PATH`.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the housing data loaded from the CSV file.
    """
    
    csv_path = os.path.join(housing_path, "housing.csv")  # Path for the CSV file
    return pd.read_csv(csv_path)  # Load the CSV data into a pandas DataFrame

# Load the housing data into a DataFrame
housing = load_housing_data()



# Split the data into training and test sets
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Create income categories for stratified sampling
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])



# StratifiedShuffleSplit for splitting the data while preserving income category proportions
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]  # Training data after stratification
    strat_test_set = housing.loc[test_index]  # Test data after stratification

def income_cat_proportions(data):
    """
    Calculate the proportions of income categories in the given DataFrame.

    This function computes the relative proportions of each unique income category 
    found in the 'income_cat' column of the provided DataFrame. The proportions 
    are calculated as the count of each category divided by the total number of 
    entries in the DataFrame, providing insight into the distribution of income 
    categories.

    Args:
    data (pd.DataFrame): A pandas DataFrame that must contain an 'income_cat' column, 
                         which holds the income category values.

    Returns:
    pd.Series: A pandas Series containing the proportions of each income category,
               indexed by the category labels. The values represent the fraction of 
               the total dataset that each category comprises.
    """
    return data["income_cat"].value_counts() / len(data)

# Compare stratified vs random sampling for income categories
compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()

# Calculate the percentage error for random and stratified sampling
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

# Remove the income category column from the training and test sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Make a copy of the stratified training set for exploration
housing = strat_train_set.copy()

# Plot housing data on a scatter plot
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)  # Add transparency

# Compute the correlation matrix and look at correlations with 'median_house_value'
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# Add new features to the dataset (feature engineering)
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

# Separate the target label from the training data
housing = strat_train_set.drop("median_house_value", axis=1)  # Drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()  # Store labels separately



# Impute missing values with the median strategy
imputer = SimpleImputer(strategy="median")

# Prepare numerical data (excluding the 'ocean_proximity' categorical column)
housing_num = housing.drop('ocean_proximity', axis=1)

# Fit the imputer to the numerical data and transform it
imputer.fit(housing_num)
X = imputer.transform(housing_num)

# Convert the transformed data back into a DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

# Recompute feature-engineered columns for the transformed data
housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

# One-hot encode the categorical 'ocean_proximity' feature
housing_cat = housing[['ocean_proximity']]
housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

# Train a linear regression model

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)  # Fit the model

# Make predictions and compute RMSE for linear regression

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# Compute MAE for linear regression

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae

# Train a decision tree regressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)  # Fit the decision tree model

# Make predictions and compute RMSE for decision tree
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# Set up a random forest regressor with hyperparameter tuning



# Define hyperparameter distributions for randomized search
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }



# Load the configuration from the config.yaml file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract values from the config file
random_state = config['random_forest']['random_state']
param_grid = config['param_grid']
grid_search_params = config['grid_search']

# Initialize the RandomForestRegressor with parameters from config
forest_reg = RandomForestRegressor(random_state=random_state)

# Set up GridSearchCV with parameters from the config file
grid_search = GridSearchCV(forest_reg, param_grid=param_grid, 
                           cv=grid_search_params['cv'],
                           scoring=grid_search_params['scoring'],
                           return_train_score=grid_search_params['return_train_score'])

# Print results from the randomized search
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Set up grid search for hyperparameter tuning


# Define the grid of hyperparameters to search
param_grid = [
    # Try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # Try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

# Perform grid search for random forest hyperparameters
forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# Get the best parameters and display results from grid search
grid_search.best_params_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Display feature importances from the best model
feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

# Final model and test set evaluation
final_model = grid_search.best_estimator_

# Prepare the test set
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# Process numerical and categorical test set features
X_test_num = X_test.drop('ocean_proximity', axis=1)
X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(X_test_prepared, columns=X_test_num.columns, index=X_test.index)
X_test_prepared["rooms_per_household"] = X_test_prepared["total_rooms"] / X_test_prepared["households"]
X_test_prepared["bedrooms_per_room"] = X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
X_test_prepared["population_per_household"] = X_test_prepared["population"] / X_test_prepared["households"]

# One-hot encode categorical test features
X_test_cat = X_test[['ocean_proximity']]
X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

# Make final predictions and compute RMSE on the test set
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

