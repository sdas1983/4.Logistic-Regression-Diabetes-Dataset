# 4. Logistic Regression on Diabetes Dataset

This repository contains a Python script for performing logistic regression on the Diabetes dataset. The script includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

## Dataset

The dataset used is the [Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database), which contains various health-related features for predicting the presence or absence of diabetes.

## Script Overview

### 1. Data Loading and Basic Checks
- Reads the dataset from a CSV file.
- Displays basic information about the dataset, including missing values and descriptive statistics.

### 2. Data Preprocessing
- Replaces zero values in specific columns with the median of those columns.
- Detects and removes outliers using a custom function.
- Drops rows with missing values.

### 3. Exploratory Data Analysis (EDA)
- Generates a heatmap to show correlations between features.
- Creates various plots to visualize the data:
  - Count plot of the `Outcome` variable.
  - Facet grids for scatter plots and count plots based on the `Outcome` variable.
  - Pair plots for different types of visualizations.
  - Histograms and box plots for feature distributions and comparisons.

### 4. Model Building and Evaluation
- Splits the dataset into training and testing sets.
- Standardizes the features using `StandardScaler`.
- Trains a logistic regression model.
- Saves the trained model using `pickle`.
- Evaluates the model using:
  - Accuracy scores on training and test datasets.
  - Classification report.
  - Confusion matrix and ROC curve visualizations.

## Requirements

The following Python libraries are required:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pickle`

## Usage

1. **Load and Preprocess Data:**
   - Update the file path to the CSV in the script according to your local setup.
   - Run the script to load, preprocess, and visualize the data.

2. **Model Training and Evaluation:**
   - The script trains a logistic regression model and saves it as `Diabetes_Logistic_model.pickle`.
   - The script also evaluates the model and prints performance metrics.

## Example Output

- **Confusion Matrix:**
  Displays the confusion matrix for model evaluation.

- **ROC Curve:**
  Visualizes the ROC curve for model performance assessment.

- **Feature Correlation Heatmap:**
  Shows the correlation between different features in the dataset.

- **Count Plot and Facet Grids:**
  Visualizes the distribution of data across various features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
