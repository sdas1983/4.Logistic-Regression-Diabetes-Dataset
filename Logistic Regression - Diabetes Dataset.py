# Logistic Regression: Diabetes Dataset

import pandas as pd
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import pickle

# Load the dataset
df = pd.read_csv(r"C:\Users\das.su\OneDrive - GEA\Documents\PDF\Machine Learning\BIT ML, AI and GenAI Course\diabetes.csv")

# Basic Checks
print("Checking for missing values and data info:")
print(df.isnull().sum())
print(df.isna().sum())
print(df.info())
print(df.columns)
print(df.describe())
print(df.head(10))

# Replace 0 values with median
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].replace(0, df['DiabetesPedigreeFunction'].median())

print("\nData after replacing 0 values with median:")
print(df.describe())

# Visualization
print("\nGenerating visualizations:")

# Heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(data=df.corr(), annot=True)
plt.title('Heatmap of Correlations')
plt.show()

# Count plot of Outcome
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df, x='Outcome')
for bars in ax.containers:
    ax.bar_label(bars)
plt.title('Count Plot of Outcome')
plt.show()

# FacetGrid visualizations
for kind in ['scatter', 'count']:
    g = sns.FacetGrid(df, col='Outcome')
    g.map(sns.scatterplot, 'Pregnancies', 'Age') if kind == 'scatter' else g.map(sns.countplot, 'Pregnancies')
    plt.title(f'FacetGrid of {kind} plot')
    plt.show()

# Pair plot of variables
for kind in ['scatter', 'kde', 'hist', 'reg']:
    sns.pairplot(df, hue='Outcome', kind=kind)
    plt.title(f'Pair Plot ({kind})')
    plt.show()

# Histogram Plot
fig = df.hist(figsize=(10, 10), rwidth=0.9)
plt.suptitle('Histogram Plot')
plt.show()

# Box Plot of Variables
plt.figure(figsize=(15, 7))
sns.boxplot(data=df)
plt.title('Box Plot of Variables')
plt.show()

# Box plot with subplots
plt.figure(figsize=(20, 10))
for i, feature in enumerate(['Glucose', 'BMI', 'Age'], 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=df, x='Outcome', y=feature)
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.show()

# Detect Outliers
def detect_outliers(data, cols):
    for x in cols:
        q75, q25 = np.percentile(data[x], [75, 25])
        intr_qr = q75 - q25
        max_val = q75 + (1.5 * intr_qr)
        min_val = q25 - (1.5 * intr_qr)
        data.loc[data[x] < min_val, x] = np.nan
        data.loc[data[x] > max_val, x] = np.nan

detect_outliers(df, ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

df = df.dropna(axis=0)
print("\nData after outlier removal:")
print(df.isnull().sum())

# Preparing for model building
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
regression = LogisticRegression()
regression.fit(X_train, y_train)

# Save the model
pickle.dump(regression, open('Diabetes_Logistic_model.pickle', 'wb'))

# Prediction
test_sample = scaler.transform([[6, 148, 72, 35, 0.0, 33.6, 0.6, 50]])
prediction = regression.predict(test_sample)
print(f"\nPrediction for the test sample: {prediction}")

# Model Evaluation
print("\nModel Evaluation:")
print(f"Test Score: {regression.score(X_test, y_test)}")
print(f"Train Score: {regression.score(X_train, y_train)}")

# Classification Report
y_pred = regression.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
ConfusionMatrixDisplay.from_estimator(regression, X_test, y_test)
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
print("\nROC Curve:")
RocCurveDisplay.from_estimator(regression, X_test, y_test)
plt.title('ROC Curve')
plt.show()
