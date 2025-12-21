import mlflow
import mlflow.sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

wine = load_wine()  # Load the wine dataset
X = wine.data  # Features
y = wine.target  # Target labels
X_train, X_test, y_train, y_test = train_test_split(
    # Split the data into training and testing sets
    X, y, test_size=0.10, random_state=42)
max_depth = 10  # Set maximum depth of the tree
n_estimators = 5  # Set number of trees in the forest
