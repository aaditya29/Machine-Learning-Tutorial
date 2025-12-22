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

mlflow.set_experiment('MLOPS-1')

with mlflow.start_run():
    rf = RandomForestClassifier(
        # Initialize the Random Forest Classifier
        max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)  # Train the model

    y_pred = rf.predict(X_test)  # Make predictions on the test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy

    mlflow.log_metric('accuracy', accuracy)  # Log accuracy metric
    mlflow.log_param('max_depth', max_depth)  # Log max_depth parameter
    # Log n_estimators parameter
    mlflow.log_param('n_estimators', n_estimators)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))  # Set figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)  # Plot heatmap
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("Confusion-matrix.png")
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'Aaditya', "Project": "Wine Classification"})

    # Log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    print(accuracy)
