import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Load the Nursery dataset from a local file
file_path = "nursery.data"  # Replace with your actual path
column_names = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"]
data = pd.read_csv(file_path, header=None, names=column_names)

# Convert categorical data to numerical data
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the data into features and target variable
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV

# Define the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier()
}

results = {}

# Hyperparameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 150, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]  # Method for sampling data points
}

# Initialize GridSearchCV with the RandomForestClassifier and the parameter grid
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Update the RandomForestClassifier in the models dictionary with the best estimator found
models['Random Forest'] = grid_search.best_estimator_

# Output the best parameters
print(f"Best parameters for Random Forest: {grid_search.best_params_}")

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = np.mean(precision_score(y_test, y_pred, average=None, zero_division=0))
    recall = np.mean(recall_score(y_test, y_pred, average=None))
    f1 = np.mean(f1_score(y_test, y_pred, average=None, zero_division=0))
    
    results[name] = [accuracy, precision, recall, f1]
    
    # Print classification report for each model
    print(f"Classification Report for {name}:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoders['class'].classes_, zero_division=0))
    print("-" * 60)

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score"]).transpose()
print(results_df)

# imports for confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Plot confusion matrix for each model
for name, model in models.items():
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cbar=False, cmap='Blues')
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print("-" * 60)

# Display histogram that compares accurracy, recall and f1-score for each model
results_df.plot(kind='bar', figsize=(10, 6))
plt.title("Comparison of Metrics for Each Model")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()




