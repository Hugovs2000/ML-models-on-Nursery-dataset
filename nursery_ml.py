import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the Nursery dataset from a local file
file_path = "nursery.data"  # Path to the dataset
column_names = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"]
data = pd.read_csv(file_path, header=None, names=column_names)

# Convert categorical data to numerical data using Label Encoding
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the data into features and target variable
X = data.drop('class', axis=1)  # Features
y = data['class']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models to be used in the project
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),  # Logistic Regression model
    "Random Forest": RandomForestClassifier()  # Random Forest model
}

# Initialize a dictionary to store the results
results = {}

# Create a pipeline that includes scaling and the logistic regression model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic_regression', LogisticRegression(solver='saga', max_iter=5000))
])

# Define a range of 'C' values for hyperparameter tuning
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
accuracies = []

# Train the model and record the accuracy for each 'C' value
for C_val in C_values:
    # Set the 'C' parameter for logistic regression
    pipeline.set_params(logistic_regression__C=C_val)
    
    # Train the pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = pipeline.predict(X_test)
    
    # Calculate and record the accuracy
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot the accuracy for each 'C' value
plt.figure(figsize=(10, 6))
plt.plot(C_values, accuracies, marker='o')
plt.xscale('log')  # Log scale for 'C' values
plt.xlabel('Value of C (Inverse of regularization strength)')
plt.ylabel('Accuracy on the test set')
plt.title('Logistic Regression Accuracy vs. Regularization Strength')
plt.show()

# Identify the best C value and its corresponding accuracy
best_C_index = np.argmax(accuracies)
best_C = C_values[best_C_index]
best_accuracy = accuracies[best_C_index]
print(f"Best C value for Logistic Regression: {best_C} with accuracy: {best_accuracy}")

# Update the Logistic Regression model with the best C value
models['Logistic Regression'] = LogisticRegression(C=best_C, max_iter=1000)

# Define the hyperparameter grid for the RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 150, 200],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at a leaf node
    'bootstrap': [True, False]  # Bootstrap samples when building trees
}

# Initialize GridSearchCV to find the best hyperparameters for RandomForest
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Perform the grid search and fit the model to the training data
grid_search.fit(X_train, y_train)

# Update the RandomForestClassifier in the models dictionary with the best parameters found
models['Random Forest'] = grid_search.best_estimator_

# Documenting the final set of hyperparameters for each model
best_params = {}

# Store the best hyperparameters
best_params['Random Forest'] = grid_search.best_params_
print(f"Best parameters for Random Forest: {best_params['Random Forest']}")

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model on the training data
    y_pred = model.predict(X_test)  # Predict on the test data
    
    # Calculate and store the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = np.mean(precision_score(y_test, y_pred, average=None, zero_division=0))
    recall = np.mean(recall_score(y_test, y_pred, average=None))
    f1 = np.mean(f1_score(y_test, y_pred, average=None, zero_division=0))
    
    results[name] = [accuracy, precision, recall, f1]
    
    # Print a classification report for each model
    print(f"Classification Report for {name}:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoders['class'].classes_, zero_division=0))
    print("-" * 60)

# Convert the results dictionary to a DataFrame for better visualization
results_df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score"]).transpose()
print(results_df)

# Plot confusion matrix for each model
for name, model in models.items():
    y_pred = model.predict(X_test)  # Predict on the test data
    cm = confusion_matrix(y_test, y_pred)  # Calculate the confusion matrix
    
    # Plot the confusion matrix using seaborn for better visualization
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cbar=False, cmap='Blues')
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Plot a histogram to compare accuracy, recall, and F1 score across all models
results_df.plot(kind='bar', figsize=(10, 6))
plt.title("Comparison of Metrics for Each Model")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()
