import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import time

# Load your dataset from the CSV file (replace 'data.csv' with your actual file path)
data = pd.read_csv(r'Code\data.csv')

# Extract the feature matrix (X) and target labels (y)
X = data.drop(columns=['class', 'ID'])  # Adjust 'target' to your actual target column name
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dictionary to store the most important feature for each method
most_important_features = {}

# Create DataFrames to store feature importance results
feature_importance_df = pd.DataFrame(index=X.columns)

execution_times = {}

# Method 1: Random Forest Feature Importance
start_time = time.time()
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
most_important_features['Random Forest'] = X_train.columns[np.argmax(rf_classifier.feature_importances_)]
feature_importance_df['Random Forest'] = X_train.columns[np.argmax(rf_classifier.feature_importances_)]
execution_times['Random Forest'] = time.time() - start_time
print("Random Forest Completed")

# Method 2: Extra Trees Feature Importance
start_time = time.time()
et_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_classifier.fit(X_train, y_train)
most_important_features['Extra Trees'] = X_train.columns[np.argmax(et_classifier.feature_importances_)]
feature_importance_df['Extra Trees'] = X_train.columns[np.argmax(et_classifier.feature_importances_)]
execution_times['Extra Trees'] = time.time() - start_time
print("Extra Trees Feature Importance Completed")

# Method 3: Gradient Boosting Feature Importance
start_time = time.time()
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)
most_important_features['Gradient Boosting'] = X_train.columns[np.argmax(gb_classifier.feature_importances_)]
feature_importance_df['Gradient Boosting'] = X_train.columns[np.argmax(gb_classifier.feature_importances_)]
execution_times['Gradient Boosting'] = time.time() - start_time
print("Gradient Boosting Feature Importance Completed")

# Method 4: L1 Regularization (Lasso)
start_time = time.time()
lasso_classifier = LogisticRegression(penalty='l1', solver='liblinear')
lasso_classifier.fit(X_train, y_train)
most_important_features['L1 Regularization (Lasso)'] = X_train.columns[np.argmax(np.abs(lasso_classifier.coef_[0]))]
feature_importance_df['L1 Regularization (Lasso)'] = X_train.columns[np.argmax(np.abs(lasso_classifier.coef_[0]))]
execution_times['L1 Regularization (Lasso)'] = time.time() - start_time
print("L1 Regularization (Lasso) Completed")

# Method 5: Recursive Feature Elimination with Cross-Validation (RFECV)
start_time = time.time()
rfecv_selector = RFECV(estimator=rf_classifier, cv=5)
rfecv_selector.fit(X_train, y_train)
most_important_features['RFECV'] = X_train.columns[np.argmax(rfecv_selector.support_)]
feature_importance_df['RFECV'] = X_train.columns[np.argmax(rfecv_selector.support_)]
execution_times['RFECV'] = time.time() - start_time
print("Recursive Feature Elimination with Cross-Validation (RFECV) Completed")

# Method 6: Permutation Importance
start_time = time.time()
permutation_result = permutation_importance(rf_classifier, X_test, y_test, n_repeats=30, random_state=42)
most_important_features['Permutation Importance'] = X_train.columns[np.argmax(permutation_result.importances_mean)]
feature_importance_df['Permutation Importance'] = X_train.columns[np.argmax(permutation_result.importances_mean)]
execution_times['Permutation Importance'] = time.time() - start_time
print("Permutation Importance Completed")

# Find features that have 0 importance across all methods
zero_importance_features = feature_importance_df.columns[(feature_importance_df == 0).all()]

# Filter the original feature matrix to exclude the zero-importance features
X_filtered = X.drop(columns=zero_importance_features)

# Display the common features with 0 importance
print("Common features with 0 importance across all methods:")
print(zero_importance_features)

# Optionally, save the filtered feature matrix to a new CSV file
X_filtered.to_csv('filtered_data.csv', index=False)

# Plot the most important features for each method
plt.figure(figsize=(10, 6))
methods = most_important_features.keys()
most_important_features_values = [most_important_features[method] for method in methods]
print(most_important_features_values)
plt.bar(methods, most_important_features_values)
plt.xlabel("Feature Importance Method")
plt.ylabel("Most Important Feature")
plt.title("Most Important Feature by Feature Importance Method")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot execution times
methods = most_important_features.keys()
times = [execution_times[method] for method in methods]

plt.figure(figsize=(10, 6))
plt.barh(methods, times)
plt.xlabel("Execution Time (seconds)")
plt.ylabel("Feature Importance Method")
plt.title("Execution Time of Feature Importance Methods")
plt.tight_layout()
plt.show()
