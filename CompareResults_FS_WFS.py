import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(r'Code\data.csv')

# Split the dataset into features (X) and target (y)
X = data.drop(columns=['ID', 'class'])
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
knn_classifier = KNeighborsClassifier()
nb_classifier = GaussianNB()
lgbm_classifier = lgb.LGBMClassifier()

# Stacked Ensemble Learning
stacked_classifier = StackingClassifier(estimators=[
    ('rf', rf_classifier),
    ('knn', knn_classifier),
    ('nb', nb_classifier),
    ('lgbm', lgbm_classifier)
], final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))

# Define baseline classifiers and Stacked ensemble classifier
models = {
    "RF": rf_classifier,
    "KNN": knn_classifier,
    "NB": nb_classifier,
    "LGBM": lgbm_classifier,
    "Stacked": stacked_classifier
}

# Without feature selection
results_no_selection = {}

# Without feature selection
results_no_selection = {}

for name, model in models.items():
    if name == "Stacked":
        model.fit(X_train, y_train)  # No feature selection for Stacked
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)  # No feature selection for other models
        y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results_no_selection[name] = accuracy

# Feature selection using Random Forest
rf_feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
X_train_rf_selected = rf_feature_selector.fit_transform(X_train, y_train)
X_test_rf_selected = rf_feature_selector.transform(X_test)

# With feature selection
results_with_selection = {}

for name, model in models.items():
    if name == "Stacked":
        model.fit(X_train_rf_selected, y_train)  # Use RF-selected features for Stacked
        y_pred = model.predict(X_test_rf_selected)
    else:
        model.fit(X_train_rf_selected, y_train)  # Use RF-selected features for other models
        y_pred = model.predict(X_test_rf_selected)
    accuracy = accuracy_score(y_test, y_pred)
    results_with_selection[name] = accuracy

# Display selected features
selected_features = X.columns[rf_feature_selector.get_support()]

# Print results
print("Accuracy without feature selection:")
for name, accuracy in results_no_selection.items():
    print(f"{name}: Accuracy = {accuracy:.4f}")

print("\nAccuracy with feature selection:")
for name, accuracy in results_with_selection.items():
    print(f"{name}: Accuracy = {accuracy:.4f}")

print("\nSelected Features:")
print(selected_features)
